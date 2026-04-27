"""
services/pdf_extractor.py  — with OCR fallback for scanned PDFs
"""
from __future__ import annotations
import logging, re, sys
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)
MIN_CHARS = 50

@dataclass
class PageData:
    page_number: int
    text: str
    tables: list[str] = field(default_factory=list)
    char_start: int = 0
    char_end: int = 0
    word_count: int = 0
    is_empty: bool = False
    ocr_used: bool = False
    def __post_init__(self):
        self.word_count = len(self.text.split()) if self.text else 0

@dataclass
class ExtractionResult:
    pages: list[PageData]
    full_text: str
    page_count: int
    char_count: int
    has_text: bool
    ocr_used: bool = False
    empty_pages: list[int] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

_LIGATURE_MAP = str.maketrans({
    "\ufb00":"ff","\ufb01":"fi","\ufb02":"fl","\ufb03":"ffi","\ufb04":"ffl",
    "\u2018":"'","\u2019":"'","\u201c":'"',"\u201d":'"',
    "\u2013":"-","\u2014":"-","\u00ad":"","\u00a0":" ",
})

def _clean(text):
    if not text: return ""
    text = text.translate(_LIGATURE_MAP)
    text = re.sub(r"-\n\s*","",text)
    text = re.sub(r"[ \t]+"," ",text)
    text = re.sub(r"\n{3,}","\n\n",text)
    text = re.sub(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]","",text)
    return text.strip()

def _clean_ocr(text):
    if not text: return ""
    lines=[]
    for line in text.split("\n"):
        line=line.strip()
        if not line: lines.append(""); continue
        if len(line)<=1 and not line.isalnum(): continue
        line=re.sub(r"[|]{3,}","",line)
        line=re.sub(r"\s{3,}","  ",line)
        lines.append(line)
    return _clean("\n".join(lines))

def _table_to_text(table):
    if not table: return ""
    rows=[]
    for row in table:
        cells=[str(c).strip() if c else "" for c in row]
        if any(cells): rows.append("| "+" | ".join(cells)+" |")
    return "\n".join(rows)

def _find_repeated(page_texts):
    from collections import Counter
    thr=max(2,len(page_texts)*0.4)
    cnt=Counter()
    for t in page_texts:
        for ln in set(t.split("\n")):
            s=ln.strip()
            if s and len(s)<80: cnt[s]+=1
    return {ln for ln,n in cnt.items() if n>=thr}

class OCREngine:
    _ok=None
    @classmethod
    def available(cls):
        if cls._ok is not None: return cls._ok
        try:
            import fitz, pytesseract
            from PIL import Image

            # Apply tesseract_cmd from Django settings if configured.
            # This must happen BEFORE get_tesseract_version() is called,
            # otherwise pytesseract looks for 'tesseract' on PATH which
            # fails on Windows where tesseract is installed to Program Files.
            try:
                from django.conf import settings
                cmd = getattr(settings, "TESSERACT_CMD", None)
                if cmd:
                    pytesseract.pytesseract.tesseract_cmd = cmd
                elif hasattr(pytesseract.pytesseract, "tesseract_cmd"):
                    # Already set directly in settings.py via
                    # pytesseract.pytesseract.tesseract_cmd = r"C:\..."
                    pass   # nothing to do — it's already applied
            except Exception:
                pass   # Django not configured — standalone mode

            pytesseract.get_tesseract_version()
            cls._ok=True
            logger.info("[OCREngine] Ready")
        except Exception as e:
            cls._ok=False
            logger.warning(f"[OCREngine] Not available: {e}")
        return cls._ok

    @classmethod
    def ocr_page(cls, pdf_path, page_idx):
        import fitz, pytesseract, io
        from PIL import Image

        # Ensure tesseract path is set (same logic as available())
        try:
            from django.conf import settings
            cmd = getattr(settings, "TESSERACT_CMD", None)
            if cmd:
                pytesseract.pytesseract.tesseract_cmd = cmd
        except Exception:
            pass

        try:
            doc=fitz.open(pdf_path)
            pix=doc[page_idx].get_pixmap(matrix=fitz.Matrix(2.5,2.5),colorspace=fitz.csRGB)
            doc.close()
            img=Image.open(io.BytesIO(pix.tobytes("png")))
            raw=pytesseract.image_to_string(img,lang="eng",config="--oem 3 --psm 6")
            return _clean_ocr(raw)
        except Exception as e:
            logger.warning(f"[OCREngine] page {page_idx+1} failed: {e}")
            return ""

class PDFExtractor:
    def __init__(self, file_path, remove_headers=True, extract_tables=True,
                 min_chars_per_page=MIN_CHARS, force_ocr=False):
        self.file_path=Path(file_path)
        self.remove_headers=remove_headers
        self.extract_tables=extract_tables
        self.min_chars=min_chars_per_page
        self.force_ocr=force_ocr
        if not self.file_path.exists():
            raise FileNotFoundError(f"PDF not found: {self.file_path}")

    def extract(self):
        try:
            import pdfplumber
        except ImportError:
            raise ImportError("pip install pdfplumber")

        path_str=str(self.file_path)
        pages_data=[];warnings=[];empty_pages=[];any_ocr=False;offset=0

        logger.info(f"[PDFExtractor] {self.file_path.name}")

        with pdfplumber.open(self.file_path) as pdf:
            total=len(pdf.pages)
            # First pass: raw texts for header detection
            raw_texts=[]
            for page in pdf.pages:
                try: raw_texts.append(page.extract_text() or "")
                except: raw_texts.append("")

            repeated=set()
            if self.remove_headers and total>2:
                repeated=_find_repeated(raw_texts)

            # Which pages need OCR?
            needs_ocr=[self.force_ocr or len(t.strip())<self.min_chars for t in raw_texts]
            ocr_count=sum(needs_ocr)

            if ocr_count>0 and not OCREngine.available():
                warnings.append(
                    f"{ocr_count} page(s) have no selectable text and OCR is unavailable. "
                    f"Install: pip install pymupdf pytesseract pillow + tesseract binary."
                )
            elif ocr_count>0:
                logger.info(f"[PDFExtractor] {ocr_count}/{total} pages need OCR")
                any_ocr=True

            # Per-page extraction
            for idx, page in enumerate(pdf.pages):
                pnum=idx+1

                if needs_ocr[idx] and OCREngine.available():
                    # OCR path
                    text=OCREngine.ocr_page(path_str, idx)
                    tables=[]
                    ocr_flag=True
                else:
                    # pdfplumber path
                    text, tables = self._plumber_page(page, repeated)
                    ocr_flag=False

                full=("\n\n".join([text]+tables) if tables else text)
                is_empty=len(full.strip())<self.min_chars
                if is_empty: empty_pages.append(pnum)

                cs=offset; ce=cs+len(full); offset=ce+1
                pages_data.append(PageData(
                    page_number=pnum, text=full, tables=tables,
                    char_start=cs, char_end=ce,
                    is_empty=is_empty, ocr_used=ocr_flag
                ))

        full_text="\n".join(p.text for p in pages_data if p.text.strip())
        has_text=bool(full_text.strip())

        if not has_text:
            warnings.append(
                "No text extracted. " +
                ("OCR ran but produced no usable text — scan quality may be too low."
                 if any_ocr else
                 "PDF is image-based. Install tesseract+pymupdf to enable OCR.")
            )
        if any_ocr:
            ocr_pages=[p.page_number for p in pages_data if p.ocr_used]
            warnings.append(
                f"OCR used on pages {ocr_pages}. Verify critical numeric values."
            )
        if empty_pages:
            warnings.append(f"Empty pages: {empty_pages}")

        logger.info(
            f"[PDFExtractor] Done: {len(pages_data)}p "
            f"{len(full_text):,}c ocr={any_ocr}"
        )
        return ExtractionResult(
            pages=pages_data, full_text=full_text,
            page_count=len(pages_data), char_count=len(full_text),
            has_text=has_text, ocr_used=any_ocr,
            empty_pages=empty_pages, warnings=warnings
        )

    def _plumber_page(self, page, repeated):
        tables=[]; bboxes=[]
        if self.extract_tables:
            try:
                for tbl in (page.extract_tables() or []):
                    t=_table_to_text(tbl)
                    if t.strip(): tables.append(t)
                for tf in (page.find_tables() or []): bboxes.append(tf.bbox)
            except: pass
        try:
            if bboxes:
                rem=page
                for b in bboxes:
                    try: rem=rem.outside_bbox(b)
                    except: pass
                raw=rem.extract_text() or ""
            else:
                raw=page.extract_text() or ""
        except: raw=""
        if repeated and raw:
            raw="\n".join(ln for ln in raw.split("\n") if ln.strip() not in repeated)
        return _clean(raw), tables


def extract_from_document(document, force_ocr=False):
    extractor=PDFExtractor(document.file.path, force_ocr=force_ocr)
    result=extractor.extract()
    document.page_count=result.page_count
    document.save(update_fields=["page_count"])
    if result.warnings:
        logger.warning(f"[extract_from_document] '{document.title}': "+" | ".join(result.warnings))
    return result


if __name__=="__main__":
    if len(sys.argv)<2:
        print("Usage: python pdf_extractor.py <path> [--force-ocr]")
        sys.exit(1)
    path=sys.argv[1]
    force="--force-ocr" in sys.argv
    print(f"OCR available: {OCREngine.available()}")
    result=PDFExtractor(path,force_ocr=force).extract()
    print(f"Pages:{result.page_count} Chars:{result.char_count:,} OCR:{result.ocr_used} Text:{result.has_text}")
    for p in result.pages:
        print(f"  P{p.page_number}({'OCR' if p.ocr_used else 'native'}) {p.word_count}w: {p.text[:200].replace(chr(10),' ')}")
    for w in result.warnings:
        print(f"  ! {w}")