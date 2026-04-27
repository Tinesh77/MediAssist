# ──────────────────────────────────────────────────────────────────
# gunicorn.conf.py
# Place this at your project root (next to manage.py)
# Run with: gunicorn MediAssist.wsgi:application -c gunicorn.conf.py
# ──────────────────────────────────────────────────────────────────

import multiprocessing

# ── Server socket ─────────────────────────────────────────────────
bind            = "0.0.0.0:8000"
backlog         = 64

# ── Workers ───────────────────────────────────────────────────────
# Rule of thumb: (2 × CPU cores) + 1
# For Railway/Render free tier (1 vCPU): 3 workers
workers         = int(multiprocessing.cpu_count() * 2) + 1
worker_class    = "sync"          # use "gevent" if you need async SSE
timeout         = 120             # important: SSE streams can be long
keepalive       = 5
graceful_timeout = 30

# ── Logging ───────────────────────────────────────────────────────
accesslog       = "-"             # stdout → picked up by Railway/Render
errorlog        = "-"
loglevel        = "info"
access_log_format = '%(h)s "%(r)s" %(s)s %(b)s %(D)sµs'

# ── Process naming ────────────────────────────────────────────────
proc_name       = "mediassist"

# ── SSE streaming ─────────────────────────────────────────────────
# For Server-Sent Events to work correctly through gunicorn:
# worker_class = "gevent"   (install: pip install gunicorn[gevent])
# workers      = 1          (gevent handles concurrency internally)
# OR keep sync workers + set X-Accel-Buffering: no header (already done in chat_views.py)