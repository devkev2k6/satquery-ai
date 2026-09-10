"""
api/index.py
============
Vercel Serverless Function entrypoint proxy for SatQuery AI.
Exports `app`, `application`, and `handler` from `frontend.app`.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from frontend.app import app, application, handler, BENCHMARK_PRESETS

__all__ = ["app", "application", "handler", "BENCHMARK_PRESETS"]
