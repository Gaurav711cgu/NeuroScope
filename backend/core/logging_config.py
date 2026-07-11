"""Structured logger configuration for production environment telemetries."""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone


class JSONFormatter(logging.Formatter):
    """Formats log records as structured single-line JSON objects for log collection utilities."""
    def format(self, record: logging.LogRecord) -> str:
        log_record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_record)


def configure_logging():
    """Setup logging format based on environment preferences."""
    log_level_str = os.environ.get("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)

    handler = logging.StreamHandler()
    if os.environ.get("NEUROSCOPE_JSON_LOGGING") == "1":
        handler.setFormatter(JSONFormatter())
    else:
        # Standard clean human-readable log formatting for local development
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)

    root = logging.getLogger()
    # Remove existing default handlers to avoid double logging
    for h in root.handlers[:]:
        root.removeHandler(h)
        
    root.addHandler(handler)
    root.setLevel(log_level)
