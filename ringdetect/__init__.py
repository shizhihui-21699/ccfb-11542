"""Offline coupon-fraud ring detection.

The package is deliberately small and stdlib-plus-numpy:

``schema``     entity types, value normalisation
``store``      SQLite state with idempotent daily imports
``ingest``     CSV/JSONL readers and column mapping
``graph``      weighting, projection, Leiden/CPM, scoring
``build``      orchestration and the result-set writer
``evaluate``   recall/precision against a ground-truth manifest
``generate``   synthetic data with planted rings and legitimate sharers
``webapp``     read-only HTTP API + viewer
"""

__version__ = "1.0.0"
