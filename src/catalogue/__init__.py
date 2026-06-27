"""
AXUM ROVER — Catalogue Package
==============================
WHAT:  Persist scanned artefact records as JSON + museum PDF catalogue.
WHY:   Judges see professional output; dashboard reads the same JSON files.

Exports:
    ObjectRecord       — dataclass schema (catalogue/records.py)
    CatalogueGenerator — ReportLab PDF builder (catalogue/generator.py)
    CatalogueService   — register_object() orchestrates JSON + PDF + fragment grouping

Typical usage (from pipeline):
    from src.catalogue.service import CatalogueService
    service = CatalogueService()
    service.register_object(record)
"""

from src.catalogue.generator import CatalogueGenerator
from src.catalogue.records import ObjectRecord
from src.catalogue.service import CatalogueService

__all__ = ["CatalogueGenerator", "ObjectRecord", "CatalogueService"]
