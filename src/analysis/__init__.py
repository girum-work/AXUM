"""Analysis modules — fragility prediction, treatment advisory, inscription fingerprinting."""

__all__ = [
    "DiagnosticInputs",
    "TreatmentAdvisor",
    "TreatmentProtocol",
    "run_treatment_advisor",
    "FragmentGrouper",
    "FragmentGroup",
    "compute_match_score",
    "run_fragment_grouping",
]


def __getattr__(name: str):
    """Lazy import so `python -m src.analysis.*` works cleanly."""
    if name in ("DiagnosticInputs", "TreatmentAdvisor", "TreatmentProtocol", "run_treatment_advisor"):
        from src.analysis import treatment_advisor
        return getattr(treatment_advisor, name)
    if name in ("FragmentGrouper", "FragmentGroup", "compute_match_score", "run_fragment_grouping"):
        from src.analysis import fragment_grouper
        return getattr(fragment_grouper, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
