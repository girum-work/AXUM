"""Classify Met Ethiopia objects by label rules."""
import requests
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.object_detection.artefact_label_rules import decide_label, LabelAction, build_met_context

s = requests.Session()
s.headers["User-Agent"] = "AXUM-Rover/1.0"
r = s.get(
    "https://collectionapi.metmuseum.org/public/collection/v1/search",
    params={"q": "Ethiopia", "hasImages": "true", "isPublicDomain": "true"},
    timeout=30,
)
ids = r.json().get("objectIDs") or []
by_class = {}
for oid in ids:
    obj = s.get(
        f"https://collectionapi.metmuseum.org/public/collection/v1/objects/{oid}",
        timeout=30,
    ).json()
    text = build_met_context(obj)
    for cls in ["coin", "pottery", "stone_carving", "inscription_fragment", "other"]:
        d = decide_label(text, cls, require_ethiopia=True)
        if d.action != LabelAction.REJECT:
            by_class.setdefault(cls, []).append((oid, obj.get("title", "")[:50]))
            break
    else:
        by_class.setdefault("reject", []).append((oid, obj.get("title", "")[:50]))

for cls, items in sorted(by_class.items()):
    print(f"{cls}: {len(items)}")
    for oid, title in items[:3]:
        print(f"  {oid} {title}")
