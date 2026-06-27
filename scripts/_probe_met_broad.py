"""Broader Met search counts."""
import requests

s = requests.Session()
s.headers["User-Agent"] = "AXUM-Rover/1.0"
for q in ["Ethiopia", "Ethiopian", "Aksum", "Axum", "Ge'ez", "Ethiopic", "Horn of Africa"]:
    r = s.get(
        "https://collectionapi.metmuseum.org/public/collection/v1/search",
        params={"q": q, "hasImages": "true", "isPublicDomain": "true"},
        timeout=30,
    )
    ids = r.json().get("objectIDs") or []
    print(f"{q!r}: {len(ids)}")
