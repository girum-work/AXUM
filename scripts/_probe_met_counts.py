"""Count Met Ethiopian/Aksumite objects with images per class hint."""
import requests

s = requests.Session()
s.headers["User-Agent"] = "AXUM-Rover/1.0"
queries = [
    "aksumite", "Ethiopia pottery", "Ethiopia coin", "Ethiopia stela",
    "Ethiopia inscription", "Ethiopia cross", "Ethiopia ceramic",
    "Axum", "Ge'ez Ethiopia",
]
for q in queries:
    r = s.get(
        "https://collectionapi.metmuseum.org/public/collection/v1/search",
        params={"q": q, "hasImages": "true", "isPublicDomain": "true"},
        timeout=30,
    )
    ids = r.json().get("objectIDs") or []
    print(f"{q!r}: {len(ids)} objects")
