"""Temporary probe — delete after museum download verified."""
import json
import requests

s = requests.Session()
s.headers["User-Agent"] = "AXUM-Rover/1.0"

r = s.get(
    "https://api.si.edu/openaccess/api/v1.0/search",
    params={"q": "Aksum coin", "api_key": "DEMO_KEY", "rows": 1},
    timeout=30,
)
row = r.json()["response"]["rows"][0]
print("SI id", row["id"])
r2 = s.get(
    f"https://api.si.edu/openaccess/api/v1.0/content/{row['id']}",
    params={"api_key": "DEMO_KEY"},
    timeout=30,
)
c = r2.json()["response"]["content"]
title = c.get("descriptiveNonRepeating", {}).get("title", {}).get("content", "")
print("SI title", title)
media = c.get("descriptiveNonRepeating", {}).get("online_media", {}).get("media", [])
print("SI media", len(media))
for m in media[:2]:
    print(" ", m.get("type"), (m.get("content") or "")[:120])

r3 = s.get(
    "https://collectionapi.metmuseum.org/public/collection/v1/objects/317877",
    timeout=30,
)
o = r3.json()
print("Met", o.get("title"), "|", o.get("culture"), o.get("country"))
print("Met img", (o.get("primaryImage") or "")[:100])
