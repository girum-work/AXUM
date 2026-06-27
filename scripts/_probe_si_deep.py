"""Deep SI search for objects with image media."""
import json
import requests

s = requests.Session()
s.headers["User-Agent"] = "AXUM-Rover/1.0"

r = s.get(
    "https://api.si.edu/openaccess/api/v1.0/search",
    params={"q": "Ethiopia", "api_key": "DEMO_KEY", "rows": 50, "start": 0},
    timeout=30,
)
data = r.json()
print("status", data.get("status"), "keys", list(data.keys()))
resp = data.get("response") or {}
rows = resp.get("rows") or []
print("total rows", len(rows), "rowCount", resp.get("rowCount"))
if not rows:
    print(data)
    raise SystemExit(0)

found = []
for row in rows:
    rid = row["id"]
    r2 = s.get(
        f"https://api.si.edu/openaccess/api/v1.0/content/{rid}",
        params={"api_key": "DEMO_KEY"},
        timeout=30,
    )
    c = r2.json().get("response", {}).get("content", {})
    title = c.get("descriptiveNonRepeating", {}).get("title", {}).get("content", "")
    media = c.get("descriptiveNonRepeating", {}).get("online_media", {}).get("media", [])
    imgs = [m for m in media if m.get("type") == "Images" and m.get("content")]
    if imgs:
        found.append((title, imgs[0]["content"]))
        print("IMG", title[:70])
        print("   ", imgs[0]["content"][:100])

print(f"\nWith images: {len(found)}/{len(rows)}")

# dump one full content if none
if not found and rows:
    rid = rows[0]["id"]
    r2 = s.get(
        f"https://api.si.edu/openaccess/api/v1.0/content/{rid}",
        params={"api_key": "DEMO_KEY"},
        timeout=30,
    )
    print(json.dumps(r2.json(), indent=2)[:3000])
