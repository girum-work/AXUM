"""Find SI objects with images for Ethiopia artefacts."""
import json
import requests

s = requests.Session()
s.headers["User-Agent"] = "AXUM-Rover/1.0"

queries = [
    "Ethiopia coin",
    "Aksum",
    "Ethiopia pottery",
    "Ethiopia stela",
    "Ethiopia cross",
    "Ethiopia inscription",
]

for q in queries:
    r = s.get(
        "https://api.si.edu/openaccess/api/v1.0/search",
        params={"q": q, "api_key": "DEMO_KEY", "rows": 10},
        timeout=30,
    )
    rows = r.json().get("response", {}).get("rows", [])
    with_img = 0
    for row in rows[:10]:
        rid = row.get("id", "")
        r2 = s.get(
            f"https://api.si.edu/openaccess/api/v1.0/content/{rid}",
            params={"api_key": "DEMO_KEY"},
            timeout=30,
        )
        c = r2.json().get("response", {}).get("content", {})
        title = (
            c.get("descriptiveNonRepeating", {})
            .get("title", {})
            .get("content", "")
        )
        media = (
            c.get("descriptiveNonRepeating", {})
            .get("online_media", {})
            .get("media", [])
        )
        imgs = [m for m in media if m.get("type") == "Images" and m.get("content")]
        if imgs:
            with_img += 1
            print(f"{q!r}: {title[:60]} | {imgs[0]['content'][:80]}")
    print(f"  -> {with_img}/10 with images for query {q!r}\n")
