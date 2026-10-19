import requests
from urllib.parse import quote

# query = 'Trump (market OR stocks OR tariff OR trade OR sanctions OR oil OR rates OR Fed) AND sourcelang:english'

query = 'Trump AND sourcelang:english'

url = (
    "https://api.gdeltproject.org/api/v2/doc/doc"
    f"?query={quote(query)}"
    "&mode=ArtList"
    "&maxrecords=5"
    "&timespan=5d"
    "&format=json"
)

try:
    response = requests.get(url, timeout=60)
    response.raise_for_status()

    data = response.json()

    for article in data.get("articles", []):
        print("=" * 80)
        print("Title :", article.get("title"))
        print("Date  :", article.get("seendate"))
        print("Source:", article.get("domain"))
        print("URL   :", article.get("url"))

except requests.exceptions.Timeout:
    print("Timeout bei GDELT")
except Exception as e:
    print(f"Fehler: {e}")