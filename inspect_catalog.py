import json

with open("catalog.json") as f:
    data = json.load(f)

print(f"Total products: {len(data)}")
print("\nFirst 15 products:")
for p in data[:15]:
    types = ", ".join(p.get("test_type_codes", []))
    print(f"  - {p['name']} | Types: {types}")
    print(f"    URL: {p['url']}")
