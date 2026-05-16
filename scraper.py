"""
SHL Individual Test Solutions Catalog Scraper
Specifically targets Individual Test Solutions (type=1) NOT pre-packaged job solutions
"""
import requests
import json
import time
from bs4 import BeautifulSoup
BASE_URL = "https://www.shl.com"
CATALOG_URL = "https://www.shl.com/solutions/products/product-catalog/"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.shl.com/solutions/products/",
}
TEST_TYPE_MAP = {
    "A": "Ability & Aptitude",
    "B": "Biodata & Situational Judgement",
    "C": "Competencies",
    "D": "Development & 360",
    "E": "Assessment Exercises",
    "K": "Knowledge & Skills",
    "M": "Motivational",
    "P": "Personality & Behaviour",
    "S": "Simulations",
}
def fetch_page(start: int) -> str | None:
    """Fetch page of Individual Test Solutions catalog."""
    params = {"start": start, "type": 1}
    try:
        resp = requests.get(CATALOG_URL, params=params, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        return resp.text
    except Exception as e:
        print(f"  Error fetching start={start}: {e}")
        return None
def parse_individual_test_solutions(html: str) -> tuple[list[dict], bool]:
    """
    Parse Individual Test Solutions from catalog HTML.
    The page has TWO tables:
      1. Pre-packaged Job Solutions (first table)  
      2. Individual Test Solutions (second table)
    We want ONLY the second table.
    """
    soup = BeautifulSoup(html, "lxml")
    products = []
    tables = soup.find_all("table")
    target_table = None
    for heading in soup.find_all(["h2", "h3", "h4", "div"], 
                                  string=lambda t: t and "individual test" in t.lower() if t else False):
        next_table = heading.find_next("table")
        if next_table:
            target_table = next_table
            break
    if not target_table and len(tables) >= 2:
        target_table = tables[1]
    elif not target_table and len(tables) == 1:
        target_table = tables[0]
    if not target_table:
        print("  No table found!")
        return [], False
    tbody = target_table.find("tbody")
    if not tbody:
        rows = target_table.find_all("tr")[1:]  
    else:
        rows = tbody.find_all("tr")
    for row in rows:
        cells = row.find_all("td")
        if len(cells) < 2:
            continue
        name_cell = cells[0]
        link = name_cell.find("a")
        if not link:
            continue
        name = link.get_text(strip=True)
        href = link.get("href", "")
        if not href:
            continue
        if not href.startswith("http"):
            href = BASE_URL + href
        type_codes = []
        for cell in cells:
            for span in cell.find_all("span"):
                text = span.get_text(strip=True).upper()
                if text in TEST_TYPE_MAP:
                    type_codes.append(text)
        type_codes = list(dict.fromkeys(type_codes))
        remote_testing = False
        adaptive_irt = False
        if len(cells) >= 3:
            remote_cell = cells[1] if len(cells) == 4 else None
            for i, cell in enumerate(cells[1:3], 1):
                has_indicator = bool(
                    cell.find("span", class_=lambda c: c and "yes" in c.lower() if c else False) or
                    cell.find("img") or
                    "yes" in cell.get_text(strip=True).lower() or
                    "\u2713" in cell.get_text() or  
                    cell.find(class_=lambda c: c and ("check" in str(c).lower() or "true" in str(c).lower()) if c else False)
                )
                if i == 1:
                    remote_testing = has_indicator
                elif i == 2:
                    adaptive_irt = has_indicator
        products.append({
            "name": name,
            "url": href,
            "test_type_codes": type_codes,
            "test_types": [TEST_TYPE_MAP.get(t, t) for t in type_codes],
            "remote_testing": remote_testing,
            "adaptive_irt": adaptive_irt,
            "description": "",
        })
    next_section = target_table.find_next(class_=lambda c: c and "pagination" in str(c).lower() if c else False)
    if not next_section:
        next_section = soup
    next_link = next_section.find("a", string=lambda t: t and ("next" in t.lower() or ">" == t.strip()) if t else False)
    if not next_link:
        next_btn = soup.find(attrs={"data-start": True})
        has_next = next_btn is not None
    else:
        has_next = True
    all_pagination = soup.find_all("a", href=lambda h: h and "start=" in str(h) if h else False)
    current_start_links = [a for a in all_pagination]
    has_next = len(current_start_links) > 0
    return products, has_next
def scrape_all():
    """Scrape all Individual Test Solutions."""
    all_products = []
    page_size = 12
    print("=" * 60)
    print("Scraping SHL Individual Test Solutions Catalog")
    print("=" * 60)
    start = 0
    empty_count = 0
    while True:
        page_num = start // page_size + 1
        print(f"\nPage {page_num} (start={start})...")
        html = fetch_page(start)
        if not html:
            empty_count += 1
            if empty_count >= 3:
                print("Too many failures. Stopping.")
                break
            start += page_size
            continue
        products, has_next = parse_individual_test_solutions(html)
        if not products:
            print(f"  No products on page {page_num}.")
            break
        all_products.extend(products)
        print(f"  Got {len(products)} products. Running total: {len(all_products)}")
        if len(products) < page_size or start >= (32 * page_size):
            print("  Reached end of catalog.")
            break
        start += page_size
        time.sleep(0.5)
    seen = set()
    unique = []
    for p in all_products:
        if p["url"] not in seen:
            seen.add(p["url"])
            unique.append(p)
    print(f"\nTotal unique Individual Test Solutions: {len(unique)}")
    return unique
if __name__ == "__main__":
    products = scrape_all()
    with open("catalog.json", "w", encoding="utf-8") as f:
        json.dump(products, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(products)} products to catalog.json")
    print("\nSample products:")
    for p in products[:10]:
        types = ", ".join(p["test_type_codes"]) if p["test_type_codes"] else "N/A"
        print(f"  - {p['name']} | Types: [{types}]")
