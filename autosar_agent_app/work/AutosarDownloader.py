import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import os

BASE_URL = "https://www.autosar.org"
RELEASE_URL = "https://www.autosar.org/search?tx_solr[filter][1]=platform%3ACP&tx_solr[filter][2]=category%3AR4.4.0&tx_solr[page]={page}&tx_solr[q]="
DOWNLOAD_DIR = "autosar_public_docs_440"

def scrape_public_links():
    print("[*] Fetching AUTOSAR 4.4.0 page...")

    doc_links = []
    for i in range (1, 13):
        response = requests.get(RELEASE_URL.format(page=i))
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, "html.parser")

        # Find all public PDF/ZIP links on the page
        for a in soup.find_all("a", href=True):
            href = a["href"]

            # Only accept public URLs (no login, no protected assets)
            if any(href.lower().endswith(ext) for ext in [".pdf", ".zip"]):
                # Ignore links that clearly require authentication
                if "?" in href or "download" in href.lower():
                    continue

                full_url = urljoin(BASE_URL, href)
                doc_links.append(full_url)

    return doc_links

def download_file(url, folder):
    filename = url.split("/")[-1]
    filepath = os.path.join(folder, filename)

    print(f"[*] Downloading: {filename}")

    r = requests.get(url)
    if r.status_code == 200:
        with open(filepath, "wb") as f:
            f.write(r.content)
    else:
        print(f"[!] Skipped (HTTP {r.status_code}): {filename}")

def main():
    if not os.path.exists(DOWNLOAD_DIR):
        os.makedirs(DOWNLOAD_DIR)

    links = scrape_public_links()

    print(f"[*] Found {len(links)} public documents.")
    print("[*] Starting downloads...\n")

    for link in links:
        download_file(link, DOWNLOAD_DIR)

    print("\n[*] Completed.")

if __name__ == "__main__":
    main()