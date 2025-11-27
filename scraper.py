import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
import logging
from typing import List, Dict, Optional

# Configure logging to look professional
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class EcommerceScraper:
    """
    A robust scraper to fetch product data including names, prices, 
    and availability from e-commerce targets.
    """
    
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.results: List[Dict] = []

    def get_soup(self, url: str) -> Optional[BeautifulSoup]:
        """Fetches the URL and returns a BeautifulSoup object with error handling."""
        try:
            # Random delay to mimic human behavior (Ethical Scraping)
            time.sleep(random.uniform(1, 3))
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            return BeautifulSoup(response.text, 'html.parser')
        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to fetch {url}: {e}")
            return None

    def parse_product_page(self, url: str):
        """Extracts details from a single product page."""
        soup = self.get_soup(url)
        if not soup:
            return

        # CSS Selectors (Adjust these based on the specific target site)
        products = soup.select('article.product_pod')
        
        for p in products:
            try:
                item = {
                    'name': p.select_one('h3 a')['title'],
                    'price_raw': p.select_one('p.price_color').text,
                    'availability': p.select_one('p.instock.availability').text.strip(),
                    'image_url': self.base_url + p.select_one('img')['src'],
                    'scraped_at': pd.Timestamp.now().isoformat()
                }
                self.results.append(item)
            except AttributeError as e:
                logging.warning(f"Skipping incomplete item: {e}")

    def run(self):
        """Main execution method."""
        logging.info(f"Starting scrape on {self.base_url}")
        # In a real scenario, we would loop through pagination here
        self.parse_product_page(self.base_url)
        
        logging.info(f"Scraped {len(self.results)} items.")
        return self.results

if __name__ == "__main__":
    # Example target (Safe for portfolio use)
    scraper = EcommerceScraper("http://books.toscrape.com/")
    data = scraper.run()
    
    # Save Raw Data
    df = pd.DataFrame(data)
    df.to_csv("raw_products.csv", index=False)
    print("✅ Data Acquisition Complete. Saved to raw_products.csv")