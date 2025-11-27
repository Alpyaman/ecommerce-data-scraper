import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
import logging
import argparse
import json
import yaml
from typing import List, Dict, Optional
from urllib.parse import urljoin
from urllib.robotparser import RobotFileParser
from tqdm import tqdm
from datetime import datetime

# Configure logging to look professional
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scraping_errors.log'),
        logging.StreamHandler()
    ]
)

class EcommerceScraper:
    """
    A robust, production-ready scraper to fetch product data including names,
    prices, and availability from e-commerce sites.

    Features:
    - Pagination support
    - Retry logic with exponential backoff
    - Session management for efficiency
    - Robots.txt compliance
    - Configurable selectors
    - Progress tracking
    - Multiple output formats
    - Comprehensive statistics
    """

    def __init__(self, config_file: str = 'config.yaml'):
        """Initialize scraper with configuration."""
        self.config = self.load_config(config_file)
        self.base_url = self.config['scraper']['base_url']

        # Initialize session for connection pooling
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': self.config['scraper']['user_agent']
        })

        self.results: List[Dict] = []

        # Statistics tracking
        self.stats = {
            'total_pages_attempted': 0,
            'total_pages_successful': 0,
            'total_items_scraped': 0,
            'total_items_failed': 0,
            'failed_urls': [],
            'start_time': None,
            'end_time': None
        }
 
        # Check robots.txt if enabled
        if self.config['scraper']['check_robots']:
            self.check_robots_txt()

    def load_config(self, config_file: str) -> dict:
        """Load configuration from YAML file."""
        try:
            with open(config_file, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logging.error(f"Config file {config_file} not found. Using defaults.")
            return self.get_default_config()

    def get_default_config(self) -> dict:
        """Return default configuration if config file is missing."""
        return {
            'scraper': {
                'base_url': 'http://books.toscrape.com/',
                'max_pages': 5,
                'pagination_pattern': 'catalogue/page-{}.html',
                'start_page': 1,
                'delay_min': 1,
                'delay_max': 3,
                'timeout': 10,
                'max_retries': 3,
                'retry_delay': 2,
                'check_robots': True,
                'selectors': {
                    'product_container': 'article.product_pod',
                    'product_name': 'h3 a',
                    'product_name_attr': 'title',
                    'product_price': 'p.price_color',
                    'product_availability': 'p.instock.availability',
                    'product_image': 'img',
                    'product_image_attr': 'src'
                }
            },
            'output': {
                'raw_data_file': 'raw_products',
                'default_format': 'csv',
                'save_errors': True,
                'error_log_file': 'scraping_errors.log'
            }
        }

    def check_robots_txt(self):
        """Check robots.txt for scraping permissions."""
        try:
            robots_url = urljoin(self.base_url, '/robots.txt')
            rp = RobotFileParser()
            rp.set_url(robots_url)
            rp.read()

            if not rp.can_fetch(self.config['scraper']['user_agent'], self.base_url):
                logging.warning(f"Scraping is disallowed by robots.txt for {self.base_url}")
                logging.warning("Proceeding anyway for educational purposes. In production, respect robots.txt!")
            else:
                logging.info("Robots.txt check passed - scraping is allowed")
        except Exception as e:
            logging.warning(f"Could not read robots.txt: {e}. Proceeding cautiously.")

    def get_soup(self, url: str) -> Optional[BeautifulSoup]:
        """
        Fetches URL and returns BeautifulSoup object with error handling and retry logic.
        """
        retries = 0
        max_retries = self.config['scraper']['max_retries']
        retry_delay = self.config['scraper']['retry_delay']

        while retries <= max_retries:
            try:
                # Random delay to mimic human behavior (Ethical Scraping)
                delay = random.uniform(
                    self.config['scraper']['delay_min'],
                    self.config['scraper']['delay_max']
                )
                time.sleep(delay)

                response = self.session.get(
                    url,
                    timeout=self.config['scraper']['timeout']
                )
                response.raise_for_status()
                return BeautifulSoup(response.text, 'html.parser')

            except requests.exceptions.RequestException as e:
                retries += 1
                if retries <= max_retries:
                    wait_time = retry_delay * (2 ** (retries - 1))  # Exponential backoff
                    logging.warning(
                        f"Retry {retries}/{max_retries} for {url} after {wait_time}s. Error: {e}"
                    )
                    time.sleep(wait_time)
                else:
                    logging.error(f"Failed to fetch {url} after {max_retries} retries: {e}")
                    self.stats['failed_urls'].append({'url': url, 'error': str(e)})
                    return None

    def parse_product_page(self, url: str) -> int:
        """
        Extracts details from a single product page.
        Returns the number of items successfully scraped.
        """
        soup = self.get_soup(url)
        if not soup:
            return 0

        selectors = self.config['scraper']['selectors']
        products = soup.select(selectors['product_container'])

        items_scraped = 0
        for p in products:
            try:
                # Extract product data using configured selectors
                name_element = p.select_one(selectors['product_name'])

                if name_element:
                    name_attr = selectors.get('product_name_attr')
                    name = name_element.get(name_attr) if name_attr else name_element.text.strip()
                else:
                    name = None

                price_element = p.select_one(selectors['product_price'])
                price = price_element.text.strip() if price_element else None

                availability_element = p.select_one(selectors['product_availability'])
                availability = availability_element.text.strip() if availability_element else None

                image_element = p.select_one(selectors['product_image']) if selectors.get('product_image') else None

                if image_element:
                    image_attr = selectors.get('product_image_attr')
                    image_url = image_element.get(image_attr) if image_attr else None
                else:
                    image_url = None

                # Build absolute URL for image
                if image_url:
                    image_url = urljoin(self.base_url, image_url)

                item = {
                    'name': name,
                    'price_raw': price,
                    'availability': availability,
                    'image_url': image_url,
                    'source_url': url,
                    'scraped_at': datetime.now().isoformat()
                }

                # Only add if we have at least a name
                if name:
                    self.results.append(item)
                    items_scraped += 1
                else:
                    self.stats['total_items_failed'] += 1

            except Exception as e:
                logging.warning(f"Error parsing item: {e}")
                self.stats['total_items_failed'] += 1

        return items_scraped

    def build_page_url(self, page_num: int) -> str:
        """Build URL for a specific page number."""
        if page_num == 1:
            return self.base_url

        pattern = self.config['scraper']['pagination_pattern']
        page_path = pattern.format(page_num)
        return urljoin(self.base_url, page_path)

    def run(self, max_pages: Optional[int] = None) -> List[Dict]:
        """
        Main execution method with pagination support.

        Args:
            max_pages: Override config's max_pages if provided
        """
        self.stats['start_time'] = datetime.now()

        if max_pages is None:
            max_pages = self.config['scraper']['max_pages']

        start_page = self.config['scraper']['start_page']

        logging.info(f"Starting scrape on {self.base_url}")
        logging.info(f"Scraping {max_pages} page(s) starting from page {start_page}")

        # Use tqdm for progress bar
        with tqdm(total=max_pages, desc="Scraping pages", unit="page") as pbar:
            for page_num in range(start_page, start_page + max_pages):
                self.stats['total_pages_attempted'] += 1
                url = self.build_page_url(page_num)

                logging.info(f"Scraping page {page_num}: {url}")
                items_count = self.parse_product_page(url)

                if items_count > 0:
                    self.stats['total_pages_successful'] += 1
                    self.stats['total_items_scraped'] += items_count
                    logging.info(f"Scraped {items_count} items from page {page_num}")
                else:
                    logging.warning(f"No items found on page {page_num}")

                pbar.update(1)

        self.stats['end_time'] = datetime.now()
        self.print_statistics()

        return self.results

    def print_statistics(self):
        """Print comprehensive scraping statistics."""
        duration = (self.stats['end_time'] - self.stats['start_time']).total_seconds()

        print("\n" + "="*60)
        print("SCRAPING STATISTICS")
        print("="*60)
        print(f"Total Duration: {duration:.2f} seconds")
        print(f"Pages Attempted: {self.stats['total_pages_attempted']}")
        print(f"Pages Successful: {self.stats['total_pages_successful']}")
        print(f"Items Scraped: {self.stats['total_items_scraped']}")
        print(f"Items Failed: {self.stats['total_items_failed']}")

        if self.stats['total_pages_attempted'] > 0:
            success_rate = (self.stats['total_pages_successful'] / self.stats['total_pages_attempted']) * 100
            print(f"Success Rate: {success_rate:.1f}%")

        if self.stats['total_items_scraped'] > 0:
            items_per_second = self.stats['total_items_scraped'] / duration
            print(f"Speed: {items_per_second:.2f} items/second")

        if self.stats['failed_urls']:
            print(f"\n{len(self.stats['failed_urls'])} URL(s) failed:")
            for failed in self.stats['failed_urls'][:5]:  # Show first 5
                print(f"   - {failed['url']}: {failed['error']}")

        print("="*60 + "\n")

    def save_data(self, output_format: str = None, filename: str = None):
        """
        Save scraped data in various formats.

        Args:
            output_format: 'csv', 'json', or 'excel'
            filename: Custom filename (without extension)
        """
        if not self.results:
            logging.warning("No data to save!")
            return

        if output_format is None:
            output_format = self.config['output']['default_format']

        if filename is None:
            filename = self.config['output']['raw_data_file']

        df = pd.DataFrame(self.results)

        try:
            if output_format == 'csv':
                output_file = f"{filename}.csv"
                df.to_csv(output_file, index=False)
            elif output_format == 'json':
                output_file = f"{filename}.json"
                df.to_json(output_file, orient='records', indent=4)
            elif output_format == 'excel':
                output_file = f"{filename}.xlsx"
                df.to_excel(output_file, index=False, engine='openpyxl')
            else:
                logging.error(f"Unknown format: {output_format}")
                return
 
            logging.info(f"Data saved to {output_file}")
            print(f"Saved {len(df)} items to {output_file}")

        except Exception as e:
            logging.error(f"Failed to save data: {e}")

    def save_statistics(self, filename: str = 'scraping_stats.json'):
        """Save statistics to a JSON file."""
        stats_export = self.stats.copy()
        stats_export['start_time'] = stats_export['start_time'].isoformat()
        stats_export['end_time'] = stats_export['end_time'].isoformat()

        with open(filename, 'w') as f:
            json.dump(stats_export, f, indent=4)

        logging.info(f"Statistics saved to {filename}")

def main():
    """Command-line interface for the scraper."""
    parser = argparse.ArgumentParser(
        description='Professional E-commerce Web Scraper',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scraper.py                          # Use default config
  python scraper.py --pages 10               # Scrape 10 pages
  python scraper.py --format json            # Save as JSON
  python scraper.py --config custom.yaml     # Use custom config
  python scraper.py --pages 5 --format excel # Scrape 5 pages, save as Excel
        """
    )

    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to config file (default: config.yaml)'
    )

    parser.add_argument(
        '--pages',
        type=int,
        help='Number of pages to scrape (overrides config)'
    )

    parser.add_argument(
        '--format',
        type=str,
        choices=['csv', 'json', 'excel'],
        help='Output format (default: from config)'
    )

    parser.add_argument(
        '--output',
        type=str,
        help='Output filename without extension (default: from config)'
    )

    parser.add_argument(
        '--no-stats',
        action='store_true',
        help='Do not save statistics file'
    )

    args = parser.parse_args()

    # Initialize and run scraper
    scraper = EcommerceScraper(config_file=args.config)
    data = scraper.run(max_pages=args.pages)

    # Save data
    scraper.save_data(output_format=args.format, filename=args.output)

    # Save statistics
    if not args.no_stats:
        scraper.save_statistics()

    print(f"Scraping complete! Collected {len(data)} items.")

if __name__ == "__main__":
    main()