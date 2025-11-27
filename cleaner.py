import pandas as pd
import json
import re
import yaml
import logging
import argparse
from typing import Dict, List, Optional
from datetime import datetime
import unicodedata

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('cleaning_errors.log'),
        logging.StreamHandler()
    ]
)

class DataCleaner:
    """
    A professional data cleaning and validation tool for scraped e-commerce data.

    Features:
    - Duplicate detection and removal
    - Text normalization (unicode, whitespace)
    - Missing data handling with multiple strategies
    - Data quality reporting
    - Configurable validation rules
    - Multiple output formats
    - Comprehensive statistics
    """

    def __init__(self, config_file: str = 'cleaner_config.yaml'):
        """Initialize cleaner with configuration."""
        self.config = self.load_config(config_file)
        self.stats = {
            'initial_records': 0,
            'final_records': 0,
            'duplicates_removed': 0,
            'invalid_prices_fixed': 0,
            'missing_data_handled': 0,
            'text_normalized': 0,
            'validation_failures': 0,
            'start_time': None,
            'end_time': None
        }
        self.quality_issues: List[Dict] = []

    def load_config(self, config_file: str) -> dict:
        """Load configuration from YAML file."""
        try:
            with open(config_file, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logging.warning(f"Config file {config_file} not found. Using defaults.")
            return self.get_default_config()

    def get_default_config(self) -> dict:
        """Return default configuration."""
        return {
            'cleaning': {
                'remove_duplicates': True,
                'duplicate_subset': ['name', 'price'],
                'normalize_text': True,
                'handle_missing': True,
                'missing_strategy': 'flag',  # Options: 'drop', 'flag', 'fill'
                'min_price': 0.01,
                'max_price': 100000,
                'required_fields': ['name']
            },
            'categorization': {
                'enabled': True,
                'default_category': 'General',
                'keywords': {
                    'Electronics': ['laptop', 'phone', 'computer', 'monitor', 'keyboard', 'mouse'],
                    'Gaming': ['game', 'console', 'controller', 'gaming'],
                    'Accessories': ['cable', 'case', 'adapter', 'charger', 'stand'],
                    'Audio': ['headphone', 'speaker', 'earphone', 'audio', 'microphone']
                }
            },
            'output': {
                'default_format': 'json',
                'save_quality_report': True,
                'quality_report_file': 'data_quality_report.json'
            }
        }

    def clean_currency(self, price_str: str) -> Optional[float]:
        """
        Converts currency strings to float.
        Example: '£51.77' -> 51.77
        """
        if pd.isna(price_str):
            return None

        try:
            # Remove currency symbols and convert to float
            clean_price = re.sub(r'[^\d.]', '', str(price_str))
            if not clean_price:
                return None

            price = float(clean_price)

            # Validate price range
            min_price = self.config['cleaning']['min_price']
            max_price = self.config['cleaning']['max_price']

            if price < min_price or price > max_price:
                self.quality_issues.append({
                    'type': 'invalid_price',
                    'value': price,
                    'reason': f'Price outside valid range ({min_price}-{max_price})'
                })
                return None

            return price

        except (ValueError, TypeError) as e:
            logging.warning(f"Failed to parse price '{price_str}': {e}")
            return None

    def normalize_text(self, text: str) -> str:
        """
        Normalize text by:
        - Removing extra whitespace
        - Normalizing unicode characters
        - Stripping leading/trailing spaces
        """
        if pd.isna(text):
            return ""

        # Convert to string
        text = str(text)

        # Normalize unicode (e.g., é -> e)
        text = unicodedata.normalize('NFKD', text)
        text = text.encode('ascii', 'ignore').decode('ascii')

        # Remove extra whitespace
        text = ' '.join(text.split())

        # Strip leading/trailing spaces
        text = text.strip()

        return text

    def categorize_product(self, name: str) -> str:
        """
        Categorize products based on keywords.
        More sophisticated than the original version.
        """
        if not self.config['categorization']['enabled']:
            return self.config['categorization']['default_category']

        name_lower = name.lower()
        keywords = self.config['categorization']['keywords']

        # Check each category
        for category, category_keywords in keywords.items():
            for keyword in category_keywords:
                if keyword.lower() in name_lower:
                    return category

        return self.config['categorization']['default_category']

    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate records based on configured fields."""
        if not self.config['cleaning']['remove_duplicates']:
            return df

        initial_count = len(df)
        subset = self.config['cleaning']['duplicate_subset']

        # Remove duplicates, keeping first occurrence
        df_clean = df.drop_duplicates(subset=subset, keep='first')

        duplicates_removed = initial_count - len(df_clean)
        self.stats['duplicates_removed'] = duplicates_removed

        if duplicates_removed > 0:
            logging.info(f"Removed {duplicates_removed} duplicate records")

        return df_clean

    def handle_missing_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing data based on configured strategy."""
        if not self.config['cleaning']['handle_missing']:
            return df
 
        strategy = self.config['cleaning']['missing_strategy']
        required_fields = self.config['cleaning']['required_fields']

        initial_count = len(df)

        if strategy == 'drop':
            # Drop rows with missing required fields
            df_clean = df.dropna(subset=required_fields)

        elif strategy == 'flag':
            # Add a flag column for missing data
            df_clean = df.copy()
            df_clean['has_missing_data'] = df_clean[required_fields].isna().any(axis=1)

        elif strategy == 'fill':
            # Fill missing values with defaults
            df_clean = df.copy()
            for field in required_fields:
                if field in df_clean.columns:
                    df_clean[field].fillna('Unknown', inplace=True)
        else:
            df_clean = df

        missing_handled = initial_count - len(df_clean)
        self.stats['missing_data_handled'] = missing_handled

        if missing_handled > 0:
            logging.info(f"Handled {missing_handled} records with missing data")

        return df_clean

    def validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate data quality and filter invalid records."""
        initial_count = len(df)

        # Check required fields exist
        required_fields = self.config['cleaning']['required_fields']
        for field in required_fields:
            if field not in df.columns:
                logging.warning(f"Required field '{field}' not found in data")

        # Filter out records with invalid data
        valid_mask = pd.Series([True] * len(df))

        # Name should not be empty
        if 'name' in df.columns:
            name_valid = df['name'].notna() & (df['name'].str.strip() != '')
            valid_mask &= name_valid

        # Price should be positive if it exists
        if 'price' in df.columns:
            price_valid = (df['price'].isna()) | (df['price'] > 0)
            valid_mask &= price_valid

        df_valid = df[valid_mask]

        validation_failures = initial_count - len(df_valid)
        self.stats['validation_failures'] = validation_failures

        if validation_failures > 0:
            logging.warning(f"Removed {validation_failures} records that failed validation")

        return df_valid

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main cleaning pipeline.
        Applies all cleaning operations in sequence.
        """
        logging.info(f"Starting data cleaning pipeline with {len(df)} records")

        # 1. Clean price field FIRST (so we can use it for deduplication)
        if 'price_raw' in df.columns:
            df['price'] = df['price_raw'].apply(self.clean_currency)
            self.stats['invalid_prices_fixed'] = df['price'].isna().sum()

        # 2. Normalize text fields
        if self.config['cleaning']['normalize_text']:
            text_fields = ['name', 'availability']
            for field in text_fields:
                if field in df.columns:
                    df[field] = df[field].apply(self.normalize_text)
                    self.stats['text_normalized'] += 1

        # 3. Remove duplicates (after price is cleaned and text is normalized)
        df = self.remove_duplicates(df)

        # 4. Categorize products
        if 'name' in df.columns:
            df['category'] = df['name'].apply(self.categorize_product)

        # 5. Handle missing data
        df = self.handle_missing_data(df)

        # 6. Validate data
        df = self.validate_data(df)

        # 7. Add metadata
        df['cleaned_at'] = datetime.now().isoformat()

        logging.info(f"Cleaning complete. {len(df)} records remain")

        return df

    def generate_quality_report(self) -> Dict:
        """Generate comprehensive data quality report."""
        duration = (self.stats['end_time'] - self.stats['start_time']).total_seconds()

        report = {
            'summary': {
                'initial_records': self.stats['initial_records'],
                'final_records': self.stats['final_records'],
                'records_removed': self.stats['initial_records'] - self.stats['final_records'],
                'data_retention_rate': (self.stats['final_records'] / self.stats['initial_records'] * 100) if self.stats['initial_records'] > 0 else 0,
                'processing_time_seconds': duration
            },
            'cleaning_stats': {
                'duplicates_removed': self.stats['duplicates_removed'],
                'invalid_prices_fixed': self.stats['invalid_prices_fixed'],
                'missing_data_handled': self.stats['missing_data_handled'],
                'text_fields_normalized': self.stats['text_normalized'],
                'validation_failures': self.stats['validation_failures']
            },
            'quality_issues': self.quality_issues[:100],  # Limit to first 100
            'timestamp': datetime.now().isoformat()
        }

        return report

    def print_statistics(self):
        """Print cleaning statistics to console."""
        print("\n" + "="*60)
        print("DATA CLEANING STATISTICS")
        print("="*60)
        print(f"Initial Records: {self.stats['initial_records']}")
        print(f"Final Records: {self.stats['final_records']}")
        print(f"Records Removed: {self.stats['initial_records'] - self.stats['final_records']}")

        if self.stats['initial_records'] > 0:
            retention = (self.stats['final_records'] / self.stats['initial_records']) * 100
            print(f"Data Retention Rate: {retention:.1f}%")

        print("\nCleaning Operations:")
        print(f"  - Duplicates Removed: {self.stats['duplicates_removed']}")
        print(f"  - Invalid Prices Fixed: {self.stats['invalid_prices_fixed']}")
        print(f"  - Missing Data Handled: {self.stats['missing_data_handled']}")
        print(f"  - Validation Failures: {self.stats['validation_failures']}")

        duration = (self.stats['end_time'] - self.stats['start_time']).total_seconds()
        print(f"\nProcessing Time: {duration:.2f} seconds")
        print("="*60 + "\n")

    def save_data(self, df: pd.DataFrame, output_file: str, format: str = None):
        """Save cleaned data in specified format."""
        if format is None:
            format = self.config['output']['default_format']

        try:
            if format == 'csv':
                output_path = f"{output_file}.csv" if not output_file.endswith('.csv') else output_file
                df.to_csv(output_path, index=False)

            elif format == 'json':
                output_path = f"{output_file}.json" if not output_file.endswith('.json') else output_file
                records = df.to_dict(orient='records')
                with open(output_path, 'w') as f:
                    json.dump(records, f, indent=4)

            elif format == 'excel':
                output_path = f"{output_file}.xlsx" if not output_file.endswith('.xlsx') else output_file
                df.to_excel(output_path, index=False, engine='openpyxl')

            else:
                logging.error(f"Unknown format: {format}")
                return

            logging.info(f"Data saved to {output_path}")
            print(f"Saved {len(df)} cleaned records to {output_path}")

        except Exception as e:
            logging.error(f"Failed to save data: {e}")

    def save_quality_report(self, report: Dict, filename: str = None):
        """Save quality report to JSON file."""
        if filename is None:
            filename = self.config['output']['quality_report_file']

        try:
            # Convert numpy types to native Python types
            def convert_types(obj):
                if isinstance(obj, dict):
                    return {key: convert_types(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_types(item) for item in obj]
                elif hasattr(obj, 'item'):  # numpy types
                    return obj.item()
                else:
                    return obj

            report_clean = convert_types(report)

            with open(filename, 'w') as f:
                json.dump(report_clean, f, indent=4)

            logging.info(f"Quality report saved to {filename}")
            print(f"Quality report saved to {filename}")

        except Exception as e:
            logging.error(f"Failed to save quality report: {e}")

    def process(self, input_file: str, output_file: str, format: str = None) -> pd.DataFrame:
        """
        Main processing method.

        Args:
            input_file: Path to input file (CSV or JSON)
            output_file: Path for output file (without extension)
            format: Output format ('csv', 'json', 'excel')

        Returns:
            Cleaned DataFrame
        """
        self.stats['start_time'] = datetime.now()

        logging.info(f"Loading data from {input_file}")

        # Load data
        try:
            if input_file.endswith('.csv'):
                df = pd.read_csv(input_file)
            elif input_file.endswith('.json'):
                df = pd.read_json(input_file)
            else:
                # Try both
                try:
                    df = pd.read_csv(input_file)
                except Exception as e:
                    print(f"Error: {e}")
                    df = pd.read_json(input_file)
        except Exception as e:
            logging.error(f"Failed to load data: {e}")
            raise

        self.stats['initial_records'] = len(df)

        # Clean data
        df_clean = self.clean_data(df)

        self.stats['final_records'] = len(df_clean)
        self.stats['end_time'] = datetime.now()

        # Print statistics
        self.print_statistics()

        # Save cleaned data
        self.save_data(df_clean, output_file, format)

        # Generate and save quality report
        if self.config['output']['save_quality_report']:
            report = self.generate_quality_report()
            self.save_quality_report(report)

        return df_clean

def main():
    """Command-line interface for the data cleaner."""
    parser = argparse.ArgumentParser(
        description='Professional Data Cleaning Tool for Scraped E-commerce Data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cleaner.py input.csv output                    # Use defaults
  python cleaner.py input.json output --format csv      # Convert JSON to CSV
  python cleaner.py data.csv cleaned --format excel     # Export to Excel
  python cleaner.py --config custom.yaml input.csv out  # Custom config
        """
    )

    parser.add_argument(
        'input_file',
        help='Input file path (CSV or JSON)'
    )

    parser.add_argument(
        'output_file',
        help='Output file path (without extension)'
    )

    parser.add_argument(
        '--config',
        type=str,
        default='cleaner_config.yaml',
        help='Path to config file (default: cleaner_config.yaml)'
    )

    parser.add_argument(
        '--format',
        type=str,
        choices=['csv', 'json', 'excel'],
        help='Output format (default: from config)'
    )

    parser.add_argument(
        '--no-report',
        action='store_true',
        help='Do not generate quality report'
    )

    args = parser.parse_args()

    # Initialize cleaner
    cleaner = DataCleaner(config_file=args.config)

    # Override config if needed
    if args.no_report:
        cleaner.config['output']['save_quality_report'] = False

    # Process data
    try:
        cleaner.process(args.input_file, args.output_file, format=args.format)
        print("\nData cleaning complete!")
    except Exception as e:
        logging.error(f"Processing failed: {e}")
        print(f"\nError: {e}")
        exit(1)

if __name__ == "__main__":
    main()