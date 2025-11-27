import pandas as pd
import json
import re

def clean_currency(price_str: str) -> float:
    """Converts '£51.77' to 51.77"""
    try:
        # Remove currency symbols and convert to float
        clean_price = re.sub(r'[^\d.]', '', price_str)
        return float(clean_price)
    except ValueError:
        return 0.0

def auto_label_category(name: str) -> str:
    """
    Simulates AI Labeling: Categorizes products based on keywords.
    """
    name = name.lower()
    if 'data' in name or 'python' in name:
        return 'Tech'
    elif 'art' in name or 'design' in name:
        return 'Creative'
    else:
        return 'General'

def process_data(input_file: str, output_file: str):
    print(f"🧹 Cleaning and Labeling data from {input_file}...")
    
    # Load Raw Data
    df = pd.read_csv(input_file)
    
    # 1. Cleaning Step
    df['price'] = df['price_raw'].apply(clean_currency)
    
    # 2. Labeling Step (The AI Part)
    df['category_label'] = df['name'].apply(auto_label_category)
    
    # 3. Validation (Ensure no missing prices)
    df = df[df['price'] > 0]
    
    # Export to JSON as requested
    records = df.to_dict(orient='records')
    with open(output_file, 'w') as f:
        json.dump(records, f, indent=4)
        
    print(f"✨ Success! Processed {len(df)} items. Saved to {output_file}")

if __name__ == "__main__":
    process_data("raw_products.csv", "final_labeled_dataset.json")