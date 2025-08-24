import pandas as pd
import os

def check_current_data():
    print("=== Current Dataset Analysis ===\n")
    
    # Check raw data
    raw_path = "data/raw/sample_fda_data.csv"
    processed_path = "data/processed/processed_fda_sample.csv"
    
    if os.path.exists(raw_path):
        raw_df = pd.read_csv(raw_path)
        print(f"Raw Data: {len(raw_df)} rows, {len(raw_df.columns)} columns")
        print("Raw columns:", list(raw_df.columns)[:10], "..." if len(raw_df.columns) > 10 else "")
        print()
    else:
        print("❌ Raw data file not found")
    
    if os.path.exists(processed_path):
        processed_df = pd.read_csv(processed_path)
        print(f"Processed Data: {len(processed_df)} rows, {len(processed_df.columns)} columns")
        print("Processed columns:", list(processed_df.columns))
        print()
        
        # Check target variable distribution
        if 'is_serious' in processed_df.columns:
            print("Target Variable Distribution:")
            print(processed_df['is_serious'].value_counts())
        print()
    else:
        print("❌ Processed data file not found")
    
    print("=== Recommendation ===")
    if os.path.exists(processed_path):
        df = pd.read_csv(processed_path)
        if len(df) < 100:
            print(f"⚠️  Current dataset ({len(df)} rows) is too small for ML training")
            print("📈 Recommended: Collect 500-1000 records minimum")
            print("🎯 Target: 5000+ records for robust ML models")
        else:
            print(f"✅ Dataset size ({len(df)} rows) is adequate for initial ML development")

if __name__ == "__main__":
    check_current_data()