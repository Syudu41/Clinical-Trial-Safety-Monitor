"""
Test script for Data Cleaning Pipeline
Run this to verify the data cleaner works with your sample data
"""

import sys
from pathlib import Path
import pandas as pd

# Add src to Python path  
sys.path.append(str(Path(__file__).parent / 'src'))

try:
    from src.data.processors.data_cleaner import DataCleaner, quick_clean_sample_data
    print("✅ Data Cleaner imported successfully")
except ImportError as e:
    print(f"❌ Failed to import Data Cleaner: {e}")
    print("Make sure you've created the data_cleaner.py file in src/data/processors/")
    sys.exit(1)

def main():
    """Main test function"""
    print("🚀 Testing Data Cleaning Pipeline")
    print("=" * 60)
    
    # Run the built-in quick test
    success = quick_clean_sample_data()
    
    if success:
        print("\n🎉 Data cleaning pipeline is working properly!")
    else:
        print("\n⚠️  Data cleaning pipeline has some issues")
    
    # Additional detailed testing if basic test worked
    if success:
        print("\n" + "=" * 60)
        print("DETAILED DATA ANALYSIS")
        print("=" * 60)
        
        try:
            # Load the processed data for analysis
            from config import config
            processed_file = config.PROCESSED_DATA_DIR / "processed_fda_sample.csv"
            
            if processed_file.exists():
                df = pd.read_csv(processed_file)
                
                print(f"\n📊 PROCESSED DATA OVERVIEW:")
                print(f"Shape: {df.shape}")
                print(f"Memory usage: {df.memory_usage().sum() / 1024:.1f} KB")
                
                print(f"\n🎯 TARGET VARIABLE ANALYSIS:")
                if 'is_serious' in df.columns:
                    serious_counts = df['is_serious'].value_counts()
                    print(f"Non-serious events (0): {serious_counts.get(0, 0)}")
                    print(f"Serious events (1): {serious_counts.get(1, 0)}")
                    if len(serious_counts) > 0:
                        serious_pct = (serious_counts.get(1, 0) / len(df)) * 100
                        print(f"Serious event rate: {serious_pct:.1f}%")
                else:
                    print("Target variable 'is_serious' not found")
                
                print(f"\n🔢 FEATURE OVERVIEW:")
                print(f"Total features: {len(df.columns)}")
                
                # Show feature types
                numeric_cols = df.select_dtypes(include=['number']).columns
                categorical_cols = df.select_dtypes(include=['object']).columns
                print(f"Numeric features: {len(numeric_cols)}")
                print(f"Categorical features: {len(categorical_cols)}")
                
                # Show sample of feature names
                print(f"\nSample numeric features: {list(numeric_cols)[:5]}")
                if len(categorical_cols) > 0:
                    print(f"Sample categorical features: {list(categorical_cols)[:5]}")
                
                print(f"\n📋 DATA QUALITY CHECK:")
                missing_data = df.isnull().sum()
                if missing_data.sum() > 0:
                    print("Features with missing values:")
                    for col, missing_count in missing_data[missing_data > 0].items():
                        print(f"  {col}: {missing_count} missing")
                else:
                    print("✅ No missing values found")
                
                # Check for key risk scores
                risk_features = ['patient_risk_score', 'drug_risk_score', 'outcome_severity']
                print(f"\n🎲 RISK SCORES SUMMARY:")
                for risk_feature in risk_features:
                    if risk_feature in df.columns:
                        risk_stats = df[risk_feature].describe()
                        print(f"{risk_feature}: mean={risk_stats['mean']:.2f}, std={risk_stats['std']:.2f}")
                    else:
                        print(f"{risk_feature}: not found")
                        
            else:
                print("❌ Processed data file not found")
                
        except Exception as e:
            print(f"❌ Detailed analysis failed: {e}")
    
    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    if success:
        print("✅ Your data is now cleaned and ready for machine learning!")
        print("Ready to proceed to:")
        print("  1. Exploratory Data Analysis (EDA)")
        print("  2. Feature Engineering refinement") 
        print("  3. ML Model Development")
    else:
        print("❌ Please fix the data cleaning issues before proceeding")

if __name__ == "__main__":
    main()