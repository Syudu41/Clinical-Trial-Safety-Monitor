"""
Test script for FDA Data Collector
Run this to verify the collector is working properly
"""

import sys
from pathlib import Path

# Add src to Python path  
sys.path.append(str(Path(__file__).parent / 'src'))

try:
    from src.data.collectors.fda_collector import FDADataCollector, quick_test
    print("✅ FDA Collector imported successfully")
except ImportError as e:
    print(f"❌ Failed to import FDA Collector: {e}")
    print("Make sure you've created all the required files and directories")
    sys.exit(1)

def main():
    """Main test function"""
    print("🚀 Testing FDA Data Collector")
    print("=" * 50)
    
    # Run the built-in quick test
    success = quick_test()
    
    if success:
        print("\n🎉 FDA Collector is working properly!")
        print("\nNext steps:")
        print("1. The API connection is working")
        print("2. Ready to fetch sample data")
        print("3. Ready to download bulk data files")
    else:
        print("\n⚠️  FDA Collector has some issues")
        print("Check your internet connection and try again")
    
    # Additional detailed test
    print("\n" + "=" * 50)
    print("DETAILED TESTING")
    print("=" * 50)
    
    try:
        collector = FDADataCollector()
        
        # Test small API fetch with proven working dates
        print("\n🔍 Testing API data fetch with Jan 2024 data...")
        sample_df = collector.fetch_recent_events_api(
            start_date="20240101", 
            end_date="20240131", 
            limit=3
        )
        
        if not sample_df.empty:
            print(f"✅ Successfully fetched {len(sample_df)} sample records")
            print("\n📊 Sample data structure:")
            print(f"Shape: {sample_df.shape}")
            print(f"Columns: {list(sample_df.columns)[:10]}...")  # First 10 columns
            
            # Show sample data
            print(f"\nSample data preview:")
            for col in ['safetyreportid', 'serious', 'receivedate'][:3]:
                if col in sample_df.columns:
                    print(f"  {col}: {sample_df[col].iloc[0] if len(sample_df) > 0 else 'N/A'}")
            
            # Save sample data
            sample_file = Path("data/raw/sample_fda_data.csv")
            sample_df.to_csv(sample_file, index=False)
            print(f"💾 Sample data saved to: {sample_file}")
            
        else:
            print("⚠️  No sample data retrieved")
            
    except Exception as e:
        print(f"❌ Detailed test failed: {e}")

if __name__ == "__main__":
    main()