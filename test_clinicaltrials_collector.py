"""
Test script for ClinicalTrials.gov Data Collector
Run this to verify the collector is working properly
"""

import sys
from pathlib import Path

# Add src to Python path  
sys.path.append(str(Path(__file__).parent / 'src'))

try:
    from src.data.collectors.clinicaltrials_collector import ClinicalTrialsCollector, quick_test
    print("✅ ClinicalTrials Collector imported successfully")
except ImportError as e:
    print(f"❌ Failed to import ClinicalTrials Collector: {e}")
    print("Make sure you've created the clinicaltrials_collector.py file")
    sys.exit(1)

def main():
    """Main test function"""
    print("🚀 Testing ClinicalTrials.gov Data Collector")
    print("=" * 60)
    
    # Run the built-in quick test
    success = quick_test()
    
    if success:
        print("\n🎉 ClinicalTrials Collector is working properly!")
    else:
        print("\n⚠️  ClinicalTrials Collector has some issues")
    
    # Additional detailed testing
    print("\n" + "=" * 60)
    print("DETAILED TESTING")
    print("=" * 60)
    
    try:
        collector = ClinicalTrialsCollector()
        
        # Test different types of queries
        print("\n1️⃣ Testing Phase 2 studies...")
        phase2_df = collector.fetch_studies_by_phase('PHASE2', max_studies=3)  # Updated for v2.0
        
        if not phase2_df.empty:
            print(f"✅ Phase 2 studies: {len(phase2_df)} records")
            print(f"Sample columns: {list(phase2_df.columns)[:5]}...")
            
            # Save sample data
            sample_file = Path("data/raw/sample_phase2_studies.csv")
            phase2_df.to_csv(sample_file, index=False)
            print(f"💾 Phase 2 data saved to: {sample_file}")
        else:
            print("⚠️  No Phase 2 studies found")
        
        print("\n2️⃣ Testing drug-specific studies...")
        # Test with a common drug
        drug_df = collector.fetch_drug_studies('aspirin', max_studies=3)
        
        if not drug_df.empty:
            print(f"✅ Aspirin studies: {len(drug_df)} records")
            print(f"Sample columns: {list(drug_df.columns)[:5]}...")
            
            # Show sample data
            if 'nctId' in drug_df.columns:  # Updated column name for v2.0
                print(f"Sample NCT IDs: {drug_df['nctId'].tolist()}")
        else:
            print("⚠️  No aspirin studies found")
        
        print("\n3️⃣ Testing recent studies...")
        recent_df = collector.fetch_recent_studies(max_studies=3)  # Simplified for v2.0
        
        if not recent_df.empty:
            print(f"✅ Recent studies: {len(recent_df)} records")
            
            # Save combined sample data
            sample_file = Path("data/raw/sample_clinical_trials.csv")
            recent_df.to_csv(sample_file, index=False)
            print(f"💾 Recent studies saved to: {sample_file}")
        else:
            print("⚠️  No recent studies found")
            
    except Exception as e:
        print(f"❌ Detailed test failed: {e}")
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("If you see ✅ marks above, the ClinicalTrials collector is working!")
    print("Ready to proceed to Step 3: Data Processing & Analysis")

if __name__ == "__main__":
    main()