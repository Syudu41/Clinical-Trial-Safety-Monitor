import requests
import pandas as pd
import time
import json
from datetime import datetime
import os

class ScalableFDACollector:
    def __init__(self):
        self.base_url = "https://api.fda.gov/drug/event.json"
        self.rate_limit_delay = 15  # seconds between requests
        self.max_per_request = 100  # FDA limit
        
    def collect_large_balanced_dataset(self, target_serious=5000, target_non_serious=5000):
        """
        Collect a large balanced dataset for professional ML training
        """
        print(f"🎯 TARGET: {target_serious:,} serious + {target_non_serious:,} non-serious events")
        print(f"📊 TOTAL: {target_serious + target_non_serious:,} records")
        print(f"⏱️  ESTIMATED TIME: ~{((target_serious + target_non_serious) / 100) * 15 / 60:.1f} minutes")
        print("=" * 60)
        
        all_records = []
        start_time = datetime.now()
        
        # Step 1: Collect serious events
        print("📊 PHASE 1: Collecting SERIOUS adverse events...")
        serious_records = self.collect_by_seriousness_large(
            serious=True, 
            target_count=target_serious
        )
        all_records.extend(serious_records)
        
        phase1_time = datetime.now()
        print(f"⏱️  Phase 1 completed in {(phase1_time - start_time).total_seconds()/60:.1f} minutes")
        
        # Step 2: Collect non-serious events  
        print("\n📊 PHASE 2: Collecting NON-SERIOUS adverse events...")
        non_serious_records = self.collect_by_seriousness_large(
            serious=False, 
            target_count=target_non_serious
        )
        all_records.extend(non_serious_records)
        
        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds() / 60
        
        print(f"\n🎉 COLLECTION COMPLETE!")
        print(f"📈 Total records: {len(all_records):,}")
        print(f"⏱️  Total time: {total_time:.1f} minutes")
        print(f"🚀 Ready for professional ML training!")
        
        return all_records
    
    def collect_by_seriousness_large(self, serious=True, target_count=5000):
        """
        Efficiently collect large number of events
        """
        records = []
        skip = 0
        batch_number = 1
        
        print(f"🎯 Collecting {target_count:,} {'SERIOUS' if serious else 'NON-SERIOUS'} events...")
        
        while len(records) < target_count:
            remaining = target_count - len(records)
            limit = min(self.max_per_request, remaining)
            
            # Build optimized query
            if serious:
                query = "serious:1+AND+receivedate:[20240101+TO+20240331]"
            else:
                query = "serious:2+AND+receivedate:[20240101+TO+20240331]"
            
            url = f"{self.base_url}?search={query}&limit={limit}&skip={skip}"
            
            progress = (len(records) / target_count) * 100
            print(f"  📡 Batch {batch_number:2d}: {len(records):,}/{target_count:,} records ({progress:.1f}%)")
            
            try:
                response = requests.get(url, timeout=30)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if 'results' in data and len(data['results']) > 0:
                        batch_records = data['results']
                        records.extend(batch_records)
                        skip += len(batch_records)
                        batch_number += 1
                        
                        # Progress update
                        if len(batch_records) < limit:
                            print(f"  ⚠️  Reached end of 2024 Q1 data. Switching to full dataset...")
                            # Switch to no-date query for remaining records
                            query = f"serious:{'1' if serious else '2'}"
                    else:
                        print(f"  ❌ No more results available")
                        break
                        
                elif response.status_code == 404:
                    print(f"  💡 Switching to broader date range...")
                    query = f"serious:{'1' if serious else '2'}"
                    continue
                    
                else:
                    print(f"  ❌ API Error: {response.status_code}")
                    break
                
                # Smart rate limiting - only wait if we're making another request
                if len(records) < target_count:
                    print(f"  ⏱️  Rate limit pause...")
                    time.sleep(self.rate_limit_delay)
                        
            except Exception as e:
                print(f"  ❌ Error: {str(e)}")
                print(f"  🔄 Retrying in 30 seconds...")
                time.sleep(30)
                continue
        
        final_count = len(records)
        print(f"✅ Collected {final_count:,} {'serious' if serious else 'non-serious'} events")
        return records[:target_count]  # Ensure exact count
    
    def save_large_dataset(self, records, filename="fda_10k_dataset.csv"):
        """
        Save large dataset with analytics
        """
        if not records:
            print("❌ No records to save")
            return
        
        print(f"\n💾 SAVING {len(records):,} RECORDS...")
        
        # Flatten records efficiently  
        flattened_records = []
        for i, record in enumerate(records):
            if i % 1000 == 0:  # Progress for large datasets
                print(f"  Processing record {i:,}/{len(records):,}")
            
            flat_record = self.flatten_record(record)
            flattened_records.append(flat_record)
        
        df = pd.DataFrame(flattened_records)
        
        # Ensure directories exist
        os.makedirs("data/raw", exist_ok=True)
        filepath = f"data/raw/{filename}"
        
        # Save with compression for large files
        df.to_csv(filepath, index=False)
        
        # Analytics
        file_size = os.path.getsize(filepath) / (1024*1024)  # MB
        print(f"\n📊 DATASET ANALYTICS:")
        print(f"💾 File: {filepath}")
        print(f"📏 Size: {file_size:.1f} MB")
        print(f"📊 Records: {len(df):,}")
        print(f"📋 Features: {len(df.columns)}")
        
        if 'serious' in df.columns:
            serious_counts = df['serious'].value_counts()
            print(f"🎯 CLASS DISTRIBUTION:")
            for value, count in serious_counts.items():
                label = "Serious" if str(value) == "1" else "Non-serious" 
                print(f"   {label}: {count:,} ({count/len(df)*100:.1f}%)")
        
        return filepath
    
    def flatten_record(self, record):
        """Optimized flattening for large datasets"""
        flat = {}
        for key, value in record.items():
            if isinstance(value, list) and len(value) > 0:
                if isinstance(value[0], dict):
                    for subkey, subvalue in value[0].items():
                        flat[f"{key}_{subkey}"] = subvalue
                else:
                    flat[key] = value[0]
            elif isinstance(value, dict):
                for subkey, subvalue in value.items():
                    flat[f"{key}_{subkey}"] = subvalue
            else:
                flat[key] = value
        return flat

def main():
    print("🏥 LARGE-SCALE FDA DATA COLLECTOR")
    print("🚀 Professional ML Dataset Generation")
    print("=" * 60)
    
    collector = ScalableFDACollector()
    
    # Same code as before, but change the targets:
    records = collector.collect_large_balanced_dataset(
        target_serious=12500,     # 12.5K serious
        target_non_serious=12500  # 12.5K non-serious  
    )
    
    if records:
        filepath = collector.save_large_dataset(records)
        print(f"\n🎉 SUCCESS! Professional dataset ready!")
        print(f"📈 {len(records):,} records ready for ML training")
        print(f"💪 This dataset will produce excellent ML models!")
    else:
        print("\n❌ Collection failed")

if __name__ == "__main__":
    main()