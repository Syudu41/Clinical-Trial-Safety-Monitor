import requests
import pandas as pd
import time
import json
from datetime import datetime, timedelta
import os

class EnhancedFDACollector:
    def __init__(self):
        self.base_url = "https://api.fda.gov/drug/event.json"
        self.rate_limit_delay = 15  # seconds between requests
        self.max_per_request = 100  # FDA limit
        
    def collect_balanced_dataset(self, target_serious=250, target_non_serious=250):
        """
        Collect a balanced dataset with both serious and non-serious events
        """
        print(f"🎯 Target: {target_serious} serious + {target_non_serious} non-serious events")
        print("⏱️  Estimated time: ~15 minutes (respecting FDA rate limits)")
        
        all_records = []
        
        # Step 1: Collect serious events
        print("\n📊 Step 1: Collecting SERIOUS adverse events...")
        serious_records = self.collect_by_seriousness(
            serious=True, 
            target_count=target_serious
        )
        all_records.extend(serious_records)
        
        # Step 2: Collect non-serious events  
        print("\n📊 Step 2: Collecting NON-SERIOUS adverse events...")
        non_serious_records = self.collect_by_seriousness(
            serious=False, 
            target_count=target_non_serious
        )
        all_records.extend(non_serious_records)
        
        print(f"\n✅ Total collected: {len(all_records)} records")
        return all_records
    
    def collect_by_seriousness(self, serious=True, target_count=250):
        """
        Collect events filtered by seriousness
        """
        records = []
        skip = 0
        
        while len(records) < target_count:
            remaining = target_count - len(records)
            limit = min(self.max_per_request, remaining)
            
            # Build query for serious vs non-serious
            if serious:
                query = f"serious:1"
            else:
                query = f"serious:2"  # 2 = non-serious in FDA data
            
            # Add date range for 2024 Q1 (known to work)
            query += "+AND+receivedate:[20240101+TO+20240331]"
            
            url = f"{self.base_url}?search={query}&limit={limit}&skip={skip}"
            
            print(f"  📡 Requesting {'serious' if serious else 'non-serious'} events: {len(records)}/{target_count}")
            
            try:
                response = requests.get(url)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if 'results' in data:
                        batch_records = data['results']
                        records.extend(batch_records)
                        skip += len(batch_records)
                        
                        print(f"  ✅ Got {len(batch_records)} records (Total: {len(records)})")
                        
                        # Check if we got fewer than requested (end of data)
                        if len(batch_records) < limit:
                            print(f"  ⚠️  Reached end of available data for {'serious' if serious else 'non-serious'} events")
                            break
                    else:
                        print(f"  ❌ No results in response")
                        break
                        
                else:
                    print(f"  ❌ API Error: {response.status_code}")
                    if response.status_code == 404:
                        print("  💡 Trying without date filter...")
                        # Retry without date restriction
                        query_no_date = f"serious:{'1' if serious else '2'}"
                        url_no_date = f"{self.base_url}?search={query_no_date}&limit={limit}&skip={skip}"
                        response = requests.get(url_no_date)
                        if response.status_code == 200:
                            data = response.json()
                            if 'results' in data:
                                batch_records = data['results']
                                records.extend(batch_records)
                                skip += len(batch_records)
                        
                # Rate limiting - respect FDA limits
                print(f"  ⏱️  Waiting {self.rate_limit_delay}s (rate limit)...")
                time.sleep(self.rate_limit_delay)
                
            except Exception as e:
                print(f"  ❌ Error: {str(e)}")
                break
        
        return records[:target_count]  # Ensure exact count
    
    def save_raw_data(self, records, filename="enhanced_fda_data.csv"):
        """
        Save raw records to CSV
        """
        if not records:
            print("❌ No records to save")
            return
            
        # Flatten nested JSON structure for CSV
        flattened_records = []
        for record in records:
            flat_record = self.flatten_record(record)
            flattened_records.append(flat_record)
        
        df = pd.DataFrame(flattened_records)
        
        # Ensure directories exist
        os.makedirs("data/raw", exist_ok=True)
        filepath = f"data/raw/{filename}"
        
        df.to_csv(filepath, index=False)
        print(f"💾 Saved {len(df)} records to {filepath}")
        print(f"📊 Columns: {len(df.columns)}")
        
        # Quick analysis
        if 'serious' in df.columns:
            serious_dist = df['serious'].value_counts()
            print(f"🎯 Seriousness distribution:")
            print(f"   Serious (1): {serious_dist.get(1, 0) if 1 in serious_dist.index else serious_dist.get('1', 0)}")
            print(f"   Non-serious (2): {serious_dist.get(2, 0) if 2 in serious_dist.index else serious_dist.get('2', 0)}")
        
        return filepath
    
    def flatten_record(self, record):
        """
        Flatten nested FDA record structure
        """
        flat = {}
        
        # Handle top-level fields
        for key, value in record.items():
            if isinstance(value, list) and len(value) > 0:
                # Take first item for lists (common in FDA data)
                if isinstance(value[0], dict):
                    # Nested object in list
                    for subkey, subvalue in value[0].items():
                        flat[f"{key}_{subkey}"] = subvalue
                else:
                    # Simple list
                    flat[key] = value[0]
            elif isinstance(value, dict):
                # Nested object
                for subkey, subvalue in value.items():
                    flat[f"{key}_{subkey}"] = subvalue
            else:
                # Simple value
                flat[key] = value
        
        return flat

def main():
    print("🏥 Enhanced FDA Adverse Event Collector")
    print("=" * 50)
    
    collector = EnhancedFDACollector()
    
    # Collect balanced dataset
    records = collector.collect_balanced_dataset(
        target_serious=100,      # Start smaller for testing
        target_non_serious=100   # Total: 200 records
    )
    
    if records:
        filepath = collector.save_raw_data(records)
        print(f"\n🎉 Success! Enhanced dataset ready at: {filepath}")
        print("📈 Ready for ML model training!")
    else:
        print("\n❌ Failed to collect data")

if __name__ == "__main__":
    main()