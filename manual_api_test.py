"""
Manual API test script to debug FDA API issues
"""

import requests
import json
from datetime import datetime

def test_basic_fda_api():
    """Test the FDA API with the simplest possible query"""
    
    print("🧪 Manual FDA API Testing")
    print("=" * 50)
    
    # Test 1: Basic connection test
    print("\n1️⃣ Testing basic API endpoint...")
    try:
        basic_url = "https://api.fda.gov/drug/event.json?limit=1"
        print(f"URL: {basic_url}")
        
        response = requests.get(basic_url, timeout=30)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            total_results = data.get('meta', {}).get('results', {}).get('total', 'unknown')
            print(f"✅ Basic API works! Total results available: {total_results}")
            
            # Show sample data structure
            results = data.get('results', [])
            if results:
                print(f"Sample record keys: {list(results[0].keys())[:10]}")
        else:
            print(f"❌ Basic API failed: {response.status_code}")
            print(f"Response: {response.text[:200]}")
            
    except Exception as e:
        print(f"❌ Basic API test error: {e}")
    
    # Test 2: Try different date ranges
    print("\n2️⃣ Testing different date ranges...")
    date_ranges = [
        ("20240101", "20240131", "Jan 2024"),
        ("20230101", "20230131", "Jan 2023"),  
        ("20220101", "20220131", "Jan 2022")
    ]
    
    for start_date, end_date, description in date_ranges:
        try:
            search_query = f"receivedate:[{start_date}+TO+{end_date}]"
            url = f"https://api.fda.gov/drug/event.json?search={search_query}&limit=1"
            print(f"\n   Testing {description}...")
            print(f"   URL: {url}")
            
            response = requests.get(url, timeout=30)
            print(f"   Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                total = data.get('meta', {}).get('results', {}).get('total', 0)
                print(f"   ✅ Found {total} records for {description}")
                if total > 0:
                    print(f"   SUCCESS! Using {description} data works")
                    return start_date, end_date
            else:
                print(f"   ❌ Failed for {description}: {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Error testing {description}: {e}")
    
    # Test 3: Try without date restrictions
    print("\n3️⃣ Testing without date restrictions...")
    try:
        url = "https://api.fda.gov/drug/event.json?limit=5"
        print(f"URL: {url}")
        
        response = requests.get(url, timeout=30)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            results = data.get('results', [])
            print(f"✅ No-date query works! Got {len(results)} records")
            
            # Show what data looks like
            if results:
                first_record = results[0]
                print(f"Sample record keys: {list(first_record.keys())[:10]}")
                
                # Check for receive date to understand date format
                if 'receivedate' in first_record:
                    print(f"Sample receivedate: {first_record['receivedate']}")
                    
        else:
            print(f"❌ No-date query failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ No-date query error: {e}")
    
    print("\n" + "=" * 50)
    print("Manual API test complete")
    return None, None

if __name__ == "__main__":
    test_basic_fda_api()