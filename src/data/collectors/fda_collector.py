"""
FDA Data Collector
Handles both bulk FAERS data download and real-time API access
"""

import requests
import json
import pandas as pd
import zipfile
import os
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import config, data_sources

class FDADataCollector:
    """Collects data from FDA sources including FAERS bulk data and OpenFDA API"""
    
    def __init__(self):
        self.logger = self._setup_logging()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Clinical-Trial-Safety-Monitor/1.0 (Research Project)'
        })
        
        # Rate limiting tracking
        self.last_request_time = 0
        self.request_count = 0
        self.daily_request_count = 0
        
    def _setup_logging(self):
        """Setup logging for the collector"""
        logger = logging.getLogger(f"{__name__}.FDADataCollector")
        return logger
        
    def _respect_rate_limits(self):
        """Implement rate limiting for API requests"""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        
        if time_since_last_request < config.FDA_REQUEST_DELAY:
            sleep_time = config.FDA_REQUEST_DELAY - time_since_last_request
            self.logger.info(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
        self.request_count += 1
        self.daily_request_count += 1
        
        # Check daily limits
        if self.daily_request_count >= config.FDA_API_DAILY_LIMIT:
            self.logger.warning(f"Approaching daily API limit ({self.daily_request_count}/{config.FDA_API_DAILY_LIMIT})")
    
    def test_api_connection(self) -> bool:
        """Test basic FDA API connectivity"""
        try:
            self.logger.info("Testing FDA API connection...")
            test_url = f"{data_sources.FDA_BASE_URL}{data_sources.FDA_TEST_ENDPOINT}"
            
            self._respect_rate_limits()
            response = self.session.get(test_url, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                total_results = data.get('meta', {}).get('results', {}).get('total', 0)
                self.logger.info(f"✅ FDA API connection successful! Found {total_results} total results")
                return True
            else:
                self.logger.error(f"❌ FDA API returned status code: {response.status_code}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ FDA API connection failed: {e}")
            return False
    
    def download_faers_bulk_data(self, quarters: List[str] = None, force_download: bool = False) -> Dict[str, Path]:
        """
        Download FAERS quarterly bulk data files
        
        Args:
            quarters: List of quarters to download (e.g., ['2024Q1', '2024Q2'])
            force_download: Re-download even if files exist
            
        Returns:
            Dictionary mapping quarter to downloaded file path
        """
        if quarters is None:
            quarters = list(data_sources.FAERS_QUARTERS.keys())
        
        downloaded_files = {}
        
        for quarter in quarters:
            if quarter not in data_sources.FAERS_QUARTERS:
                self.logger.warning(f"Unknown quarter: {quarter}")
                continue
                
            url = data_sources.FAERS_QUARTERS[quarter]
            filename = f"faers_{quarter.lower()}.zip"
            filepath = config.RAW_DATA_DIR / filename
            
            # Skip if file exists and not forcing download
            if filepath.exists() and not force_download:
                self.logger.info(f"📁 {quarter} data already exists at {filepath}")
                downloaded_files[quarter] = filepath
                continue
            
            try:
                self.logger.info(f"📥 Downloading FAERS {quarter} data...")
                self.logger.info(f"URL: {url}")
                
                # Download with progress tracking
                response = self.session.get(url, stream=True, timeout=120)
                response.raise_for_status()
                
                total_size = int(response.headers.get('content-length', 0))
                self.logger.info(f"File size: {total_size / 1024 / 1024:.1f} MB")
                
                with open(filepath, 'wb') as f:
                    downloaded = 0
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            
                            # Progress update every 10MB
                            if downloaded % (10 * 1024 * 1024) == 0:
                                progress = (downloaded / total_size) * 100 if total_size > 0 else 0
                                self.logger.info(f"Progress: {progress:.1f}%")
                
                self.logger.info(f"✅ {quarter} downloaded successfully to {filepath}")
                downloaded_files[quarter] = filepath
                
            except Exception as e:
                self.logger.error(f"❌ Failed to download {quarter}: {e}")
        
        return downloaded_files
    
    def extract_faers_data(self, zip_path: Path, extract_files: List[str] = None) -> Dict[str, Path]:
        """
        Extract specific files from FAERS ZIP archive
        
        Args:
            zip_path: Path to the ZIP file
            extract_files: List of file prefixes to extract (e.g., ['DEMO', 'REAC'])
            
        Returns:
            Dictionary mapping file type to extracted file path
        """
        if extract_files is None:
            extract_files = ['DEMO', 'REAC', 'DRUG', 'OUTC']  # Essential files for Day 1
        
        extracted_files = {}
        extract_dir = config.RAW_DATA_DIR / f"extracted_{zip_path.stem}"
        extract_dir.mkdir(exist_ok=True)
        
        try:
            self.logger.info(f"📂 Extracting files from {zip_path.name}")
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # List all files in the archive
                file_list = zip_ref.namelist()
                self.logger.info(f"Found {len(file_list)} files in archive")
                
                # Extract matching files
                for file_prefix in extract_files:
                    matching_files = [f for f in file_list if f.upper().startswith(file_prefix)]
                    
                    if matching_files:
                        # Usually there's one file per type (e.g., DEMO24Q1.txt)
                        source_file = matching_files[0]
                        
                        # Create a standardized filename
                        target_filename = f"{file_prefix.lower()}_{zip_path.stem}.txt"
                        target_path = extract_dir / target_filename
                        
                        # Extract and rename
                        zip_ref.extract(source_file, extract_dir)
                        extracted_source = extract_dir / source_file
                        extracted_source.rename(target_path)
                        
                        extracted_files[file_prefix] = target_path
                        self.logger.info(f"✅ Extracted {file_prefix} data to {target_filename}")
                    else:
                        self.logger.warning(f"⚠️  No files found for {file_prefix}")
        
        except Exception as e:
            self.logger.error(f"❌ Failed to extract {zip_path}: {e}")
        
        return extracted_files
    
    def fetch_recent_events_api(self, 
                               start_date: str = None, 
                               end_date: str = None, 
                               limit: int = 100,
                               search_terms: List[str] = None) -> pd.DataFrame:
        """
        Fetch recent adverse events from FDA OpenFDA API
        
        Args:
            start_date: Start date in YYYYMMDD format
            end_date: End date in YYYYMMDD format  
            limit: Number of records to fetch (max 1000 per request)
            search_terms: Additional search terms for filtering
            
        Returns:
            DataFrame with adverse event data
        """
        # Set default date range (use proven working dates from manual testing)
        if not start_date or not end_date:
            start_date = config.DEFAULT_START_DATE  # "20240101"
            end_date = config.DEFAULT_END_DATE      # "20240331"
        
        try:
            # First try a simple query without date restrictions to test basic functionality
            if not search_terms:
                self.logger.info("🔍 Trying basic query first (no date restrictions)")
                simple_url = f"{data_sources.FDA_BASE_URL}/drug/event.json"
                simple_params = {'limit': min(limit, 100)}
                
                self._respect_rate_limits()
                simple_response = self.session.get(simple_url, params=simple_params, timeout=60)
                
                if simple_response.status_code == 200:
                    self.logger.info("✅ Basic query successful, now trying with dates")
                else:
                    self.logger.warning(f"Basic query returned: {simple_response.status_code}")
            
            # Build search query with dates - fix URL encoding
            search_query = f"receivedate:[{start_date} TO {end_date}]"  # Remove the + signs
            if search_terms:
                search_query += " AND " + " AND ".join(search_terms)
            
            # Build API URL
            api_url = f"{data_sources.FDA_BASE_URL}/drug/event.json"
            params = {
                'search': search_query,  # Let requests handle the encoding
                'limit': min(limit, 1000)  # FDA API limit is 1000 per request
            }
            
            self.logger.info(f"🔍 Fetching events from {start_date} to {end_date}")
            self.logger.info(f"API URL: {api_url}")
            self.logger.info(f"Search query: {search_query}")
            
            self._respect_rate_limits()
            response = self.session.get(api_url, params=params, timeout=60)
            
            # Better error handling
            if response.status_code != 200:
                self.logger.error(f"API returned status code: {response.status_code}")
                self.logger.error(f"Response content: {response.text[:500]}")
                
                # Try a fallback query without dates
                self.logger.info("🔄 Trying fallback query without date restrictions")
                fallback_params = {'limit': min(limit, 100)}
                
                self._respect_rate_limits()
                fallback_response = self.session.get(f"{data_sources.FDA_BASE_URL}/drug/event.json", 
                                                   params=fallback_params, timeout=60)
                
                if fallback_response.status_code == 200:
                    response = fallback_response
                    self.logger.info("✅ Fallback query successful")
                else:
                    response.raise_for_status()
            
            data = response.json()
            results = data.get('results', [])
            
            if not results:
                self.logger.warning("No results found for the specified criteria")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.json_normalize(results)
            self.logger.info(f"✅ Fetched {len(df)} adverse event records")
            
            # Log some basic info about the data
            self.logger.info(f"Columns found: {list(df.columns)[:10]}...")  # First 10 columns
            
            return df
            
        except requests.exceptions.HTTPError as e:
            self.logger.error(f"❌ HTTP Error: {e}")
            self.logger.error(f"Response status code: {response.status_code}")
            self.logger.error(f"Response content: {response.text[:500]}")
            return pd.DataFrame()
        except requests.exceptions.RequestException as e:
            self.logger.error(f"❌ Request Error: {e}")
            return pd.DataFrame()
        except Exception as e:
            self.logger.error(f"❌ Unexpected error: {e}")
            return pd.DataFrame()
    
    def load_faers_file_to_dataframe(self, file_path: Path, file_type: str) -> pd.DataFrame:
        """
        Load a FAERS text file into a pandas DataFrame
        
        Args:
            file_path: Path to the extracted FAERS file
            file_type: Type of file (DEMO, REAC, DRUG, OUTC)
            
        Returns:
            DataFrame with the file data
        """
        try:
            self.logger.info(f"📊 Loading {file_type} data from {file_path.name}")
            
            # FAERS files are typically pipe-delimited
            df = pd.read_csv(file_path, sep='$', encoding='utf-8', low_memory=False, on_bad_lines='skip')
            
            self.logger.info(f"✅ Loaded {len(df)} rows and {len(df.columns)} columns")
            self.logger.info(f"Columns: {list(df.columns)}")
            
            # Basic data info
            self.logger.info(f"Memory usage: {df.memory_usage().sum() / 1024 / 1024:.1f} MB")
            
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load {file_path}: {e}")
            return pd.DataFrame()

# Utility function for easy usage
def quick_test():
    """Quick test function to verify FDA collector is working"""
    print("🧪 Testing FDA Data Collector...")
    
    collector = FDADataCollector()
    
    # Test API connection
    if collector.test_api_connection():
        print("✅ API connection test passed")
    else:
        print("❌ API connection test failed")
        return False
    
    # Test fetching a small sample with older, more reliable dates
    try:
        print("🔍 Testing with 2024 Q1 data...")
        df = collector.fetch_recent_events_api(
            start_date="20240101", 
            end_date="20240131", 
            limit=5
        )
        
        if not df.empty:
            print(f"✅ Sample data fetch successful: {len(df)} records")
            print(f"Sample columns: {list(df.columns)[:5]}...")
            return True
        else:
            print("⚠️  No sample data returned, trying fallback...")
            # Try without date restrictions
            df_fallback = collector.fetch_recent_events_api(limit=3)
            if not df_fallback.empty:
                print(f"✅ Fallback data fetch successful: {len(df_fallback)} records")
                return True
            else:
                print("❌ No data available from API")
                return False
                
    except Exception as e:
        print(f"❌ Sample data fetch failed: {e}")
        return False

if __name__ == "__main__":
    # Run quick test if script is executed directly
    quick_test()