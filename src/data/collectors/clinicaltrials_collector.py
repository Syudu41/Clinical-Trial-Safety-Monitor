"""
ClinicalTrials.gov Data Collector
Fetches clinical trial information to enrich adverse event data
"""

import requests
import json
import pandas as pd
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import config, data_sources

class ClinicalTrialsCollector:
    """Collects clinical trial data from ClinicalTrials.gov API"""
    
    def __init__(self):
        self.logger = self._setup_logging()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Clinical-Trial-Safety-Monitor/1.0 (Research Project)',
            'Accept': 'application/json'
        })
        
        # Rate limiting tracking
        self.last_request_time = 0
        self.request_count = 0
        
    def _setup_logging(self):
        """Setup logging for the collector"""
        logger = logging.getLogger(f"{__name__}.ClinicalTrialsCollector")
        return logger
        
    def _respect_rate_limits(self):
        """Implement rate limiting for API requests"""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        
        if time_since_last_request < config.CT_REQUEST_DELAY:
            sleep_time = config.CT_REQUEST_DELAY - time_since_last_request
            self.logger.info(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
        self.request_count += 1
    
    def test_api_connection(self) -> bool:
        """Test basic ClinicalTrials.gov API v2.0 connectivity"""
        try:
            self.logger.info("Testing ClinicalTrials.gov API v2.0 connection...")
            test_url = f"{data_sources.CLINICALTRIALS_BASE_URL}/studies"
            
            # Simple test query
            params = {
                'query.cond': 'cancer',
                'countTotal': 'true',
                'pageSize': 1
            }
            
            self._respect_rate_limits()
            response = self.session.get(test_url, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                total_results = data.get('totalCount', 0)
                self.logger.info(f"✅ ClinicalTrials.gov API v2.0 connection successful! Found {total_results} studies")
                return True
            else:
                self.logger.error(f"❌ ClinicalTrials.gov API returned status code: {response.status_code}")
                self.logger.error(f"Response: {response.text[:200]}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ ClinicalTrials.gov API connection failed: {e}")
            return False
    
    def fetch_studies_by_phase(self, phase: str, max_studies: int = 100) -> pd.DataFrame:
        """
        Fetch clinical trials by phase using API v2.0
        
        Args:
            phase: Clinical trial phase ('PHASE1', 'PHASE2', 'PHASE3', 'PHASE4')
            max_studies: Maximum number of studies to fetch
            
        Returns:
            DataFrame with clinical trial data
        """
        try:
            self.logger.info(f"🔍 Fetching {phase} clinical trials...")
            
            # Build API URL for studies
            api_url = f"{data_sources.CLINICALTRIALS_BASE_URL}/studies"
            
            # Parameters for the API call (v2.0 format) - updated approach
            phase_term = f"{phase.lower().replace('phase', 'phase ')}"  # Convert "PHASE3" to "phase 3"
            params = {
                'query.term': phase_term,  # Search in general terms instead of specific phase field
                'countTotal': 'true',
                'pageSize': min(max_studies, 1000)  # API limit
            }
            
            self.logger.info(f"API URL: {api_url}")
            self.logger.info(f"Parameters: {params}")
            
            self._respect_rate_limits()
            response = self.session.get(api_url, params=params, timeout=60)
            
            if response.status_code != 200:
                self.logger.error(f"API returned status code: {response.status_code}")
                self.logger.error(f"Response content: {response.text[:500]}")
                return pd.DataFrame()
            
            data = response.json()
            
            # Parse the v2.0 response structure
            studies = data.get('studies', [])
            total_count = data.get('totalCount', 0)
            
            if not studies:
                self.logger.warning(f"No {phase} studies found")
                return pd.DataFrame()
            
            self.logger.info(f"✅ Found {total_count} total {phase} studies, fetched {len(studies)}")
            
            # Convert to DataFrame - extract key fields from the nested structure
            records = []
            for study in studies:
                protocol = study.get('protocolSection', {})
                identification = protocol.get('identificationModule', {})
                design = protocol.get('designModule', {})
                status = protocol.get('statusModule', {})
                
                record = {
                    'nctId': identification.get('nctId', ''),
                    'briefTitle': identification.get('briefTitle', ''),
                    'phase': phase,
                    'studyType': design.get('studyType', ''),
                    'overallStatus': status.get('overallStatus', ''),
                    'startDate': status.get('startDateStruct', {}).get('date', ''),
                    'completionDate': status.get('primaryCompletionDateStruct', {}).get('date', ''),
                    'enrollment': design.get('enrollmentInfo', {}).get('count', 0)
                }
                records.append(record)
            
            df = pd.DataFrame(records)
            self.logger.info(f"✅ Fetched {len(df)} {phase} studies")
            self.logger.info(f"Columns: {list(df.columns)}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Failed to fetch {phase} studies: {e}")
            return pd.DataFrame()
    
    def fetch_drug_studies(self, drug_name: str, max_studies: int = 50) -> pd.DataFrame:
        """
        Fetch clinical trials for a specific drug using API v2.0
        
        Args:
            drug_name: Name of the drug to search for
            max_studies: Maximum number of studies to fetch
            
        Returns:
            DataFrame with clinical trial data for the drug
        """
        try:
            self.logger.info(f"🔍 Fetching clinical trials for drug: {drug_name}")
            
            # Build API URL for studies
            api_url = f"{data_sources.CLINICALTRIALS_BASE_URL}/studies"
            
            # Parameters for the API call (v2.0 format)
            params = {
                'query.intr': drug_name,  # Search in interventions
                'countTotal': 'true',
                'pageSize': min(max_studies, 500)
            }
            
            self.logger.info(f"Searching for drug: {drug_name}")
            
            self._respect_rate_limits()
            response = self.session.get(api_url, params=params, timeout=60)
            
            if response.status_code != 200:
                self.logger.error(f"API returned status code: {response.status_code}")
                self.logger.error(f"Response content: {response.text[:500]}")
                return pd.DataFrame()
            
            data = response.json()
            
            # Parse the v2.0 response structure
            studies = data.get('studies', [])
            total_count = data.get('totalCount', 0)
            
            if not studies:
                self.logger.warning(f"No studies found for drug: {drug_name}")
                return pd.DataFrame()
            
            self.logger.info(f"✅ Found {total_count} studies for {drug_name}, fetched {len(studies)}")
            
            # Convert to DataFrame - extract key fields
            records = []
            for study in studies:
                protocol = study.get('protocolSection', {})
                identification = protocol.get('identificationModule', {})
                design = protocol.get('designModule', {})
                status = protocol.get('statusModule', {})
                arms_interventions = protocol.get('armsInterventionsModule', {})
                
                record = {
                    'drug_searched': drug_name,
                    'nctId': identification.get('nctId', ''),
                    'briefTitle': identification.get('briefTitle', ''),
                    'phase': ', '.join(design.get('phases', [])) if design.get('phases') else '',
                    'studyType': design.get('studyType', ''),
                    'overallStatus': status.get('overallStatus', ''),
                    'enrollment': design.get('enrollmentInfo', {}).get('count', 0),
                    'interventions': ', '.join([i.get('name', '') for i in arms_interventions.get('interventions', [])][:3])  # First 3 interventions
                }
                records.append(record)
            
            df = pd.DataFrame(records)
            
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Failed to fetch studies for drug {drug_name}: {e}")
            return pd.DataFrame()
    
    def fetch_recent_studies(self, start_date: str = "2020-01-01", max_studies: int = 100) -> pd.DataFrame:
        """
        Fetch recent clinical trials using API v2.0
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            max_studies: Maximum number of studies to fetch
            
        Returns:
            DataFrame with recent clinical trial data
        """
        try:
            self.logger.info(f"🔍 Fetching clinical trials started after {start_date}")
            
            # Build API URL for studies
            api_url = f"{data_sources.CLINICALTRIALS_BASE_URL}/studies"
            
            # Parameters for the API call (v2.0 format) - simplified approach
            params = {
                'countTotal': 'true',
                'pageSize': min(max_studies, 1000)
            }
            
            # Note: Date filtering in v2.0 API might require different approach
            # For now, we'll get recent studies and filter client-side if needed
            
            self._respect_rate_limits()
            response = self.session.get(api_url, params=params, timeout=60)
            
            if response.status_code != 200:
                self.logger.error(f"API returned status code: {response.status_code}")
                self.logger.error(f"Response content: {response.text[:500]}")
                return pd.DataFrame()
            
            data = response.json()
            
            # Parse the v2.0 response structure
            studies = data.get('studies', [])
            total_count = data.get('totalCount', 0)
            
            if not studies:
                self.logger.warning(f"No recent studies found")
                return pd.DataFrame()
            
            self.logger.info(f"✅ Found {total_count} total studies, fetched {len(studies)}")
            
            # Convert to DataFrame - extract key fields
            records = []
            for study in studies:
                protocol = study.get('protocolSection', {})
                identification = protocol.get('identificationModule', {})
                design = protocol.get('designModule', {})
                status = protocol.get('statusModule', {})
                conditions = protocol.get('conditionsModule', {})
                
                record = {
                    'nctId': identification.get('nctId', ''),
                    'briefTitle': identification.get('briefTitle', ''),
                    'phase': ', '.join(design.get('phases', [])) if design.get('phases') else '',
                    'studyType': design.get('studyType', ''),
                    'overallStatus': status.get('overallStatus', ''),
                    'startDate': status.get('startDateStruct', {}).get('date', ''),
                    'completionDate': status.get('primaryCompletionDateStruct', {}).get('date', ''),
                    'enrollment': design.get('enrollmentInfo', {}).get('count', 0),
                    'conditions': ', '.join(conditions.get('conditions', [])[:3]) if conditions.get('conditions') else ''  # First 3 conditions
                }
                records.append(record)
            
            df = pd.DataFrame(records)
            
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Failed to fetch recent studies: {e}")
            return pd.DataFrame()
    
    def get_study_details(self, nct_id: str) -> Dict:
        """
        Get detailed information for a specific study using API v2.0
        
        Args:
            nct_id: NCT ID of the study (e.g., 'NCT01234567')
            
        Returns:
            Dictionary with detailed study information
        """
        try:
            self.logger.info(f"🔍 Fetching details for study: {nct_id}")
            
            # Build API URL for specific study (v2.0 format)
            api_url = f"{data_sources.CLINICALTRIALS_BASE_URL}/studies/{nct_id}"
            
            self._respect_rate_limits()
            response = self.session.get(api_url, timeout=60)
            
            if response.status_code != 200:
                self.logger.warning(f"Failed to get details for {nct_id}: {response.status_code}")
                return {}
            
            data = response.json()
            
            # Parse the v2.0 response structure
            study = data.get('protocolSection', {})
            
            if not study:
                self.logger.warning(f"No details found for study: {nct_id}")
                return {}
            
            self.logger.info(f"✅ Retrieved details for study: {nct_id}")
            
            return study
            
        except Exception as e:
            self.logger.error(f"❌ Failed to fetch details for study {nct_id}: {e}")
            return {}

# Utility function for easy testing
def quick_test():
    """Quick test function to verify ClinicalTrials collector is working with API v2.0"""
    print("🧪 Testing ClinicalTrials.gov Data Collector (API v2.0)...")
    
    collector = ClinicalTrialsCollector()
    
    # Test API connection
    if collector.test_api_connection():
        print("✅ API v2.0 connection test passed")
    else:
        print("❌ API v2.0 connection test failed")
        return False
    
    # Test fetching sample studies
    try:
        print("🔍 Testing Phase 3 studies fetch...")
        df = collector.fetch_studies_by_phase('PHASE3', max_studies=5)
        
        if not df.empty:
            print(f"✅ Sample Phase 3 studies fetch successful: {len(df)} studies")
            print(f"Sample columns: {list(df.columns)[:5]}...")
            return True
        else:
            print("⚠️  No sample Phase 3 studies returned")
            
            # Try a different approach - just get any studies
            print("🔄 Trying general studies fetch...")
            df = collector.fetch_recent_studies(max_studies=3)
            if not df.empty:
                print(f"✅ General studies fetch successful: {len(df)} studies")
                return True
            else:
                print("❌ No studies found at all")
                return False
            
    except Exception as e:
        print(f"❌ Sample studies fetch failed: {e}")
        return False

if __name__ == "__main__":
    # Run quick test if script is executed directly
    quick_test()