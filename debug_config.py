"""
Debug script to check configuration loading
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

try:
    from config import config, data_sources
    print("✅ Configuration imported successfully")
    
    # Check Config class
    print("\n📋 CONFIG ATTRIBUTES:")
    print(f"PROJECT_ROOT: {config.PROJECT_ROOT}")
    print(f"DATA_DIR: {config.DATA_DIR}")
    print(f"FDA_API_RATE_LIMIT: {config.FDA_API_RATE_LIMIT}")
    
    # Check DataSources class
    print("\n🌐 DATA SOURCES ATTRIBUTES:")
    print(f"FDA_BASE_URL: {getattr(data_sources, 'FDA_BASE_URL', 'NOT FOUND')}")
    print(f"CLINICALTRIALS_BASE_URL: {getattr(data_sources, 'CLINICALTRIALS_BASE_URL', 'NOT FOUND')}")
    print(f"FDA_TEST_ENDPOINT: {getattr(data_sources, 'FDA_TEST_ENDPOINT', 'NOT FOUND')}")
    print(f"FAERS_QUARTERS: {list(data_sources.FAERS_QUARTERS.keys()) if hasattr(data_sources, 'FAERS_QUARTERS') else 'NOT FOUND'}")
    
    # Check if all required attributes exist
    required_attrs = ['FDA_BASE_URL', 'CLINICALTRIALS_BASE_URL', 'FDA_TEST_ENDPOINT', 'FAERS_QUARTERS']
    missing_attrs = []
    
    for attr in required_attrs:
        if not hasattr(data_sources, attr):
            missing_attrs.append(attr)
    
    if missing_attrs:
        print(f"\n❌ Missing attributes in DataSources: {missing_attrs}")
    else:
        print("\n✅ All required attributes found in DataSources")
        
except ImportError as e:
    print(f"❌ Failed to import configuration: {e}")
except Exception as e:
    print(f"❌ Error checking configuration: {e}")

if __name__ == "__main__":
    print("🔍 Configuration Debug Complete")