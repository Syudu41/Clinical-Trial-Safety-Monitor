"""
Clinical Trial Safety Monitoring System Setup Script
Run this to initialize your development environment and verify configuration
"""

import os
import sys
import logging
from pathlib import Path

# Add src to Python path
sys.path.append(str(Path(__file__).parent / 'src'))

try:
    from config import config, data_sources
    print("✅ Configuration loaded successfully")
except ImportError as e:
    print(f"❌ Error importing configuration: {e}")
    sys.exit(1)

def setup_logging():
    """Configure logging system"""
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL),
        format=config.LOG_FORMAT,
        handlers=[
            logging.FileHandler(config.LOG_FILE),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def create_directories():
    """Create necessary project directories"""
    directories = [
        config.DATA_DIR,
        config.RAW_DATA_DIR,
        config.PROCESSED_DATA_DIR,
        config.MODELS_DIR,
        config.LOGS_DIR,
        config.NOTEBOOKS_DIR
    ]
    
    for directory in directories:
        directory.mkdir(exist_ok=True)
        print(f"✅ Directory created/verified: {directory}")

def verify_dependencies():
    """Verify required packages are installed"""
    required_packages = [
        'pandas', 'numpy', 'requests', 'matplotlib', 
        'seaborn', 'scikit-learn', 'python-dotenv'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package} is installed")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} is NOT installed")
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    return True

def test_api_connectivity():
    """Test basic API connectivity"""
    import requests
    
    # Test FDA API
    try:
        fda_url = f"{data_sources.FDA_BASE_URL}/drug/event.json?limit=1"
        response = requests.get(fda_url, timeout=10)
        if response.status_code == 200:
            print("✅ FDA API is accessible")
        else:
            print(f"⚠️  FDA API returned status code: {response.status_code}")
    except Exception as e:
        print(f"❌ FDA API test failed: {e}")
    
    # Test ClinicalTrials API
    try:
        ct_url = f"{data_sources.CLINICALTRIALS_BASE_URL}/query/study_fields?expr=AREA[Phase]&fields=NCTId&fmt=json&max_rnk=1"
        response = requests.get(ct_url, timeout=10)
        if response.status_code == 200:
            print("✅ ClinicalTrials.gov API is accessible")
        else:
            print(f"⚠️  ClinicalTrials.gov API returned status code: {response.status_code}")
    except Exception as e:
        print(f"❌ ClinicalTrials.gov API test failed: {e}")

def display_configuration():
    """Display current configuration settings"""
    print("\n" + "="*50)
    print("CURRENT CONFIGURATION")
    print("="*50)
    print(f"Project Root: {config.PROJECT_ROOT}")
    print(f"Data Directory: {config.DATA_DIR}")
    print(f"Log Level: {config.LOG_LEVEL}")
    print(f"Batch Size: {config.BATCH_SIZE}")
    print(f"Random Seed: {config.RANDOM_SEED}")
    print(f"FDA Rate Limit: {config.FDA_API_RATE_LIMIT}/hour")
    print("="*50)

def main():
    """Main setup function"""
    print("🚀 Clinical Trial Safety Monitoring System Setup")
    print("="*60)
    
    # Setup logging
    logger = setup_logging()
    logger.info("Starting setup process")
    
    # Create directories
    print("\n📁 Creating project directories...")
    create_directories()
    
    # Verify dependencies
    print("\n📦 Verifying dependencies...")
    if not verify_dependencies():
        print("\n❌ Setup incomplete. Please install missing packages.")
        return False
    
    # Test API connectivity
    print("\n🌐 Testing API connectivity...")
    test_api_connectivity()
    
    # Display configuration
    display_configuration()
    
    print("\n✅ Setup completed successfully!")
    print("\nNext Steps:")
    print("1. Review and customize .env file if needed")
    print("2. Run: jupyter notebook (to start exploring)")
    print("3. Begin Day 1 data collection")
    
    logger.info("Setup process completed")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)