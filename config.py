"""
Clinical Trial Safety Monitoring System Configuration
Centralized configuration management for all project components
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Main configuration class"""
    
    # Project Structure
    PROJECT_ROOT = Path(__file__).parent.absolute()
    SRC_DIR = PROJECT_ROOT / "src"
    DATA_DIR = PROJECT_ROOT / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    MODELS_DIR = DATA_DIR / "models"
    LOGS_DIR = PROJECT_ROOT / "logs"
    NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
    
    # Create directories if they don't exist
    for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, LOGS_DIR, NOTEBOOKS_DIR]:
        directory.mkdir(exist_ok=True)
    
    # FDA API Settings
    FDA_API_RATE_LIMIT = 240  # requests per hour
    FDA_API_DAILY_LIMIT = 1000  # requests per day
    FDA_REQUEST_DELAY = 15  # seconds between requests to respect rate limits
    
    # ClinicalTrials API Settings  
    CT_API_RATE_LIMIT = 1000  # reasonable limit (no official limit)
    CT_REQUEST_DELAY = 1  # seconds between requests
    
    # Data Processing Settings
    BATCH_SIZE = 1000  # records to process at once
    MAX_RECORDS_PER_FILE = 50000  # for Day 1 development
    RANDOM_SEED = 42  # for reproducibility
    
    # Database Configuration
    DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://localhost:5432/clinical_safety')
    SQLITE_DB_PATH = DATA_DIR / "clinical_safety.db"  # fallback option
    
    # Feature Engineering Settings
    MIN_PATIENT_AGE = 0
    MAX_PATIENT_AGE = 120
    SERIOUS_OUTCOME_CODES = ['DE', 'LT', 'HO', 'DS', 'CA', 'RI']  # FDA outcome codes
    
    # Logging Configuration
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    LOG_FILE = LOGS_DIR / 'clinical_safety.log'

class DataSources:
    """Data source specific configurations"""
    
    # API Base URLs
    FDA_BASE_URL = "https://api.fda.gov"
    CLINICALTRIALS_BASE_URL = "https://clinicaltrials.gov/api"
    
    # FDA FAERS Quarterly Data URLs (recent quarters)
    FAERS_QUARTERS = {
        '2024Q1': 'https://fis.fda.gov/content/Exports/faers_ascii_2024q1.zip',
        '2024Q2': 'https://fis.fda.gov/content/Exports/faers_ascii_2024q2.zip',
        '2024Q3': 'https://fis.fda.gov/content/Exports/faers_ascii_2024q3.zip'
    }
    
    # FAERS file structure (files within each quarterly ZIP)
    FAERS_FILES = {
        'DEMO': 'demographics',      # Patient demographics
        'DRUG': 'drug_info',         # Drug information  
        'REAC': 'reactions',         # Adverse reactions
        'OUTC': 'outcomes',          # Patient outcomes
        'RPSR': 'report_sources',    # Reporting sources
        'THER': 'therapy',           # Drug therapy info
        'INDI': 'indications'        # Drug indications
    }
    
    # Sample API endpoints for testing
    FDA_TEST_ENDPOINT = "/drug/event.json?search=receivedate:[20240101+TO+20241231]&limit=10"
    CT_TEST_ENDPOINT = "/query/study_fields?expr=AREA[Phase]&fields=NCTId,BriefTitle,Phase&fmt=json&max_rnk=10"

class MLConfig:
    """Machine Learning specific configurations"""
    
    # Model Parameters
    RANDOM_FOREST_PARAMS = {
        'n_estimators': 100,
        'max_depth': 10,
        'random_state': Config.RANDOM_SEED,
        'n_jobs': -1
    }
    
    GRADIENT_BOOSTING_PARAMS = {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 6,
        'random_state': Config.RANDOM_SEED
    }
    
    # Target variable
    TARGET_COLUMN = 'is_serious'
    
    # Feature categories
    DEMOGRAPHIC_FEATURES = ['age_group', 'gender', 'weight_group']
    DRUG_FEATURES = ['drug_risk_score', 'therapeutic_class', 'generic_flag']
    TEMPORAL_FEATURES = ['days_since_approval', 'reporting_lag_days']
    EVENT_FEATURES = ['symptom_severity', 'outcome_severity', 'event_duration']
    
    # Train/validation split
    TEST_SIZE = 0.2
    VALIDATION_SIZE = 0.1

class KafkaConfig:
    """Kafka streaming configuration (for Day 2+)"""
    
    # Kafka Broker Settings
    BOOTSTRAP_SERVERS = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')
    
    # Topic Configuration
    TOPICS = {
        'adverse_events': 'adverse-events',
        'safety_alerts': 'safety-alerts', 
        'processed_events': 'processed-events'
    }
    
    # Producer Settings
    PRODUCER_CONFIG = {
        'bootstrap_servers': BOOTSTRAP_SERVERS,
        'value_serializer': 'json'
    }
    
    # Consumer Settings
    CONSUMER_CONFIG = {
        'bootstrap_servers': BOOTSTRAP_SERVERS,
        'group_id': 'safety-monitor',
        'value_deserializer': 'json'
    }

class AWSConfig:
    """AWS configuration (for Day 3+)"""
    
    # AWS Credentials (use environment variables)
    AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
    AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
    AWS_REGION = os.getenv('AWS_REGION', 'us-east-1')
    
    # S3 Configuration
    S3_BUCKET = os.getenv('S3_BUCKET', 'clinical-safety-data')
    
    # Lambda Configuration
    LAMBDA_FUNCTION_NAME = os.getenv('LAMBDA_FUNCTION_NAME', 'safety-event-processor')

# Export main config instance
config = Config()
data_sources = DataSources()
ml_config = MLConfig()
kafka_config = KafkaConfig()
aws_config = AWSConfig()