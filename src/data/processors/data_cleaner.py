"""
Data Cleaning and Processing Pipeline
Handles FDA adverse events and ClinicalTrials data cleaning and preparation
"""

import pandas as pd
import numpy as np
import re
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import sys
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import config

class DataCleaner:
    """Cleans and processes FDA and ClinicalTrials data for ML pipeline"""
    
    def __init__(self):
        self.logger = self._setup_logging()
        
        # Serious outcome codes from FDA documentation
        self.serious_outcome_codes = config.SERIOUS_OUTCOME_CODES
        
        # Common drug name mappings (brand -> generic)
        self.drug_name_mappings = {
            'advil': 'ibuprofen',
            'tylenol': 'acetaminophen', 
            'motrin': 'ibuprofen',
            'aleve': 'naproxen',
            'aspirin': 'aspirin'  # already generic
        }
        
    def _setup_logging(self):
        """Setup logging for the cleaner"""
        logger = logging.getLogger(f"{__name__}.DataCleaner")
        return logger
    
    def clean_fda_adverse_events(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and process FDA adverse events data
        
        Args:
            df: Raw FDA adverse events DataFrame
            
        Returns:
            Cleaned DataFrame ready for analysis
        """
        if df.empty:
            self.logger.warning("Empty DataFrame provided to clean_fda_adverse_events")
            return df
            
        self.logger.info(f"🧹 Cleaning FDA adverse events data: {len(df)} records")
        
        # Make a copy to avoid modifying original
        cleaned_df = df.copy()
        
        # 1. Handle basic data types and missing values
        self.logger.info("1️⃣ Processing basic data types...")
        
        # Convert serious field to numeric (1=serious, 2=non-serious)
        if 'serious' in cleaned_df.columns:
            cleaned_df['serious'] = pd.to_numeric(cleaned_df['serious'], errors='coerce')
            cleaned_df['is_serious'] = (cleaned_df['serious'] == 1).astype(int)
        else:
            self.logger.warning("'serious' column not found - creating default")
            cleaned_df['is_serious'] = 0
        
        # 2. Clean and standardize dates
        self.logger.info("2️⃣ Processing dates...")
        date_columns = ['receivedate', 'transmissiondate', 'receiptdate']
        
        for date_col in date_columns:
            if date_col in cleaned_df.columns:
                cleaned_df[f'{date_col}_clean'] = self._clean_fda_dates(cleaned_df[date_col])
        
        # 3. Create age groups from patient data
        self.logger.info("3️⃣ Processing patient demographics...")
        if 'patient' in cleaned_df.columns:
            cleaned_df = self._extract_patient_demographics(cleaned_df)
        else:
            # Create default demographic features
            cleaned_df['age_group'] = 'unknown'
            cleaned_df['gender'] = 'unknown'
            cleaned_df['weight_group'] = 'unknown'
        
        # 4. Process drug information
        self.logger.info("4️⃣ Processing drug information...")
        if 'patient' in cleaned_df.columns:
            cleaned_df = self._extract_drug_info(cleaned_df)
        else:
            cleaned_df['primary_drug'] = 'unknown'
            cleaned_df['drug_count'] = 0
        
        # 5. Process adverse reactions
        self.logger.info("5️⃣ Processing adverse reactions...")
        if 'patient' in cleaned_df.columns:
            cleaned_df = self._extract_reaction_info(cleaned_df)
        else:
            cleaned_df['primary_reaction'] = 'unknown'
            cleaned_df['reaction_count'] = 0
        
        # 6. Create outcome severity score
        self.logger.info("6️⃣ Creating severity scores...")
        cleaned_df['outcome_severity'] = self._calculate_outcome_severity(cleaned_df)
        
        # 7. Create reporting lag features
        cleaned_df['reporting_lag_days'] = self._calculate_reporting_lag(cleaned_df)
        
        # 8. Remove unnecessary columns and handle final cleanup
        self.logger.info("7️⃣ Final cleanup...")
        cleaned_df = self._final_cleanup(cleaned_df)
        
        self.logger.info(f"✅ FDA data cleaning complete: {len(cleaned_df)} records, {len(cleaned_df.columns)} features")
        
        return cleaned_df
    
    def _clean_fda_dates(self, date_series: pd.Series) -> pd.Series:
        """Clean FDA date formats (typically YYYYMMDD)"""
        def parse_fda_date(date_val):
            if pd.isna(date_val):
                return None
            
            date_str = str(date_val).strip()
            if len(date_str) == 8 and date_str.isdigit():
                try:
                    return datetime.strptime(date_str, '%Y%m%d')
                except ValueError:
                    return None
            return None
        
        return date_series.apply(parse_fda_date)
    
    def _extract_patient_demographics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract patient demographic information from nested patient data"""
        # This is a simplified version - in real FDA data, patient info is nested
        # For now, create some demographic features based on available data
        
        df['age_group'] = 'unknown'
        df['gender'] = 'unknown' 
        df['weight_group'] = 'unknown'
        
        # If we have patient demographic fields, process them
        if 'patientonsetage' in df.columns:
            df['age_group'] = pd.cut(
                pd.to_numeric(df['patientonsetage'], errors='coerce'),
                bins=[0, 18, 35, 50, 65, 100],
                labels=['child', 'young_adult', 'adult', 'middle_age', 'senior'],
                include_lowest=True
            ).astype(str)
        
        return df
    
    def _extract_drug_info(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract drug information from patient data"""
        # Simplified drug extraction - in real data this would be more complex
        df['primary_drug'] = 'unknown'
        df['drug_count'] = 1  # Default assumption
        
        return df
    
    def _extract_reaction_info(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract adverse reaction information"""
        df['primary_reaction'] = 'unknown'
        df['reaction_count'] = 1  # Default assumption
        
        return df
    
    def _calculate_outcome_severity(self, df: pd.DataFrame) -> pd.Series:
        """Calculate outcome severity score based on available fields"""
        severity_score = pd.Series(0, index=df.index)
        
        # Check various severity indicators
        severity_fields = [
            'seriousnessdeath', 'seriousnesshospitalization', 
            'seriousnesslifethreatening', 'seriousnessdisabling'
        ]
        
        for field in severity_fields:
            if field in df.columns:
                # FDA uses '1' for Yes, missing/other for No
                field_score = pd.to_numeric(df[field], errors='coerce').fillna(0)
                severity_score += (field_score == 1).astype(int)
        
        return severity_score
    
    def _calculate_reporting_lag(self, df: pd.DataFrame) -> pd.Series:
        """Calculate lag between event and reporting"""
        lag_days = pd.Series(np.nan, index=df.index)
        
        if 'receivedate_clean' in df.columns and 'receiptdate_clean' in df.columns:
            lag_days = (df['receivedate_clean'] - df['receiptdate_clean']).dt.days
        
        return lag_days.fillna(0)  # Default to 0 if can't calculate
    
    def _final_cleanup(self, df: pd.DataFrame) -> pd.DataFrame:
        """Final cleanup and column selection"""
        # Keep essential columns for ML
        essential_columns = [
            'safetyreportid', 'is_serious', 'serious', 'receivedate',
            'age_group', 'gender', 'weight_group',
            'primary_drug', 'drug_count',
            'primary_reaction', 'reaction_count',
            'outcome_severity', 'reporting_lag_days'
        ]
        
        # Keep columns that exist
        keep_columns = [col for col in essential_columns if col in df.columns]
        
        # Add any additional important columns that exist
        additional_columns = [
            'transmissiondate', 'primarysourcecountry', 'occurcountry'
        ]
        keep_columns.extend([col for col in additional_columns if col in df.columns])
        
        return df[keep_columns].copy()
    
    def clean_clinical_trials(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and process ClinicalTrials.gov data
        
        Args:
            df: Raw ClinicalTrials DataFrame
            
        Returns:
            Cleaned DataFrame ready for analysis
        """
        if df.empty:
            self.logger.warning("Empty DataFrame provided to clean_clinical_trials")
            return df
            
        self.logger.info(f"🧹 Cleaning ClinicalTrials data: {len(df)} records")
        
        cleaned_df = df.copy()
        
        # 1. Standardize phase information
        if 'phase' in cleaned_df.columns:
            cleaned_df['phase_numeric'] = self._standardize_phase(cleaned_df['phase'])
        else:
            cleaned_df['phase_numeric'] = 0
        
        # 2. Clean enrollment data
        if 'enrollment' in cleaned_df.columns:
            cleaned_df['enrollment_clean'] = pd.to_numeric(cleaned_df['enrollment'], errors='coerce').fillna(0)
        else:
            cleaned_df['enrollment_clean'] = 0
        
        # 3. Standardize status
        if 'overallStatus' in cleaned_df.columns:
            cleaned_df['status_category'] = self._categorize_status(cleaned_df['overallStatus'])
        else:
            cleaned_df['status_category'] = 'unknown'
        
        # 4. Process study type
        if 'studyType' in cleaned_df.columns:
            cleaned_df['study_type_clean'] = cleaned_df['studyType'].fillna('unknown').str.lower()
        else:
            cleaned_df['study_type_clean'] = 'unknown'
        
        # 5. Extract year from dates
        date_columns = ['startDate', 'completionDate']
        for date_col in date_columns:
            if date_col in cleaned_df.columns:
                cleaned_df[f'{date_col}_year'] = self._extract_year_from_date(cleaned_df[date_col])
        
        self.logger.info(f"✅ ClinicalTrials data cleaning complete: {len(cleaned_df)} records")
        
        return cleaned_df
    
    def _standardize_phase(self, phase_series: pd.Series) -> pd.Series:
        """Standardize phase information to numeric values"""
        def parse_phase(phase_val):
            if pd.isna(phase_val):
                return 0
            
            phase_str = str(phase_val).lower()
            if 'phase 1' in phase_str or 'phase1' in phase_str:
                return 1
            elif 'phase 2' in phase_str or 'phase2' in phase_str:
                return 2  
            elif 'phase 3' in phase_str or 'phase3' in phase_str:
                return 3
            elif 'phase 4' in phase_str or 'phase4' in phase_str:
                return 4
            else:
                return 0  # Unknown/other
        
        return phase_series.apply(parse_phase)
    
    def _categorize_status(self, status_series: pd.Series) -> pd.Series:
        """Categorize trial status into broader categories"""
        def categorize_status(status_val):
            if pd.isna(status_val):
                return 'unknown'
            
            status_str = str(status_val).lower()
            if 'completed' in status_str:
                return 'completed'
            elif 'recruiting' in status_str or 'active' in status_str:
                return 'active'
            elif 'terminated' in status_str or 'withdrawn' in status_str:
                return 'terminated'
            else:
                return 'other'
        
        return status_series.apply(categorize_status)
    
    def _extract_year_from_date(self, date_series: pd.Series) -> pd.Series:
        """Extract year from various date formats"""
        def extract_year(date_val):
            if pd.isna(date_val):
                return None
            
            date_str = str(date_val).strip()
            
            # Try to extract 4-digit year
            year_match = re.search(r'(\d{4})', date_str)
            if year_match:
                year = int(year_match.group(1))
                if 1990 <= year <= 2030:  # Reasonable range for clinical trials
                    return year
            
            return None
        
        return date_series.apply(extract_year)
    
    def create_ml_features(self, fda_df: pd.DataFrame, ct_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Create machine learning features from cleaned data
        
        Args:
            fda_df: Cleaned FDA adverse events data
            ct_df: Cleaned ClinicalTrials data (optional)
            
        Returns:
            DataFrame with ML-ready features
        """
        self.logger.info("🔧 Creating ML features...")
        
        ml_df = fda_df.copy()
        
        # 1. Encode categorical variables
        categorical_features = ['age_group', 'gender', 'primary_drug', 'primary_reaction']
        
        for feature in categorical_features:
            if feature in ml_df.columns:
                ml_df = pd.get_dummies(ml_df, columns=[feature], prefix=feature, dummy_na=True)
        
        # 2. Create risk scores
        ml_df['patient_risk_score'] = self._calculate_patient_risk_score(ml_df)
        ml_df['drug_risk_score'] = self._calculate_drug_risk_score(ml_df)
        
        # 3. Create temporal features
        if 'receivedate' in ml_df.columns:
            ml_df['report_month'] = pd.to_datetime(ml_df['receivedate'], format='%Y%m%d', errors='coerce').dt.month
            ml_df['report_year'] = pd.to_datetime(ml_df['receivedate'], format='%Y%m%d', errors='coerce').dt.year
        
        # 4. Handle missing values
        ml_df = ml_df.fillna(0)
        
        self.logger.info(f"✅ ML features created: {len(ml_df.columns)} features")
        
        return ml_df
    
    def _calculate_patient_risk_score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate patient risk score based on demographics and history"""
        risk_score = pd.Series(0.0, index=df.index)
        
        # Age-based risk (older patients generally higher risk)
        if 'age_group_senior' in df.columns:
            risk_score += df['age_group_senior'] * 0.3
        if 'age_group_middle_age' in df.columns:
            risk_score += df['age_group_middle_age'] * 0.2
        
        # Add outcome severity
        if 'outcome_severity' in df.columns:
            risk_score += df['outcome_severity'] * 0.5
        
        return risk_score
    
    def _calculate_drug_risk_score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate drug risk score based on drug characteristics"""
        risk_score = pd.Series(1.0, index=df.index)  # Base risk of 1.0
        
        # Increase risk for multiple drugs
        if 'drug_count' in df.columns:
            risk_score += (df['drug_count'] - 1) * 0.1
        
        # Add more drug-specific risk factors as needed
        
        return risk_score

# Utility functions for easy usage
def quick_clean_sample_data():
    """Quick function to clean sample FDA data and create ML features"""
    print("🧹 Testing Data Cleaning Pipeline...")
    
    try:
        # Load sample FDA data
        sample_fda_path = config.RAW_DATA_DIR / "sample_fda_data.csv"
        
        if not sample_fda_path.exists():
            print(f"❌ Sample FDA data not found at {sample_fda_path}")
            print("Please run the FDA collector test first to generate sample data")
            return False
        
        # Load and clean data
        cleaner = DataCleaner()
        
        print(f"📊 Loading sample data from {sample_fda_path}")
        fda_df = pd.read_csv(sample_fda_path)
        print(f"Raw data shape: {fda_df.shape}")
        
        # Clean FDA data
        print("🧹 Cleaning FDA data...")
        cleaned_fda = cleaner.clean_fda_adverse_events(fda_df)
        print(f"Cleaned FDA shape: {cleaned_fda.shape}")
        
        # Create ML features
        print("🔧 Creating ML features...")
        ml_features = cleaner.create_ml_features(cleaned_fda)
        print(f"ML features shape: {ml_features.shape}")
        
        # Save processed data
        processed_file = config.PROCESSED_DATA_DIR / "processed_fda_sample.csv"
        ml_features.to_csv(processed_file, index=False)
        print(f"💾 Processed data saved to: {processed_file}")
        
        # Show summary
        print("\n📋 DATA SUMMARY:")
        print(f"Target variable distribution:")
        if 'is_serious' in ml_features.columns:
            print(ml_features['is_serious'].value_counts())
        print(f"\nTotal features: {len(ml_features.columns)}")
        print(f"Feature names: {list(ml_features.columns)[:10]}...")  # First 10 features
        
        return True
        
    except Exception as e:
        print(f"❌ Data cleaning test failed: {e}")
        return False

if __name__ == "__main__":
    # Run quick test if script is executed directly
    quick_clean_sample_data()