"""
Exploratory Data Analysis (EDA) Analyzer
Generates insights and visualizations from cleaned FDA adverse event data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
from datetime import datetime
import logging
from typing import Dict, List

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import config

class EDAAnalyzer:
    """Performs exploratory data analysis on cleaned adverse event data"""
    
    def __init__(self):
        self.logger = self._setup_logging()
        
        # Set up matplotlib style
        plt.style.use('default')
        sns.set_palette("husl")
        
    def _setup_logging(self):
        """Setup logging for the analyzer"""
        logger = logging.getLogger(f"{__name__}.EDAAnalyzer")
        return logger
    
    def analyze_sample_data(self, df: pd.DataFrame) -> Dict:
        """
        Perform comprehensive EDA on the cleaned sample data
        
        Args:
            df: Cleaned adverse events DataFrame
            
        Returns:
            Dictionary containing analysis results and insights
        """
        self.logger.info(f"🔍 Starting EDA analysis on {len(df)} records")
        
        insights = {
            'data_overview': self._analyze_data_overview(df),
            'target_analysis': self._analyze_target_variable(df),
            'risk_analysis': self._analyze_risk_factors(df),
            'temporal_analysis': self._analyze_temporal_patterns(df),
            'business_insights': self._generate_business_insights(df)
        }
        
        self.logger.info("✅ EDA analysis complete")
        return insights
    
    def _analyze_data_overview(self, df: pd.DataFrame) -> Dict:
        """Analyze basic data characteristics"""
        overview = {
            'total_records': len(df),
            'total_features': len(df.columns),
            'data_quality_score': self._calculate_data_quality_score(df),
            'feature_types': {
                'numeric': len(df.select_dtypes(include=['number']).columns),
                'categorical': len(df.select_dtypes(include=['object']).columns)
            }
        }
        
        # Memory and processing info
        memory_usage = df.memory_usage(deep=True).sum() / 1024  # KB
        overview['memory_usage_kb'] = round(memory_usage, 2)
        
        return overview
    
    def _analyze_target_variable(self, df: pd.DataFrame) -> Dict:
        """Analyze the target variable distribution"""
        if 'is_serious' not in df.columns:
            return {'error': 'Target variable is_serious not found'}
        
        target_counts = df['is_serious'].value_counts()
        total_events = len(df)
        
        analysis = {
            'total_events': total_events,
            'serious_events': target_counts.get(1, 0),
            'non_serious_events': target_counts.get(0, 0),
            'serious_rate_percent': round((target_counts.get(1, 0) / total_events) * 100, 1),
            'class_balance': 'balanced' if abs(target_counts.get(1, 0) - target_counts.get(0, 0)) <= 1 else 'imbalanced'
        }
        
        return analysis
    
    def _analyze_risk_factors(self, df: pd.DataFrame) -> Dict:
        """Analyze risk factors and scores"""
        risk_features = ['patient_risk_score', 'drug_risk_score', 'outcome_severity']
        risk_analysis = {}
        
        for feature in risk_features:
            if feature in df.columns:
                stats = df[feature].describe()
                risk_analysis[feature] = {
                    'mean': round(stats['mean'], 3),
                    'std': round(stats['std'], 3),
                    'min': round(stats['min'], 3),
                    'max': round(stats['max'], 3),
                    'non_zero_count': (df[feature] > 0).sum()
                }
            else:
                risk_analysis[feature] = {'error': f'{feature} not found'}
        
        return risk_analysis
    
    def _analyze_temporal_patterns(self, df: pd.DataFrame) -> Dict:
        """Analyze temporal patterns in the data"""
        temporal_analysis = {}
        
        # Reporting lag analysis
        if 'reporting_lag_days' in df.columns:
            lag_stats = df['reporting_lag_days'].describe()
            temporal_analysis['reporting_lag'] = {
                'mean_days': round(lag_stats['mean'], 1),
                'max_days': round(lag_stats['max'], 1),
                'quick_reports': (df['reporting_lag_days'] <= 1).sum(),
                'delayed_reports': (df['reporting_lag_days'] > 7).sum()
            }
        
        # Date analysis if available
        if 'receivedate' in df.columns:
            # Try to parse dates
            try:
                df_temp = df.copy()
                df_temp['receive_datetime'] = pd.to_datetime(df_temp['receivedate'], format='%Y%m%d', errors='coerce')
                valid_dates = df_temp['receive_datetime'].dropna()
                
                if len(valid_dates) > 0:
                    temporal_analysis['date_range'] = {
                        'earliest_date': valid_dates.min().strftime('%Y-%m-%d'),
                        'latest_date': valid_dates.max().strftime('%Y-%m-%d'),
                        'date_span_days': (valid_dates.max() - valid_dates.min()).days
                    }
            except Exception as e:
                temporal_analysis['date_analysis_error'] = str(e)
        
        return temporal_analysis
    
    def _generate_business_insights(self, df: pd.DataFrame) -> List[str]:
        """Generate business-relevant insights from the analysis (without emojis)"""
        insights = []
        
        # Data volume insights
        total_records = len(df)
        if total_records < 10:
            insights.append(f"Small sample size ({total_records} records) - suitable for proof-of-concept development")
        elif total_records < 100:
            insights.append(f"Medium sample size ({total_records} records) - good for initial model training")
        else:
            insights.append(f"Large sample size ({total_records} records) - excellent for robust model training")
        
        # Target variable insights
        if 'is_serious' in df.columns:
            serious_rate = (df['is_serious'].sum() / len(df)) * 100
            if serious_rate == 0:
                insights.append("Current sample contains only non-serious events - need diverse data for balanced ML training")
            elif serious_rate < 10:
                insights.append(f"Low serious event rate ({serious_rate:.1f}%) - typical for post-market surveillance data")
            elif serious_rate > 50:
                insights.append(f"High serious event rate ({serious_rate:.1f}%) - may indicate focus on high-risk drugs/populations")
        
        # Data quality insights
        missing_data_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        if missing_data_pct == 0:
            insights.append("Perfect data quality - no missing values detected after cleaning")
        elif missing_data_pct < 5:
            insights.append(f"High data quality - minimal missing values ({missing_data_pct:.1f}%)")
        
        # Risk score insights
        if 'outcome_severity' in df.columns:
            max_severity = df['outcome_severity'].max()
            if max_severity == 0:
                insights.append("Current sample shows low-severity outcomes - suitable for safety baseline establishment")
            else:
                insights.append(f"Severity scores range 0-{max_severity} - good for outcome prediction modeling")
        
        # Feature engineering insights
        feature_count = len(df.columns)
        if feature_count >= 20:
            insights.append(f"Rich feature set ({feature_count} features) - excellent foundation for ML model performance")
        
        # Scalability insights
        insights.append("Pipeline successfully processes FDA data - ready to scale to larger datasets")
        insights.append("Recommend collecting 1000+ records for production-ready ML model training")
        
        return insights
    
    def _calculate_data_quality_score(self, df: pd.DataFrame) -> float:
        """Calculate overall data quality score (0-100)"""
        total_cells = len(df) * len(df.columns)
        missing_cells = df.isnull().sum().sum()
        
        if total_cells == 0:
            return 0.0
        
        completeness_score = ((total_cells - missing_cells) / total_cells) * 100
        return round(completeness_score, 1)
    
    def create_summary_report(self, insights: Dict) -> str:
        """Create a formatted summary report without emojis"""
        report = []
        report.append("=" * 80)
        report.append("CLINICAL TRIAL SAFETY MONITORING - EDA REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Data Overview
        overview = insights.get('data_overview', {})
        report.append("DATA OVERVIEW")
        report.append("-" * 40)
        report.append(f"Total Records: {overview.get('total_records', 'N/A')}")
        report.append(f"Total Features: {overview.get('total_features', 'N/A')}")
        report.append(f"Data Quality Score: {overview.get('data_quality_score', 'N/A')}/100")
        report.append(f"Memory Usage: {overview.get('memory_usage_kb', 'N/A')} KB")
        report.append("")
        
        # Target Analysis
        target = insights.get('target_analysis', {})
        report.append("TARGET VARIABLE ANALYSIS")
        report.append("-" * 40)
        report.append(f"Total Events: {target.get('total_events', 'N/A')}")
        report.append(f"Serious Events: {target.get('serious_events', 'N/A')}")
        report.append(f"Non-Serious Events: {target.get('non_serious_events', 'N/A')}")
        report.append(f"Serious Event Rate: {target.get('serious_rate_percent', 'N/A')}%")
        report.append(f"Class Balance: {target.get('class_balance', 'N/A').title()}")
        report.append("")
        
        # Risk Analysis
        risk = insights.get('risk_analysis', {})
        report.append("RISK FACTOR ANALYSIS")
        report.append("-" * 40)
        for risk_factor, stats in risk.items():
            if 'error' not in stats:
                report.append(f"{risk_factor.replace('_', ' ').title()}:")
                report.append(f"  Mean: {stats.get('mean', 'N/A')}")
                report.append(f"  Range: {stats.get('min', 'N/A')} - {stats.get('max', 'N/A')}")
                report.append(f"  Non-zero: {stats.get('non_zero_count', 'N/A')} records")
        report.append("")
        
        # Business Insights - Remove emojis
        business = insights.get('business_insights', [])
        report.append("BUSINESS INSIGHTS")
        report.append("-" * 40)
        for i, insight in enumerate(business, 1):
            # Remove emojis from insights
            clean_insight = self._remove_emojis(insight)
            report.append(f"{i}. {clean_insight}")
        report.append("")
        
        report.append("=" * 80)
        report.append("READY FOR MACHINE LEARNING MODEL DEVELOPMENT")
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def _remove_emojis(self, text: str) -> str:
        """Remove emojis from text to prevent encoding issues"""
        import re
        # Remove emojis and other unicode symbols
        clean_text = re.sub(r'[^\x00-\x7F]+', '', text)
        # Clean up any extra spaces
        clean_text = ' '.join(clean_text.split())
        return clean_text

def quick_eda_analysis():
    """Quick function to run EDA analysis on processed sample data"""
    print("🔍 Running EDA Analysis on Processed Data...")
    
    try:
        # Load processed data
        processed_file = config.PROCESSED_DATA_DIR / "processed_fda_sample.csv"
        
        if not processed_file.exists():
            print(f"❌ Processed data not found at {processed_file}")
            print("Please run the data cleaning test first")
            return False
        
        # Load and analyze data
        analyzer = EDAAnalyzer()
        
        print(f"📊 Loading processed data from {processed_file}")
        df = pd.read_csv(processed_file)
        print(f"Data shape: {df.shape}")
        
        # Run EDA analysis
        print("🔍 Performing EDA analysis...")
        insights = analyzer.analyze_sample_data(df)
        
        # Generate and display report
        report = analyzer.create_summary_report(insights)
        print(report)
        
        # Save report with proper encoding and error handling
        report_file = config.PROCESSED_DATA_DIR / "eda_report.txt"
        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"\nReport saved to: {report_file}")
        except UnicodeEncodeError:
            # Fallback: save as ASCII only
            ascii_report = report.encode('ascii', errors='ignore').decode('ascii')
            with open(report_file, 'w', encoding='ascii') as f:
                f.write(ascii_report)
            print(f"\nReport saved to: {report_file} (ASCII mode)")
        
        return True
        
    except Exception as e:
        print(f"❌ EDA analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Run quick EDA if script is executed directly
    quick_eda_analysis()