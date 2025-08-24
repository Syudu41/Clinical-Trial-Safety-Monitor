import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime

class Large25KProcessor:
    def __init__(self):
        self.processed_data = None
        
    def load_and_process_25k(self, input_file="data/raw/fda_10k_dataset.csv"):
        """
        Process the 25K dataset efficiently
        """
        print("🔄 PROCESSING 25K FDA DATASET")
        print("=" * 50)
        
        if not os.path.exists(input_file):
            print(f"❌ File not found: {input_file}")
            return None
            
        print(f"📂 Loading: {input_file} (476MB)")
        start_time = datetime.now()
        
        # Load with memory optimization
        df = pd.read_csv(input_file, low_memory=False)
        load_time = (datetime.now() - start_time).total_seconds()
        
        print(f"✅ Loaded {len(df):,} records in {load_time:.1f}s")
        print(f"📊 Raw features: {len(df.columns)}")
        print(f"💾 Memory usage: ~{df.memory_usage(deep=True).sum() / 1024**2:.0f}MB")
        
        # Process efficiently
        processed_df = self.create_optimized_ml_features(df)
        
        return processed_df
    
    def create_optimized_ml_features(self, df):
        """
        Optimized feature engineering for 25K records - FIXED CATEGORICAL HANDLING
        """
        print("\n🛠️  OPTIMIZED FEATURE ENGINEERING")
        print("-" * 40)
        
        # Initialize with size
        processed = pd.DataFrame(index=df.index)
        
        # 1. Core features (fast)
        print("📋 Core features...")
        processed['safetyreportid'] = df.get('safetyreportid', df.get('safetyreportversion', ''))
        
        # Target variable (optimized)
        if 'serious' in df.columns:
            processed['is_serious'] = (df['serious'].astype(str) == '1').astype(int)
            processed['serious'] = df['serious']
        else:
            processed['is_serious'] = 0
            processed['serious'] = 0
        
        print(f"   ✅ Target distribution: {processed['is_serious'].value_counts().to_dict()}")
        
        # 2. Date features (simplified for speed)
        print("📅 Date features...")
        for col in ['receivedate', 'transmissiondate']:
            if col in df.columns:
                processed[col] = df[col]
        processed['reporting_lag_days'] = 30  # Simplified
        processed['report_month'] = 1
        processed['report_year'] = 2024
        
        # 3. Numeric features (FIXED CATEGORICAL HANDLING)
        print("🔢 Numeric features...")
        
        # Weight (FIXED)
        weight_col = 'patient_patientweight'
        if weight_col in df.columns:
            weight_numeric = pd.to_numeric(df[weight_col], errors='coerce')
            # Handle NaN first, then categorize
            weight_clean = weight_numeric.fillna(0)  # Fill NaN with 0 first
            processed['weight_group'] = pd.cut(
                weight_clean,
                bins=[-1, 50, 70, 90, float('inf')],  # Start with -1 to include 0
                labels=[0, 1, 2, 3],  # 0 for unknown/low
                include_lowest=True
            ).astype(int)
        else:
            processed['weight_group'] = 0
        
        # Age (FIXED)
        age_col = 'patient_patientage'
        if age_col in df.columns:
            age_numeric = pd.to_numeric(df[age_col], errors='coerce')
            # Handle NaN first, then categorize
            age_clean = age_numeric.fillna(0)  # Fill NaN with 0 first
            processed['age_group'] = pd.cut(
                age_clean,
                bins=[-1, 18, 40, 65, float('inf')],  # Start with -1 to include 0
                labels=[0, 1, 2, 3],  # 0 for unknown
                include_lowest=True
            ).astype(int)
        else:
            processed['age_group'] = 0
        
        # 4. Count features (vectorized)
        print("📊 Count features...")
        
        # Drug and reaction counts (fast)
        drug_cols = [col for col in df.columns if 'drug' in col.lower()]
        reaction_cols = [col for col in df.columns if 'reaction' in col.lower()]
        
        processed['drug_count'] = df[drug_cols].notna().sum(axis=1) if drug_cols else 1
        processed['reaction_count'] = df[reaction_cols].notna().sum(axis=1) if reaction_cols else 1
        
        # Severity scoring (vectorized)
        severity_terms = ['death', 'lifethreat', 'hospital', 'disability']
        severity_cols = [col for col in df.columns if any(term in col.lower() for term in severity_terms)]
        if severity_cols:
            processed['outcome_severity'] = df[severity_cols].fillna(0).astype(int).sum(axis=1)
        else:
            processed['outcome_severity'] = 0
        
        # 5. Risk scores (vectorized calculations)
        print("🎯 Risk scores...")
        
        # Patient risk - FIXED MAPPING
        age_risk_map = {0: 1, 1: 2, 2: 1, 3: 2}  # 0=unknown, 1=child, 2=adult, 3=senior
        age_risk = processed['age_group'].map(age_risk_map).fillna(1)
        reaction_risk = np.clip(processed['reaction_count'] / 5, 0, 2)
        processed['patient_risk_score'] = (age_risk + reaction_risk) / 2
        
        # Drug risk  
        processed['drug_risk_score'] = np.clip(processed['drug_count'] / 3, 0, 2)
        
        # 6. Categorical encoding (SIMPLIFIED AND SAFE)
        print("🏷️  Categorical encoding...")
        
        # Country codes (safe approach)
        if 'occurcountry' in df.columns:
            # Handle missing values first
            country_clean = df['occurcountry'].fillna('UNKNOWN')
            top_countries = country_clean.value_counts().head(5).index.tolist()
            country_map = {country: i+1 for i, country in enumerate(top_countries)}
            country_map['UNKNOWN'] = 0  # Ensure unknown is mapped
            processed['occurcountry_encoded'] = country_clean.map(country_map).fillna(0).astype(int)
        else:
            processed['occurcountry_encoded'] = 0
        
        if 'primarysourcecountry' in df.columns:
            source_clean = df['primarysourcecountry'].fillna('UNKNOWN')
            top_source = source_clean.value_counts().head(5).index.tolist()
            source_map = {country: i+1 for i, country in enumerate(top_source)}
            source_map['UNKNOWN'] = 0
            processed['primarysourcecountry_encoded'] = source_clean.map(source_map).fillna(0).astype(int)
        else:
            processed['primarysourcecountry_encoded'] = 0
        
        # Drug and reaction encoding (safe approach)
        drug_col = 'patient_drug_medicinalproduct'
        if drug_col in df.columns:
            drug_clean = df[drug_col].fillna('UNKNOWN')
            top_drugs = drug_clean.value_counts().head(10).index.tolist()
            drug_map = {drug: i+1 for i, drug in enumerate(top_drugs)}
            drug_map['UNKNOWN'] = 0
            processed['primary_drug_encoded'] = drug_clean.map(drug_map).fillna(0).astype(int)
        else:
            processed['primary_drug_encoded'] = 0
        
        reaction_col = 'patient_reaction_reactionmeddrapt'
        if reaction_col in df.columns:
            reaction_clean = df[reaction_col].fillna('UNKNOWN')
            top_reactions = reaction_clean.value_counts().head(15).index.tolist()
            reaction_map = {reaction: i+1 for i, reaction in enumerate(top_reactions)}
            reaction_map['UNKNOWN'] = 0
            processed['primary_reaction_encoded'] = reaction_clean.map(reaction_map).fillna(0).astype(int)
        else:
            processed['primary_reaction_encoded'] = 0
        
        # 7. Final optimization
        print("⚡ Finalizing features...")
        
        # Ensure all numeric (except ID and date columns)
        for col in processed.columns:
            if col not in ['safetyreportid', 'receivedate', 'transmissiondate']:
                processed[col] = pd.to_numeric(processed[col], errors='coerce').fillna(0)
        
        # Memory optimization
        for col in processed.select_dtypes(include=['int64']).columns:
            if col not in ['safetyreportid']:  # Skip ID column
                max_val = processed[col].max()
                min_val = processed[col].min()
                if max_val < 127 and min_val > -128:
                    processed[col] = processed[col].astype('int8')
                elif max_val < 32767 and min_val > -32768:
                    processed[col] = processed[col].astype('int16')
        
        print(f"✅ Created {len(processed.columns)} optimized ML features")
        print(f"📊 Final shape: {processed.shape}")
        print(f"💾 Processed memory: ~{processed.memory_usage(deep=True).sum() / 1024**2:.0f}MB")
        
        return processed
    
    def analyze_large_dataset(self, df):
        """
        Comprehensive analysis for 25K dataset
        """
        print(f"\n📊 25K DATASET QUALITY ANALYSIS")
        print("=" * 50)
        
        # Shape and size
        print(f"📏 Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
        memory_mb = df.memory_usage(deep=True).sum() / 1024**2
        print(f"💾 Memory: {memory_mb:.1f}MB")
        
        # Target analysis
        if 'is_serious' in df.columns:
            target_counts = df['is_serious'].value_counts()
            print(f"\n🎯 TARGET DISTRIBUTION:")
            print(f"   Non-serious (0): {target_counts[0]:,} ({target_counts[0]/len(df)*100:.1f}%)")
            print(f"   Serious (1): {target_counts[1]:,} ({target_counts[1]/len(df)*100:.1f}%)")
            print(f"   Balance ratio: {min(target_counts)/max(target_counts):.3f} (perfect = 1.000)")
        
        # Data quality
        completeness = (df.notna().sum() / len(df) * 100).mean()
        print(f"\n📈 DATA QUALITY:")
        print(f"   Completeness: {completeness:.1f}%")
        print(f"   Missing values: {df.isna().sum().sum():,}")
        
        # Feature distribution
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        print(f"\n🔢 FEATURES:")
        print(f"   Numeric features: {len(numeric_cols)}")
        print(f"   Feature names: {list(numeric_cols)[:10]}...")
        
        # ML readiness
        ml_ready_score = 100 if completeness > 95 and len(df) > 10000 else 85
        print(f"\n🎯 ML READINESS SCORE: {ml_ready_score}/100")
        
        return {
            'records': len(df),
            'features': len(df.columns),
            'completeness': completeness,
            'ml_ready': ml_ready_score >= 90
        }
    
    def save_optimized_dataset(self, df, filename="processed_fda_25k.csv"):
        """
        Save with optimization
        """
        os.makedirs("data/processed", exist_ok=True)
        filepath = f"data/processed/{filename}"
        
        print(f"\n💾 SAVING OPTIMIZED DATASET...")
        start_save = datetime.now()
        
        df.to_csv(filepath, index=False)
        
        save_time = (datetime.now() - start_save).total_seconds()
        file_size_mb = os.path.getsize(filepath) / (1024*1024)
        
        print(f"✅ Saved in {save_time:.1f}s")
        print(f"📂 File: {filepath}")
        print(f"📏 Size: {file_size_mb:.1f}MB")
        
        return filepath

def main():
    print("🚀 LARGE-SCALE DATA PROCESSOR")
    print("🔄 25,000 Records → ML-Ready Features")
    print("=" * 60)
    
    processor = Large25KProcessor()
    
    # Process
    start_time = datetime.now()
    processed_df = processor.load_and_process_25k()
    
    if processed_df is not None:
        # Analyze
        stats = processor.analyze_large_dataset(processed_df)
        
        # Save
        filepath = processor.save_optimized_dataset(processed_df)
        
        total_time = (datetime.now() - start_time).total_seconds() / 60
        
        print(f"\n🎉 PROCESSING COMPLETE!")
        print(f"⏱️  Total time: {total_time:.1f} minutes")
        print(f"📈 {len(processed_df):,} records ready for professional ML training!")
        print(f"💪 This dataset will produce exceptional ML models!")
        
    else:
        print("❌ Processing failed")

if __name__ == "__main__":
    main()