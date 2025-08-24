import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, 
                           roc_auc_score, f1_score, precision_score, recall_score)
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import joblib
import sys
import os

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

# Now we can import your existing modules
from src.data.processors.data_cleaner import DataCleaner  

class ClinicalTrialMLPipeline:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.results = {}
        self.feature_names = []
        
    def load_processed_data(self, filepath="data/processed/processed_fda_25k.csv"):
        """
        Load the processed 25K dataset
        """
        print("📂 LOADING 25K PROCESSED DATASET")
        print("=" * 50)
        
        if not os.path.exists(filepath):
            print(f"❌ File not found: {filepath}")
            return None, None
            
        df = pd.read_csv(filepath)
        print(f"✅ Loaded {len(df):,} records with {len(df.columns)} features")
        
        # Separate features and target
        # Remove ID and date columns for ML
        exclude_cols = ['safetyreportid', 'receivedate', 'transmissiondate', 'is_serious', 'serious']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols].copy()
        y = df['is_serious'].copy()
        
        self.feature_names = feature_cols
        
        print(f"🎯 Features for ML: {len(feature_cols)}")
        print(f"📊 Target distribution: {y.value_counts().to_dict()}")
        
        return X, y
    
    def create_train_test_splits(self, X, y):
        """
        Professional train/validation/test split
        """
        print("\n📊 CREATING DATA SPLITS")
        print("=" * 40)
        
        # First split: 80% train+val, 20% test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.10, random_state=42, stratify=y
        )
        
        # Second split: 70% train, 20% validation (from the 90%)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.22, random_state=42, stratify=y_temp  # 0.22 * 0.9 ≈ 0.20
        )
        
        print(f"📈 Training Set: {len(X_train):,} records ({len(X_train)/len(X)*100:.1f}%)")
        print(f"📊 Validation Set: {len(X_val):,} records ({len(X_val)/len(X)*100:.1f}%)")  
        print(f"🧪 Test Set: {len(X_test):,} records ({len(X_test)/len(X)*100:.1f}%)")
        
        # Check balance
        print(f"\n🎯 CLASS BALANCE CHECK:")
        train_balance = y_train.value_counts(normalize=True)
        val_balance = y_val.value_counts(normalize=True)
        test_balance = y_test.value_counts(normalize=True)
        
        print(f"   Train: {train_balance[0]:.3f} non-serious, {train_balance[1]:.3f} serious")
        print(f"   Val: {val_balance[0]:.3f} non-serious, {val_balance[1]:.3f} serious")
        print(f"   Test: {test_balance[0]:.3f} non-serious, {test_balance[1]:.3f} serious")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def train_models(self, X_train, y_train, X_val, y_val):
        """
        Train multiple ML models
        """
        print("\n🤖 TRAINING ML MODELS")
        print("=" * 40)
        
        # 1. Random Forest
        print("🌳 Training Random Forest...")
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10, 
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1  # Use all CPU cores
        )
        
        start_time = datetime.now()
        rf_model.fit(X_train, y_train)
        rf_time = (datetime.now() - start_time).total_seconds()
        
        # Validation predictions
        rf_pred = rf_model.predict(X_val)
        rf_accuracy = accuracy_score(y_val, rf_pred)
        rf_f1 = f1_score(y_val, rf_pred)
        
        self.models['Random Forest'] = rf_model
        self.results['Random Forest'] = {
            'accuracy': rf_accuracy,
            'f1_score': rf_f1,
            'training_time': rf_time
        }
        
        print(f"   ✅ Accuracy: {rf_accuracy:.4f} | F1: {rf_f1:.4f} | Time: {rf_time:.1f}s")
        
        # 2. Gradient Boosting
        print("⚡ Training Gradient Boosting...")
        gb_model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            min_samples_split=5,
            random_state=42
        )
        
        start_time = datetime.now()
        gb_model.fit(X_train, y_train)
        gb_time = (datetime.now() - start_time).total_seconds()
        
        gb_pred = gb_model.predict(X_val)
        gb_accuracy = accuracy_score(y_val, gb_pred)
        gb_f1 = f1_score(y_val, gb_pred)
        
        self.models['Gradient Boosting'] = gb_model
        self.results['Gradient Boosting'] = {
            'accuracy': gb_accuracy,
            'f1_score': gb_f1,
            'training_time': gb_time
        }
        
        print(f"   ✅ Accuracy: {gb_accuracy:.4f} | F1: {gb_f1:.4f} | Time: {gb_time:.1f}s")
        
        # 3. Logistic Regression (with scaling)
        print("📈 Training Logistic Regression...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        lr_model = LogisticRegression(
            random_state=42,
            max_iter=1000
        )
        
        start_time = datetime.now()
        lr_model.fit(X_train_scaled, y_train)
        lr_time = (datetime.now() - start_time).total_seconds()
        
        lr_pred = lr_model.predict(X_val_scaled)
        lr_accuracy = accuracy_score(y_val, lr_pred)
        lr_f1 = f1_score(y_val, lr_pred)
        
        self.models['Logistic Regression'] = lr_model
        self.scalers['Logistic Regression'] = scaler
        self.results['Logistic Regression'] = {
            'accuracy': lr_accuracy,
            'f1_score': lr_f1,
            'training_time': lr_time
        }
        
        print(f"   ✅ Accuracy: {lr_accuracy:.4f} | F1: {lr_f1:.4f} | Time: {lr_time:.1f}s")
        
        return X_train_scaled, X_val_scaled  # Return scaled versions for LR
    
    def evaluate_models(self, X_test, y_test):
        """
        Comprehensive model evaluation on test set
        """
        print("\n🧪 FINAL MODEL EVALUATION")
        print("=" * 50)
        
        evaluation_results = {}
        
        for model_name, model in self.models.items():
            print(f"\n📊 Evaluating {model_name}...")
            
            # Handle scaling for Logistic Regression
            if model_name == 'Logistic Regression':
                X_test_input = self.scalers[model_name].transform(X_test)
            else:
                X_test_input = X_test
            
            # Predictions
            y_pred = model.predict(X_test_input)
            y_pred_proba = model.predict_proba(X_test_input)[:, 1]
            
            # Comprehensive metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1_score': f1_score(y_test, y_pred),
                'roc_auc': roc_auc_score(y_test, y_pred_proba)
            }
            
            evaluation_results[model_name] = metrics
            
            # Print results
            print(f"   🎯 Accuracy: {metrics['accuracy']:.4f}")
            print(f"   📊 Precision: {metrics['precision']:.4f}")
            print(f"   📈 Recall: {metrics['recall']:.4f}")
            print(f"   ⚖️ F1-Score: {metrics['f1_score']:.4f}")
            print(f"   📈 ROC-AUC: {metrics['roc_auc']:.4f}")
        
        # Find best model
        best_model = max(evaluation_results, key=lambda x: evaluation_results[x]['f1_score'])
        print(f"\n🏆 BEST MODEL: {best_model}")
        print(f"   🎯 F1-Score: {evaluation_results[best_model]['f1_score']:.4f}")
        
        return evaluation_results, best_model
    
    def analyze_feature_importance(self, model_name='Random Forest'):
        """
        Analyze feature importance for tree-based models
        """
        print(f"\n🔍 FEATURE IMPORTANCE ANALYSIS - {model_name}")
        print("=" * 60)
        
        if model_name in self.models and hasattr(self.models[model_name], 'feature_importances_'):
            model = self.models[model_name]
            importances = model.feature_importances_
            
            # Create feature importance dataframe
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            print("🔝 TOP 10 MOST IMPORTANT FEATURES:")
            for i, row in feature_importance.head(10).iterrows():
                print(f"   {row['feature']:.<30} {row['importance']:.4f}")
            
            return feature_importance
        else:
            print(f"❌ Feature importance not available for {model_name}")
            return None
    
    def save_models(self, best_model_name):
        """
        Save trained models for production use
        """
        print(f"\n💾 SAVING MODELS")
        print("=" * 30)
        
        os.makedirs("models", exist_ok=True)
        
        # Save best model
        best_model = self.models[best_model_name]
        model_path = f"models/{best_model_name.lower().replace(' ', '_')}_model.pkl"
        joblib.dump(best_model, model_path)
        
        # Save scaler if needed
        if best_model_name in self.scalers:
            scaler_path = f"models/{best_model_name.lower().replace(' ', '_')}_scaler.pkl"
            joblib.dump(self.scalers[best_model_name], scaler_path)
            print(f"✅ Saved scaler: {scaler_path}")
        
        print(f"✅ Saved best model: {model_path}")
        
        return model_path

def main():
    print("🏥 CLINICAL TRIAL SAFETY ML PIPELINE")
    print("🚀 Professional Machine Learning Development")
    print("=" * 70)
    
    pipeline = ClinicalTrialMLPipeline()
    
    # Load data
    X, y = pipeline.load_processed_data()
    if X is None:
        return
    
    # Create splits
    X_train, X_val, X_test, y_train, y_val, y_test = pipeline.create_train_test_splits(X, y)
    
    # Train models
    X_train_scaled, X_val_scaled = pipeline.train_models(X_train, y_train, X_val, y_val)
    
    # Evaluate models
    results, best_model = pipeline.evaluate_models(X_test, y_test)
    
    # Feature importance
    feature_importance = pipeline.analyze_feature_importance()
    
    # Save models
    model_path = pipeline.save_models(best_model)
    
    print(f"\n🎉 ML PIPELINE COMPLETE!")
    print(f"🏆 Best Model: {best_model}")
    print(f"📈 Ready for production deployment!")

if __name__ == "__main__":
    main()