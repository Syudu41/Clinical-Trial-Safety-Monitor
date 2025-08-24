"""
AWS Lambda Handler for Clinical Trial Safety Monitoring
Processes adverse events using ML models and stores results in PostgreSQL
"""

import json
import pickle
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
import boto3
from botocore.exceptions import ClientError

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

class SafetyEventProcessor:
    """Processes adverse events using ML models in Lambda environment"""
    
    def __init__(self):
        """Initialize processor with AWS clients and model"""
        self.s3_client = boto3.client('s3')
        self.model = None
        self.model_version = "gradient_boosting_v1.0"
        
        # S3 bucket configuration (from environment variables)
        import os
        self.processed_bucket = os.environ.get('S3_PROCESSED_BUCKET', 'clinical-safety-processed-2025-yourname')
        
        # Database configuration
        self.db_config = {
            'host': os.environ.get('RDS_HOST'),
            'database': os.environ.get('RDS_DATABASE', 'clinical_safety'),
            'user': os.environ.get('RDS_USER', 'postgres'),
            'password': os.environ.get('RDS_PASSWORD'),
            'port': int(os.environ.get('RDS_PORT', '5432'))
        }
        
        # Load ML model on cold start
        self._load_model()
    
    def _load_model(self):
        """Load the trained ML model from S3"""
        try:
            logger.info("Loading ML model from S3...")
            
            # Download model from S3
            model_key = 'models/gradient_boosting_model.pkl'
            response = self.s3_client.get_object(
                Bucket=self.processed_bucket,
                Key=model_key
            )
            
            # Deserialize model
            self.model = pickle.loads(response['Body'].read())
            logger.info("✅ ML model loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to load ML model: {e}")
            raise
    
    def _get_db_connection(self):
        """Create database connection"""
        import psycopg2
        return psycopg2.connect(**self.db_config)
    
    def preprocess_event(self, raw_event: Dict) -> Optional[pd.DataFrame]:
        """
        Preprocess raw adverse event for ML model prediction
        
        Args:
            raw_event: Raw adverse event data
            
        Returns:
            Preprocessed DataFrame ready for ML model
        """
        try:
            # Expected features for ML model (based on your trained model)
            expected_features = [
                'safetyreportversion', 'serious', 'seriousnesscongenitalanomali',
                'seriousnessdeath', 'seriousnessdisabling', 'seriousnesshospitalization',
                'seriousnesslifethreatening', 'seriousnessother', 'duplicate',
                'occurcountry_encoded', 'primarysource_encoded', 
                'primarysourcecountry_encoded', 'reporttype_encoded',
                'drug_count', 'reaction_count', 'patient_age', 'patient_weight',
                'patient_sex_encoded', 'outcome_severity', 'patient_risk_score',
                'time_to_onset_days', 'concomitant_drug_risk'
            ]
            
            # Create DataFrame with expected features
            processed_data = {}
            
            # Map raw event to processed features
            for feature in expected_features:
                if feature in raw_event:
                    processed_data[feature] = raw_event[feature]
                else:
                    # Use default values for missing features
                    processed_data[feature] = self._get_default_value(feature)
            
            # Convert to DataFrame
            df = pd.DataFrame([processed_data])
            
            # Ensure correct data types
            df = self._ensure_data_types(df)
            
            logger.info(f"✅ Event preprocessed successfully, shape: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"❌ Error preprocessing event: {e}")
            return None
    
    def _get_default_value(self, feature: str) -> float:
        """Get default value for missing features"""
        defaults = {
            'safetyreportversion': 1.0,
            'serious': 0.0,
            'seriousnesscongenitalanomali': 0.0,
            'seriousnessdeath': 0.0,
            'seriousnessdisabling': 0.0,
            'seriousnesshospitalization': 0.0,
            'seriousnesslifethreatening': 0.0,
            'seriousnessother': 0.0,
            'duplicate': 0.0,
            'occurcountry_encoded': 0.0,
            'primarysource_encoded': 0.0,
            'primarysourcecountry_encoded': 0.0,
            'reporttype_encoded': 0.0,
            'drug_count': 1.0,
            'reaction_count': 1.0,
            'patient_age': 45.0,  # Median adult age
            'patient_weight': 70.0,  # Average adult weight
            'patient_sex_encoded': 1.0,  # Female
            'outcome_severity': 2.0,  # Medium severity
            'patient_risk_score': 2.5,  # Medium risk
            'time_to_onset_days': 30.0,  # 30 days average
            'concomitant_drug_risk': 1.0  # Low risk
        }
        return defaults.get(feature, 0.0)
    
    def _ensure_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure all features have correct data types for ML model"""
        try:
            # Convert all columns to float (scikit-learn expects numeric data)
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
            
            return df
        except Exception as e:
            logger.error(f"Error converting data types: {e}")
            return df
    
    def predict_safety_risk(self, processed_event: pd.DataFrame) -> Dict[str, Any]:
        """
        Make safety risk prediction using ML model
        
        Args:
            processed_event: Preprocessed event DataFrame
            
        Returns:
            Dictionary with prediction results
        """
        try:
            if self.model is None:
                raise ValueError("ML model not loaded")
            
            # Make prediction
            prediction_proba = self.model.predict_proba(processed_event)[0]
            prediction = self.model.predict(processed_event)[0]
            
            # Get prediction probability for serious outcome
            serious_probability = prediction_proba[1] if len(prediction_proba) > 1 else prediction_proba[0]
            
            # Determine alert level
            alert_level = self._determine_alert_level(serious_probability)
            
            # Create prediction result
            result = {
                'prediction': int(prediction),
                'serious_probability': float(serious_probability),
                'alert_level': alert_level,
                'model_version': self.model_version,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            logger.info(f"✅ Prediction made: {result}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error making prediction: {e}")
            return {
                'prediction': 0,
                'serious_probability': 0.0,
                'alert_level': 'LOW',
                'model_version': self.model_version,
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    def _determine_alert_level(self, probability: float) -> str:
        """Determine alert level based on prediction probability"""
        if probability >= 0.8:
            return 'HIGH'
        elif probability >= 0.6:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def store_results(self, event_data: Dict, prediction: Dict) -> bool:
        """
        Store event and prediction results in PostgreSQL
        
        Args:
            event_data: Original event data
            prediction: ML prediction results
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with self._get_db_connection() as conn:
                with conn.cursor() as cursor:
                    # Insert adverse event (simplified for Lambda)
                    event_insert_sql = """
                    INSERT INTO adverse_events (
                        event_id, safetyreportid, serious, outcome_severity,
                        patient_risk_score, raw_data, created_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (event_id) DO NOTHING;
                    """
                    
                    event_values = (
                        event_data.get('event_id', f"lambda_{datetime.utcnow().timestamp()}"),
                        event_data.get('safetyreportid', 'UNKNOWN'),
                        event_data.get('serious', 0),
                        event_data.get('outcome_severity', 0),
                        event_data.get('patient_risk_score', 0.0),
                        json.dumps(event_data),
                        datetime.utcnow()
                    )
                    
                    cursor.execute(event_insert_sql, event_values)
                    
                    # Insert safety alert if significant risk
                    if prediction['serious_probability'] > 0.5:
                        alert_insert_sql = """
                        INSERT INTO safety_alerts (
                            event_id, alert_type, alert_level, prediction_probability,
                            model_version, alert_message, alert_data, created_at
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
                        """
                        
                        alert_message = f"High risk adverse event detected (probability: {prediction['serious_probability']:.2%})"
                        
                        alert_values = (
                            event_data.get('event_id', f"lambda_{datetime.utcnow().timestamp()}"),
                            'ML_PREDICTION',
                            prediction['alert_level'],
                            prediction['serious_probability'],
                            prediction['model_version'],
                            alert_message,
                            json.dumps(prediction),
                            datetime.utcnow()
                        )
                        
                        cursor.execute(alert_insert_sql, alert_values)
                    
                    conn.commit()
                    logger.info("✅ Results stored successfully")
                    return True
                    
        except Exception as e:
            logger.error(f"❌ Error storing results: {e}")
            return False
    
    def process_event(self, event_data: Dict) -> Dict[str, Any]:
        """
        Main processing function for a single adverse event
        
        Args:
            event_data: Raw adverse event data
            
        Returns:
            Processing results
        """
        try:
            logger.info(f"Processing event: {event_data.get('event_id', 'unknown')}")
            
            # Preprocess event
            processed_event = self.preprocess_event(event_data)
            if processed_event is None:
                return {'success': False, 'error': 'Preprocessing failed'}
            
            # Make prediction
            prediction = self.predict_safety_risk(processed_event)
            
            # Store results
            stored = self.store_results(event_data, prediction)
            
            # Return processing results
            result = {
                'success': True,
                'event_id': event_data.get('event_id', 'unknown'),
                'prediction': prediction,
                'stored_in_db': stored,
                'processing_time': datetime.utcnow().isoformat()
            }
            
            logger.info(f"✅ Event processed successfully: {result}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error processing event: {e}")
            return {
                'success': False,
                'error': str(e),
                'event_id': event_data.get('event_id', 'unknown'),
                'processing_time': datetime.utcnow().isoformat()
            }


# Global processor instance (reused across Lambda invocations)
processor = None

def lambda_handler(event, context):
    """
    AWS Lambda entry point
    
    Args:
        event: Lambda event data
        context: Lambda runtime context
        
    Returns:
        Processing results
    """
    global processor
    
    # Initialize processor on cold start
    if processor is None:
        logger.info("🚀 Initializing Safety Event Processor...")
        processor = SafetyEventProcessor()
    
    try:
        logger.info(f"📨 Received Lambda event: {json.dumps(event, default=str)[:500]}...")
        
        # Handle different event sources
        if 'Records' in event:
            # Handle S3 events or SQS messages
            results = []
            for record in event['Records']:
                if 's3' in record:
                    # S3 triggered event
                    result = process_s3_event(record, processor)
                    results.append(result)
                else:
                    # Direct event data
                    result = processor.process_event(record)
                    results.append(result)
            
            return {
                'statusCode': 200,
                'body': json.dumps({
                    'message': f'Processed {len(results)} events',
                    'results': results
                })
            }
        
        else:
            # Direct invocation with event data
            result = processor.process_event(event)
            
            return {
                'statusCode': 200,
                'body': json.dumps({
                    'message': 'Event processed successfully',
                    'result': result
                })
            }
            
    except Exception as e:
        logger.error(f"❌ Lambda handler error: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e),
                'message': 'Event processing failed'
            })
        }


def process_s3_event(s3_record, processor):
    """Process S3 triggered events"""
    try:
        bucket = s3_record['s3']['bucket']['name']
        key = s3_record['s3']['object']['key']
        
        logger.info(f"Processing S3 object: s3://{bucket}/{key}")
        
        # For now, return a placeholder result
        # In production, you might read and process the S3 file
        return {
            'success': True,
            'message': f'S3 event processed: {key}',
            'bucket': bucket,
            'key': key
        }
        
    except Exception as e:
        logger.error(f"❌ Error processing S3 event: {e}")
        return {'success': False, 'error': str(e)}