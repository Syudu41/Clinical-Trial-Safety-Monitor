from kafka import KafkaConsumer, KafkaProducer
import json
import pandas as pd
import joblib
import sys
import os
from datetime import datetime
import numpy as np

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

class AdverseEventMLConsumer:
    def __init__(self, bootstrap_servers='localhost:9092'):
        """
        Initialize Kafka Consumer with ML model integration
        """
        print("🤖 Initializing ML-Powered Adverse Event Consumer...")
        
        # Kafka Consumer for incoming adverse events
        self.consumer = KafkaConsumer(
            'adverse-events',
            bootstrap_servers=bootstrap_servers,
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            key_deserializer=lambda m: m.decode('utf-8') if m else None,
            consumer_timeout_ms=30000,  # 30 second timeout
            auto_offset_reset='latest'  # Start from latest messages
        )
        
        # Kafka Producer for safety alerts
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda x: json.dumps(x).encode('utf-8'),
            key_serializer=lambda x: x.encode('utf-8') if x else None
        )
        
        # Load trained ML model
        self.ml_model = None
        self.feature_columns = None
        self.load_ml_model()
        
        # Statistics tracking
        self.stats = {
            'total_processed': 0,
            'serious_predicted': 0,
            'alerts_sent': 0,
            'start_time': datetime.now()
        }
        
        print("✅ Consumer initialized successfully!")
        
    def load_ml_model(self):
        """
        Load the trained Gradient Boosting model
        """
        model_path = "models/gradient_boosting_model.pkl"
        
        print(f"🧠 Loading trained ML model from: {model_path}")
        
        if not os.path.exists(model_path):
            print(f"❌ Model file not found: {model_path}")
            print("💡 Make sure you ran the ML training first!")
            return False
            
        try:
            self.ml_model = joblib.load(model_path)
            print("✅ ML model loaded successfully!")
            
            # Define expected feature columns (from your training)
            self.feature_columns = [
                'reporting_lag_days', 'report_month', 'report_year', 
                'weight_group', 'age_group', 'drug_count', 'reaction_count',
                'outcome_severity', 'patient_risk_score', 'drug_risk_score',
                'occurcountry_encoded', 'primarysourcecountry_encoded',
                'primary_drug_encoded', 'primary_reaction_encoded'
            ]
            
            print(f"🎯 Model expects {len(self.feature_columns)} features")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load ML model: {str(e)}")
            return False
    
    def preprocess_event_for_ml(self, event_data):
        """
        Convert streaming event to ML model input format
        """
        try:
            # Extract ML features from event
            ml_features = {}
            
            for feature in self.feature_columns:
                if feature in event_data:
                    ml_features[feature] = event_data[feature]
                else:
                    # Handle missing features with defaults
                    ml_features[feature] = 0
                    
            # Convert to DataFrame (model expects this format)
            feature_df = pd.DataFrame([ml_features])
            
            # Ensure correct data types
            for col in feature_df.columns:
                feature_df[col] = pd.to_numeric(feature_df[col], errors='coerce').fillna(0)
            
            return feature_df
            
        except Exception as e:
            print(f"❌ Preprocessing error: {str(e)}")
            return None
    
    def predict_event_severity(self, event_data):
        """
        Apply ML model to predict event severity
        """
        if self.ml_model is None:
            print("❌ ML model not loaded")
            return None
            
        try:
            # Preprocess event
            ml_input = self.preprocess_event_for_ml(event_data)
            if ml_input is None:
                return None
                
            # Make prediction
            prediction = self.ml_model.predict(ml_input)[0]
            prediction_proba = self.ml_model.predict_proba(ml_input)[0]
            
            # Get confidence scores
            confidence_non_serious = prediction_proba[0]
            confidence_serious = prediction_proba[1]
            
            result = {
                'prediction': int(prediction),
                'confidence_non_serious': float(confidence_non_serious),
                'confidence_serious': float(confidence_serious),
                'prediction_label': 'SERIOUS' if prediction == 1 else 'NON_SERIOUS',
                'high_confidence': max(prediction_proba) > 0.8
            }
            
            return result
            
        except Exception as e:
            print(f"❌ Prediction error: {str(e)}")
            return None
    
    def generate_safety_alert(self, event_data, prediction_result):
        """
        Generate safety alert for serious events - FIXED JSON SERIALIZATION
        """
        alert = {
            'alert_id': f"ALERT_{self.stats['alerts_sent'] + 1:06d}",
            'timestamp': datetime.now().isoformat(),
            'event_id': event_data.get('event_id', 'UNKNOWN'),
            'safetyreportid': str(event_data.get('safetyreportid', 'UNKNOWN')),  # Convert to string
            
            # Original event data
            'original_event': {
                'outcome_severity': float(event_data.get('outcome_severity', 0)),
                'drug_count': int(event_data.get('drug_count', 0)),
                'reaction_count': int(event_data.get('reaction_count', 0)),
                'patient_risk_score': float(event_data.get('patient_risk_score', 0)),
                'source': str(event_data.get('source', 'UNKNOWN'))
            },
            
            # ML prediction details - FIXED BOOLEAN SERIALIZATION
            'ml_prediction': {
                'predicted_serious': 1 if prediction_result['prediction'] == 1 else 0,  # Convert bool to int
                'confidence_serious': float(prediction_result['confidence_serious']),
                'high_confidence': 1 if prediction_result['high_confidence'] else 0,  # Convert bool to int  
                'model_version': 'gradient_boosting_v1.0'
            },
            
            # Alert details
            'alert_level': 'HIGH' if prediction_result['confidence_serious'] > 0.9 else 'MEDIUM',
            'requires_review': 1 if prediction_result['confidence_serious'] > 0.8 else 0,  # Convert bool to int
            'alert_message': self.create_alert_message(event_data, prediction_result)
        }
        
        return alert
    
    def create_alert_message(self, event_data, prediction_result):
        """
        Create human-readable alert message
        """
        confidence = prediction_result['confidence_serious'] * 100
        severity = event_data.get('outcome_severity', 0)
        
        message = f"SERIOUS ADVERSE EVENT DETECTED (Confidence: {confidence:.1f}%)"
        
        if severity >= 3:
            message += " - High outcome severity"
        if event_data.get('drug_count', 0) > 2:
            message += " - Multiple drugs involved"
        if event_data.get('patient_risk_score', 0) > 1.5:
            message += " - High-risk patient"
            
        return message
    
    def send_safety_alert(self, alert):
        """
        Send safety alert to Kafka safety-alerts topic
        """
        try:
            future = self.producer.send(
                'safety-alerts',
                value=alert,
                key=alert['alert_id']
            )
            
            # Don't wait for confirmation to maintain speed
            self.stats['alerts_sent'] += 1
            return True
            
        except Exception as e:
            print(f"❌ Failed to send alert: {str(e)}")
            return False
    
    def process_stream(self, max_events=None):
        """
        Main consumer loop - process adverse events in real-time
        """
        print(f"\n🚀 STARTING REAL-TIME ML PROCESSING")
        print("=" * 60)
        print("🎯 Consuming from: adverse-events")
        print("📤 Publishing alerts to: safety-alerts") 
        print("🤖 ML Model: Gradient Boosting (86.28% accuracy)")
        print("⏸️  Press Ctrl+C to stop...")
        print("=" * 60)
        
        try:
            for message in self.consumer:
                event_data = message.value
                self.stats['total_processed'] += 1
                
                print(f"\n📨 Processing Event #{self.stats['total_processed']}")
                print(f"   🆔 Event ID: {event_data.get('event_id', 'N/A')}")
                print(f"   📋 Report ID: {event_data.get('safetyreportid', 'N/A')}")
                print(f"   ⏰ Timestamp: {event_data.get('timestamp', 'N/A')}")
                
                # Apply ML model
                prediction_result = self.predict_event_severity(event_data)
                
                if prediction_result:
                    predicted_serious = prediction_result['prediction'] == 1
                    confidence = prediction_result['confidence_serious'] * 100
                    
                    print(f"   🤖 ML Prediction: {prediction_result['prediction_label']}")
                    print(f"   📊 Confidence: {confidence:.1f}%")
                    
                    # Generate alert for serious events
                    if predicted_serious:
                        self.stats['serious_predicted'] += 1
                        
                        alert = self.generate_safety_alert(event_data, prediction_result)
                        alert_sent = self.send_safety_alert(alert)
                        
                        if alert_sent:
                            print(f"   🚨 SAFETY ALERT SENT - {alert['alert_level']} priority")
                            print(f"   💬 {alert['alert_message']}")
                        else:
                            print(f"   ❌ Failed to send alert")
                    else:
                        print(f"   ✅ Non-serious event - no alert needed")
                else:
                    print(f"   ❌ ML prediction failed")
                
                # Print running statistics every 10 events
                if self.stats['total_processed'] % 10 == 0:
                    self.print_statistics()
                
                # Stop if max events reached
                if max_events and self.stats['total_processed'] >= max_events:
                    print(f"\n🎯 Reached max events limit: {max_events}")
                    break
                    
        except KeyboardInterrupt:
            print("\n⏸️  Consumer stopped by user")
        except Exception as e:
            print(f"\n❌ Consumer error: {str(e)}")
        finally:
            self.print_final_statistics()
            self.close()
    
    def print_statistics(self):
        """
        Print current processing statistics
        """
        elapsed = (datetime.now() - self.stats['start_time']).total_seconds()
        rate = self.stats['total_processed'] / (elapsed / 60) if elapsed > 0 else 0
        serious_pct = (self.stats['serious_predicted'] / self.stats['total_processed'] * 100) if self.stats['total_processed'] > 0 else 0
        
        print(f"\n📊 PROCESSING STATISTICS:")
        print(f"   📈 Total processed: {self.stats['total_processed']}")
        print(f"   🚨 Serious predicted: {self.stats['serious_predicted']} ({serious_pct:.1f}%)")  
        print(f"   📤 Alerts sent: {self.stats['alerts_sent']}")
        print(f"   ⚡ Rate: {rate:.1f} events/minute")
    
    def print_final_statistics(self):
        """
        Print final session statistics
        """
        elapsed = (datetime.now() - self.stats['start_time']).total_seconds()
        
        print(f"\n🎉 FINAL PROCESSING STATISTICS")
        print("=" * 50)
        print(f"📈 Total Events Processed: {self.stats['total_processed']}")
        print(f"🚨 Serious Events Detected: {self.stats['serious_predicted']}")
        print(f"📤 Safety Alerts Sent: {self.stats['alerts_sent']}")
        print(f"⏱️  Session Duration: {elapsed:.1f} seconds")
        print(f"⚡ Average Rate: {self.stats['total_processed'] / (elapsed / 60):.1f} events/minute")
        
        if self.stats['total_processed'] > 0:
            serious_rate = self.stats['serious_predicted'] / self.stats['total_processed'] * 100
            print(f"📊 Serious Event Rate: {serious_rate:.1f}%")
        
    def close(self):
        """
        Close consumer and producer
        """
        print("\n🔒 Closing consumer and producer...")
        self.consumer.close()
        self.producer.close()
        print("✅ Closed successfully")

def main():
    print("🏥 CLINICAL TRIAL SAFETY ML CONSUMER")
    print("🤖 Real-time Adverse Event Classification")
    print("=" * 70)
    
    # Initialize consumer
    consumer = AdverseEventMLConsumer()
    
    if consumer.ml_model is None:
        print("❌ Cannot start without ML model. Exiting.")
        return
    
    print("🚀 Consumer ready! Run the producer in another terminal to send events.")
    print("💡 Command: python src/streaming/kafka_producer.py")
    
    # Start processing stream
    consumer.process_stream(max_events=100)  # Process up to 100 events

if __name__ == "__main__":
    main()