"""
AWS Lambda Handler - Rule-Based Risk Assessment
Professional adverse event processing without ML dependencies
"""

import json
import logging
import os
from datetime import datetime
from typing import Dict, Any
import boto3

logger = logging.getLogger()
logger.setLevel(logging.INFO)

class RuleBasedSafetyProcessor:
    def __init__(self):
        self.db_config = {
            'host': os.environ.get('RDS_HOST'),
            'database': os.environ.get('RDS_DATABASE', 'clinical_safety'),
            'user': os.environ.get('RDS_USER', 'postgres'),
            'password': os.environ.get('RDS_PASSWORD'),
            'port': int(os.environ.get('RDS_PORT', '5432'))
        }
    
    def _get_db_connection(self):
        import psycopg2
        return psycopg2.connect(**self.db_config)
    
    def assess_risk(self, event_data: Dict) -> Dict[str, Any]:
        try:
            risk_score = 0.0
            risk_factors = []
            
            # Critical outcomes (highest weight)
            if event_data.get('seriousnessdeath', 0) == 1:
                risk_score += 1.0
                risk_factors.append('death')
            
            if event_data.get('seriousnesslifethreatening', 0) == 1:
                risk_score += 0.9
                risk_factors.append('life_threatening')
            
            # Severe outcomes
            if event_data.get('seriousnesshospitalization', 0) == 1:
                risk_score += 0.7
                risk_factors.append('hospitalization')
            
            if event_data.get('seriousnessdisabling', 0) == 1:
                risk_score += 0.6
                risk_factors.append('disabling')
            
            if event_data.get('seriousnesscongenitalanomali', 0) == 1:
                risk_score += 0.5
                risk_factors.append('birth_defect')
            
            # Patient demographics
            patient_age = event_data.get('patient_age', 0)
            if patient_age > 65:
                risk_score += 0.2
                risk_factors.append('elderly')
            elif patient_age < 18 and patient_age > 0:
                risk_score += 0.25
                risk_factors.append('pediatric')
            
            # Drug complexity
            drug_count = event_data.get('drug_count', 1)
            if drug_count > 5:
                risk_score += 0.3
                risk_factors.append('multiple_drugs')
            elif drug_count > 3:
                risk_score += 0.15
                risk_factors.append('several_drugs')
            
            # Reaction severity
            reaction_count = event_data.get('reaction_count', 1)
            if reaction_count > 3:
                risk_score += 0.2
                risk_factors.append('multiple_reactions')
            
            # Outcome severity score
            outcome_severity = event_data.get('outcome_severity', 0)
            if outcome_severity > 7:
                risk_score += 0.3
                risk_factors.append('high_severity_score')
            
            # Normalize to 0-1 range
            risk_score = min(risk_score, 1.0)
            
            # Determine alert level
            if risk_score >= 0.8:
                alert_level = 'HIGH'
            elif risk_score >= 0.5:
                alert_level = 'MEDIUM'
            else:
                alert_level = 'LOW'
            
            prediction = 1 if risk_score >= 0.5 else 0
            
            return {
                'prediction': prediction,
                'risk_score': risk_score,
                'alert_level': alert_level,
                'risk_factors': risk_factors,
                'model_version': 'rule_based_v1.0',
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Risk assessment error: {e}")
            return {
                'prediction': 0,
                'risk_score': 0.0,
                'alert_level': 'LOW',
                'risk_factors': [],
                'model_version': 'rule_based_v1.0',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    def store_results(self, event_data: Dict, assessment: Dict) -> bool:
        try:
            with self._get_db_connection() as conn:
                with conn.cursor() as cursor:
                    event_insert_sql = """
                    INSERT INTO adverse_events (
                        event_id, safetyreportid, serious, outcome_severity,
                        patient_risk_score, raw_data, created_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (event_id) DO NOTHING;
                    """
                    
                    cursor.execute(event_insert_sql, (
                        event_data.get('event_id', f"rule_{int(datetime.utcnow().timestamp())}"),
                        event_data.get('safetyreportid', 'UNKNOWN'),
                        assessment['prediction'],
                        int(assessment['risk_score'] * 10),
                        assessment['risk_score'],
                        json.dumps(event_data),
                        datetime.utcnow()
                    ))
                    
                    if assessment['risk_score'] >= 0.5:
                        alert_sql = """
                        INSERT INTO safety_alerts (
                            event_id, alert_type, alert_level, prediction_probability,
                            model_version, alert_message, alert_data, created_at
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
                        """
                        
                        alert_message = f"Rule-based alert: {assessment['alert_level']} risk - {', '.join(assessment['risk_factors'])}"
                        
                        cursor.execute(alert_sql, (
                            event_data.get('event_id', f"rule_{int(datetime.utcnow().timestamp())}"),
                            'RULE_BASED',
                            assessment['alert_level'],
                            assessment['risk_score'],
                            assessment['model_version'],
                            alert_message,
                            json.dumps(assessment),
                            datetime.utcnow()
                        ))
                    
                    conn.commit()
                    return True
        except Exception as e:
            logger.error(f"Database error: {e}")
            return False
    
    def process_event(self, event_data: Dict) -> Dict[str, Any]:
        try:
            assessment = self.assess_risk(event_data)
            stored = self.store_results(event_data, assessment)
            
            return {
                'success': True,
                'event_id': event_data.get('event_id', f"rule_{int(datetime.utcnow().timestamp())}"),
                'assessment': assessment,
                'stored_in_db': stored,
                'processing_time': datetime.utcnow().isoformat()
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'event_id': event_data.get('event_id', 'unknown'),
                'processing_time': datetime.utcnow().isoformat()
            }

processor = None

def lambda_handler(event, context):
    global processor
    
    try:
        if processor is None:
            processor = RuleBasedSafetyProcessor()
        
        result = processor.process_event(event)
        
        return {
            'statusCode': 200,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({
                'success': True,
                'message': 'Event processed with rule-based assessment',
                'result': result,
                'timestamp': datetime.utcnow().isoformat()
            })
        }
    except Exception as e:
        return {
            'statusCode': 500,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({
                'success': False,
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            })
        }