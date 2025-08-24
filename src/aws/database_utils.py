"""
PostgreSQL Database Utilities for Clinical Trial Safety Monitoring System
Handles database connections, table creation, and data operations
"""

import psycopg2
import pandas as pd
import logging
import json
from datetime import datetime
from typing import Optional, List, Dict, Tuple
from psycopg2.extras import RealDictCursor, execute_values
from contextlib import contextmanager

# Import project configuration
from config import aws_config

class DatabaseManager:
    """Manages PostgreSQL database connections and operations"""
    
    def __init__(self, host: str = None, database: str = None, user: str = None, password: str = None, port: int = 5432):
        """
        Initialize database connection parameters
        
        Args:
            host: Database host (will use RDS endpoint)
            database: Database name
            user: Database user
            password: Database password
            port: Database port (default 5432)
        """
        self.host = host or aws_config.RDS_HOST
        self.database = database or aws_config.RDS_DATABASE
        self.user = user or aws_config.RDS_USER
        self.password = password or aws_config.RDS_PASSWORD
        self.port = port
        
        # Validate required parameters
        if not all([self.host, self.database, self.user, self.password]):
            missing = [name for name, value in [
                ('host', self.host), ('database', self.database), 
                ('user', self.user), ('password', self.password)
            ] if not value]
            raise ValueError(f"Missing required database parameters: {missing}")
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = None
        try:
            conn = psycopg2.connect(
                host=self.host,
                database=self.database,
                user=self.user,
                password=self.password,
                port=self.port
            )
            yield conn
        except psycopg2.Error as e:
            if conn:
                conn.rollback()
            logging.error(f"❌ Database connection error: {e}")
            raise
        finally:
            if conn:
                conn.close()
    
    def test_connection(self) -> bool:
        """Test database connection and return True if successful"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT version();")
                    version = cursor.fetchone()[0]
                    logging.info(f"✅ Connected to PostgreSQL: {version}")
                    return True
        except Exception as e:
            logging.error(f"❌ Connection test failed: {e}")
            return False
    
    def create_tables(self) -> bool:
        """Create all necessary tables for the clinical safety system"""
        
        # SQL statements for table creation
        create_tables_sql = [
            # Adverse Events table - stores all adverse event records
            """
            CREATE TABLE IF NOT EXISTS adverse_events (
                id SERIAL PRIMARY KEY,
                event_id VARCHAR(100) UNIQUE,
                safetyreportid VARCHAR(100),
                safetyreportversion VARCHAR(20),
                receivedate DATE,
                receiptdate DATE,
                serious INTEGER,
                seriousnesscongenitalanomali INTEGER,
                seriousnessdeath INTEGER,
                seriousnessdisabling INTEGER,
                seriousnesshospitalization INTEGER,
                seriousnesslifethreatening INTEGER,
                seriousnessother INTEGER,
                transmissiondate DATE,
                duplicate INTEGER,
                companynumb VARCHAR(100),
                occurcountry_encoded INTEGER,
                primarysource_encoded INTEGER,
                primarysourcecountry_encoded INTEGER,
                reporttype_encoded INTEGER,
                drug_count INTEGER,
                reaction_count INTEGER,
                patient_age FLOAT,
                patient_weight FLOAT,
                patient_sex_encoded INTEGER,
                outcome_severity INTEGER,
                patient_risk_score FLOAT,
                time_to_onset_days FLOAT,
                concomitant_drug_risk FLOAT,
                raw_data JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """,
            
            # Safety Alerts table - stores generated alerts
            """
            CREATE TABLE IF NOT EXISTS safety_alerts (
                id SERIAL PRIMARY KEY,
                event_id VARCHAR(100) REFERENCES adverse_events(event_id),
                alert_type VARCHAR(50) NOT NULL,
                alert_level VARCHAR(20) NOT NULL, -- 'HIGH', 'MEDIUM', 'LOW'
                prediction_probability FLOAT,
                model_version VARCHAR(50),
                alert_message TEXT,
                alert_data JSONB,
                is_reviewed BOOLEAN DEFAULT FALSE,
                reviewed_by VARCHAR(100),
                reviewed_at TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """,
            
            # Model Performance table - tracks ML model performance
            """
            CREATE TABLE IF NOT EXISTS model_performance (
                id SERIAL PRIMARY KEY,
                model_name VARCHAR(100),
                model_version VARCHAR(50),
                accuracy FLOAT,
                precision_score FLOAT,
                recall FLOAT,
                f1_score FLOAT,
                roc_auc FLOAT,
                training_date TIMESTAMP,
                feature_importance JSONB,
                performance_data JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """,
            
            # System Metrics table - tracks system performance
            """
            CREATE TABLE IF NOT EXISTS system_metrics (
                id SERIAL PRIMARY KEY,
                metric_name VARCHAR(100),
                metric_value FLOAT,
                metric_unit VARCHAR(50),
                component VARCHAR(100), -- 'kafka', 'lambda', 's3', 'rds'
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                additional_data JSONB
            );
            """
        ]
        
        # Create indexes for better performance
        create_indexes_sql = [
            "CREATE INDEX IF NOT EXISTS idx_adverse_events_receivedate ON adverse_events(receivedate);",
            "CREATE INDEX IF NOT EXISTS idx_adverse_events_serious ON adverse_events(serious);",
            "CREATE INDEX IF NOT EXISTS idx_adverse_events_outcome_severity ON adverse_events(outcome_severity);",
            "CREATE INDEX IF NOT EXISTS idx_safety_alerts_event_id ON safety_alerts(event_id);",
            "CREATE INDEX IF NOT EXISTS idx_safety_alerts_alert_level ON safety_alerts(alert_level);",
            "CREATE INDEX IF NOT EXISTS idx_safety_alerts_created_at ON safety_alerts(created_at);",
            "CREATE INDEX IF NOT EXISTS idx_system_metrics_component ON system_metrics(component);",
            "CREATE INDEX IF NOT EXISTS idx_system_metrics_timestamp ON system_metrics(timestamp);"
        ]
        
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    # Create tables
                    for sql in create_tables_sql:
                        cursor.execute(sql)
                        logging.info("✅ Table created successfully")
                    
                    # Create indexes
                    for sql in create_indexes_sql:
                        cursor.execute(sql)
                        logging.info("✅ Index created successfully")
                    
                    conn.commit()
                    logging.info("✅ All database tables and indexes created successfully")
                    return True
                    
        except Exception as e:
            logging.error(f"❌ Error creating tables: {e}")
            return False
    
    def insert_adverse_event(self, event_data: Dict) -> Optional[int]:
        """
        Insert a single adverse event record
        
        Args:
            event_data: Dictionary containing adverse event data
            
        Returns:
            int: ID of inserted record, or None if failed
        """
        insert_sql = """
        INSERT INTO adverse_events (
            event_id, safetyreportid, safetyreportversion, receivedate, receiptdate,
            serious, seriousnesscongenitalanomali, seriousnessdeath, seriousnessdisabling,
            seriousnesshospitalization, seriousnesslifethreatening, seriousnessother,
            transmissiondate, duplicate, companynumb, occurcountry_encoded,
            primarysource_encoded, primarysourcecountry_encoded, reporttype_encoded,
            drug_count, reaction_count, patient_age, patient_weight, patient_sex_encoded,
            outcome_severity, patient_risk_score, time_to_onset_days, concomitant_drug_risk,
            raw_data
        ) VALUES (
            %(event_id)s, %(safetyreportid)s, %(safetyreportversion)s, %(receivedate)s, %(receiptdate)s,
            %(serious)s, %(seriousnesscongenitalanomali)s, %(seriousnessdeath)s, %(seriousnessdisabling)s,
            %(seriousnesshospitalization)s, %(seriousnesslifethreatening)s, %(seriousnessother)s,
            %(transmissiondate)s, %(duplicate)s, %(companynumb)s, %(occurcountry_encoded)s,
            %(primarysource_encoded)s, %(primarysourcecountry_encoded)s, %(reporttype_encoded)s,
            %(drug_count)s, %(reaction_count)s, %(patient_age)s, %(patient_weight)s, %(patient_sex_encoded)s,
            %(outcome_severity)s, %(patient_risk_score)s, %(time_to_onset_days)s, %(concomitant_drug_risk)s,
            %(raw_data)s
        ) RETURNING id;
        """
        
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(insert_sql, event_data)
                    record_id = cursor.fetchone()[0]
                    conn.commit()
                    logging.info(f"✅ Inserted adverse event: {record_id}")
                    return record_id
        except Exception as e:
            logging.error(f"❌ Error inserting adverse event: {e}")
            return None
    
    def insert_safety_alert(self, alert_data: Dict) -> Optional[int]:
        """
        Insert a safety alert record
        
        Args:
            alert_data: Dictionary containing alert data
            
        Returns:
            int: ID of inserted record, or None if failed
        """
        insert_sql = """
        INSERT INTO safety_alerts (
            event_id, alert_type, alert_level, prediction_probability,
            model_version, alert_message, alert_data
        ) VALUES (
            %(event_id)s, %(alert_type)s, %(alert_level)s, %(prediction_probability)s,
            %(model_version)s, %(alert_message)s, %(alert_data)s
        ) RETURNING id;
        """
        
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(insert_sql, alert_data)
                    record_id = cursor.fetchone()[0]
                    conn.commit()
                    logging.info(f"✅ Inserted safety alert: {record_id}")
                    return record_id
        except Exception as e:
            logging.error(f"❌ Error inserting safety alert: {e}")
            return None
    
    def bulk_insert_events(self, events_df: pd.DataFrame) -> bool:
        """
        Bulk insert multiple adverse events from DataFrame
        
        Args:
            events_df: DataFrame with adverse event data
            
        Returns:
            bool: True if successful
        """
        if events_df.empty:
            logging.warning("⚠️ Empty DataFrame provided for bulk insert")
            return False
        
        # Prepare data for bulk insert
        columns = events_df.columns.tolist()
        values = [tuple(row) for row in events_df.values]
        
        # Create placeholders for the INSERT statement
        placeholders = ', '.join(['%s'] * len(columns))
        column_names = ', '.join(columns)
        
        insert_sql = f"""
        INSERT INTO adverse_events ({column_names})
        VALUES ({placeholders})
        ON CONFLICT (event_id) DO NOTHING;
        """
        
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    execute_values(
                        cursor, insert_sql, values, template=None, page_size=1000
                    )
                    conn.commit()
                    logging.info(f"✅ Bulk inserted {len(events_df)} adverse events")
                    return True
        except Exception as e:
            logging.error(f"❌ Error in bulk insert: {e}")
            return False
    
    def get_recent_alerts(self, hours: int = 24, alert_level: str = None) -> List[Dict]:
        """
        Get recent safety alerts
        
        Args:
            hours: Number of hours to look back
            alert_level: Filter by alert level ('HIGH', 'MEDIUM', 'LOW')
            
        Returns:
            List of alert dictionaries
        """
        sql = """
        SELECT sa.*, ae.safetyreportid, ae.receivedate
        FROM safety_alerts sa
        JOIN adverse_events ae ON sa.event_id = ae.event_id
        WHERE sa.created_at >= NOW() - INTERVAL '%s hours'
        """
        
        params = [hours]
        
        if alert_level:
            sql += " AND sa.alert_level = %s"
            params.append(alert_level)
        
        sql += " ORDER BY sa.created_at DESC;"
        
        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    cursor.execute(sql, params)
                    alerts = [dict(row) for row in cursor.fetchall()]
                    logging.info(f"✅ Retrieved {len(alerts)} recent alerts")
                    return alerts
        except Exception as e:
            logging.error(f"❌ Error retrieving recent alerts: {e}")
            return []
    
    def get_system_stats(self) -> Dict:
        """
        Get system statistics and metrics
        
        Returns:
            Dictionary with system statistics
        """
        stats_sql = [
            ("total_events", "SELECT COUNT(*) FROM adverse_events;"),
            ("serious_events", "SELECT COUNT(*) FROM adverse_events WHERE serious = 1;"),
            ("total_alerts", "SELECT COUNT(*) FROM safety_alerts;"),
            ("high_alerts_24h", """
                SELECT COUNT(*) FROM safety_alerts 
                WHERE alert_level = 'HIGH' AND created_at >= NOW() - INTERVAL '24 hours';
            """),
            ("events_last_24h", """
                SELECT COUNT(*) FROM adverse_events 
                WHERE created_at >= NOW() - INTERVAL '24 hours';
            """)
        ]
        
        stats = {}
        
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    for stat_name, sql in stats_sql:
                        cursor.execute(sql)
                        stats[stat_name] = cursor.fetchone()[0]
            
            logging.info("✅ Retrieved system statistics")
            return stats
            
        except Exception as e:
            logging.error(f"❌ Error retrieving system stats: {e}")
            return {}
    
    def record_model_performance(self, model_data: Dict) -> bool:
        """
        Record ML model performance metrics
        
        Args:
            model_data: Dictionary with model performance data
            
        Returns:
            bool: True if successful
        """
        insert_sql = """
        INSERT INTO model_performance (
            model_name, model_version, accuracy, precision_score, recall,
            f1_score, roc_auc, training_date, feature_importance, performance_data
        ) VALUES (
            %(model_name)s, %(model_version)s, %(accuracy)s, %(precision_score)s, %(recall)s,
            %(f1_score)s, %(roc_auc)s, %(training_date)s, %(feature_importance)s, %(performance_data)s
        );
        """
        
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(insert_sql, model_data)
                    conn.commit()
                    logging.info("✅ Recorded model performance")
                    return True
        except Exception as e:
            logging.error(f"❌ Error recording model performance: {e}")
            return False


# Testing and utility functions
def test_database_connection(host: str, database: str, user: str, password: str, port: int = 5432):
    """Test database connection with provided credentials"""
    print("🧪 Testing PostgreSQL Connection...")
    
    try:
        db_manager = DatabaseManager(host, database, user, password, port)
        
        if db_manager.test_connection():
            print("✅ Database connection successful!")
            
            # Test table creation
            print("🏗️ Creating database tables...")
            if db_manager.create_tables():
                print("✅ Database tables created successfully!")
                
                # Get system stats
                stats = db_manager.get_system_stats()
                print(f"📊 System Stats: {stats}")
                
                return True
            else:
                print("❌ Table creation failed")
                return False
        else:
            print("❌ Database connection failed")
            return False
            
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        return False


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("Database utilities module loaded.")
    print("Use test_database_connection(host, database, user, password) to test your connection.")