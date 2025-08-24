from kafka import KafkaProducer
import json
import pandas as pd
import time
import sys
import os
from datetime import datetime
import random

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

class AdverseEventProducer:
    def __init__(self, bootstrap_servers='localhost:9092'):
        """
        Initialize Kafka Producer for adverse events
        """
        print("🚀 Initializing Adverse Event Producer...")
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda x: json.dumps(x).encode('utf-8'),
            key_serializer=lambda x: x.encode('utf-8') if x else None,
            acks='all',  # Ensure message delivery
            retries=3
        )
        self.topics = {
            'adverse_events': 'adverse-events',
            'safety_alerts': 'safety-alerts'  
        }
        print("✅ Producer initialized successfully!")
        
    def load_sample_data(self, filepath="data/processed/processed_fda_25k.csv"):
        """
        Load processed 25K dataset to simulate real-time events
        """
        print(f"📂 Loading 25K processed dataset...")
        
        if not os.path.exists(filepath):
            print(f"❌ File not found: {filepath}")
            return None
            
        df = pd.read_csv(filepath)
        print(f"✅ Loaded {len(df):,} adverse events for streaming")
        print(f"🎯 Target balance: {df['is_serious'].value_counts().to_dict()}")
        return df
        
    def simulate_real_time_stream(self, df, events_per_minute=30, duration_minutes=5):
        """
        Simulate real-time adverse event stream using your 25K dataset
        """
        print(f"\n🚀 STARTING REAL-TIME ADVERSE EVENT STREAM")
        print("=" * 60)
        print(f"📊 Rate: {events_per_minute} events/minute")
        print(f"⏱️  Duration: {duration_minutes} minutes") 
        print(f"📈 Total events: {events_per_minute * duration_minutes}")
        print(f"🗂️  Source: {len(df)} real FDA adverse events")
        print("=" * 60)
        
        total_events = 0
        serious_events = 0
        start_time = datetime.now()
        
        for minute in range(duration_minutes):
            minute_start = datetime.now()
            print(f"\n⏰ MINUTE {minute + 1}/{duration_minutes}")
            
            # Send events for this minute
            for event_num in range(events_per_minute):
                # Pick random event from your real dataset
                event_data = df.sample(1).iloc[0].to_dict()
                
                # Add real-time metadata
                event_data['timestamp'] = datetime.now().isoformat()
                event_data['event_id'] = f"AE_{total_events + 1:06d}"
                event_data['source'] = 'FDA_FAERS_STREAM'
                event_data['stream_batch'] = minute + 1
                
                # Track serious events
                if event_data.get('is_serious', 0) == 1:
                    serious_events += 1
                
                # Create message key for partitioning  
                key = f"safety_report_{event_data.get('safetyreportid', total_events)}"
                
                # Send to Kafka
                try:
                    future = self.producer.send(
                        self.topics['adverse_events'],
                        value=event_data,
                        key=key
                    )
                    # Don't wait for confirmation to maintain speed
                    total_events += 1
                    
                except Exception as e:
                    print(f"❌ Failed to send event {total_events}: {str(e)}")
                    continue
                
                # Rate limiting - spread events evenly across the minute
                time.sleep(60 / events_per_minute)
            
            # Flush every minute to ensure delivery
            self.producer.flush()
            
            # Minute summary
            minute_time = (datetime.now() - minute_start).total_seconds()
            elapsed_total = (datetime.now() - start_time).total_seconds()
            
            print(f"  ✅ Sent {events_per_minute} events in {minute_time:.1f}s")
            print(f"  📊 Total: {total_events} events | Serious: {serious_events} ({serious_events/total_events*100:.1f}%)")
            print(f"  ⚡ Rate: {total_events / (elapsed_total / 60):.1f} events/min")
        
        # Final summary
        final_time = (datetime.now() - start_time).total_seconds()
        print(f"\n🎉 STREAMING COMPLETE!")
        print("=" * 50)
        print(f"📈 Total events sent: {total_events}")
        print(f"⚠️  Serious events: {serious_events} ({serious_events/total_events*100:.1f}%)")
        print(f"⏱️  Total time: {final_time:.1f} seconds")
        print(f"📊 Average rate: {total_events / (final_time / 60):.1f} events/minute")
        print(f"🎯 Ready for consumer processing!")
        
        return {
            'total_events': total_events,
            'serious_events': serious_events,
            'duration_seconds': final_time,
            'rate_per_minute': total_events / (final_time / 60)
        }
        
    def send_test_event(self):
        """
        Send a single test event (for debugging)
        """
        print("🧪 Sending test adverse event...")
        
        test_event = {
            'event_id': 'TEST_001',
            'safetyreportid': 'TEST123456',
            'is_serious': 1,
            'outcome_severity': 3,
            'drug_count': 2,
            'reaction_count': 2,
            'patient_risk_score': 1.5,
            'drug_risk_score': 1.2,
            'weight_group': 2,
            'age_group': 3,
            'occurcountry_encoded': 1,
            'primarysourcecountry_encoded': 1,
            'primary_drug_encoded': 5,
            'primary_reaction_encoded': 8,
            'report_month': 8,
            'report_year': 2025,
            'reporting_lag_days': 15,
            'timestamp': datetime.now().isoformat(),
            'source': 'TEST_PRODUCER'
        }
        
        try:
            future = self.producer.send(
                self.topics['adverse_events'],
                value=test_event,
                key='test_event'
            )
            
            result = future.get(timeout=10)
            print(f"✅ Test event sent successfully!")
            print(f"   📍 Topic: {result.topic}")
            print(f"   📊 Partition: {result.partition}")
            print(f"   📈 Offset: {result.offset}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to send test event: {str(e)}")
            return False
        
    def close(self):
        """
        Close the producer cleanly
        """
        print("📤 Closing Kafka producer...")
        self.producer.flush()  # Ensure all messages are sent
        self.producer.close()
        print("✅ Producer closed successfully")

def main():
    print("🏥 CLINICAL TRIAL ADVERSE EVENT PRODUCER")
    print("🚀 Real-time Safety Monitoring Stream")
    print("=" * 70)
    
    # Initialize producer
    producer = AdverseEventProducer()
    
    # Test single event first
    print("\n🧪 TESTING SINGLE EVENT...")
    test_success = producer.send_test_event()
    
    if test_success:
        print("✅ Single event test passed!")
        
        # Load your 25K processed dataset
        df = producer.load_sample_data()
        if df is not None:
            print("\n🚀 Starting real-time adverse event simulation...")
            
            # Stream adverse events (adjust rate as needed)
            results = producer.simulate_real_time_stream(
                df, 
                events_per_minute=20,  # Moderate rate for testing
                duration_minutes=3     # 3 minutes = 60 total events
            )
            
            print(f"\n📊 STREAMING STATISTICS:")
            print(f"   Events: {results['total_events']}")
            print(f"   Serious: {results['serious_events']}")
            print(f"   Rate: {results['rate_per_minute']:.1f}/min")
            
        else:
            print("❌ Could not load 25K dataset")
    else:
        print("❌ Test event failed - check Kafka connection")
    
    # Cleanup
    producer.close()
    print("\n🎯 Producer session complete. Ready for consumer!")

if __name__ == "__main__":
    main()