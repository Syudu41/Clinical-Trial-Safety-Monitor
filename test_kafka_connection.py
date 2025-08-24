from kafka import KafkaProducer, KafkaConsumer
import json
import time

def test_kafka():
    print("🧪 TESTING KAFKA CONNECTION")
    print("=" * 40)
    
    try:
        # Test Producer
        print("📤 Testing Kafka Producer...")
        producer = KafkaProducer(
            bootstrap_servers='localhost:9092',
            value_serializer=lambda x: json.dumps(x).encode('utf-8')
        )
        
        # Send test message
        test_message = {'test': 'Hello Kafka!', 'timestamp': time.time()}
        future = producer.send('test-topic', test_message)
        result = future.get(timeout=10)
        
        print(f"✅ Producer working! Sent to partition {result.partition}, offset {result.offset}")
        producer.close()
        
        print("🎉 KAFKA CONNECTION SUCCESSFUL!")
        return True
            
    except Exception as e:
        print(f"❌ Kafka connection failed: {str(e)}")
        return False

if __name__ == "__main__":
    test_kafka()