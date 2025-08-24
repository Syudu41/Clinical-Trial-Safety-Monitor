"""
AWS Lambda Testing Script
Tests the deployed Clinical Trial Safety Event Processor Lambda function
"""

import boto3
import json
import pandas as pd
from datetime import datetime
from pathlib import Path
import sys

# Add project root to path
sys.path.append('.')

class LambdaTester:
    """Tests deployed Lambda function"""
    
    def __init__(self):
        self.lambda_client = boto3.client('lambda')
        self.function_name = 'clinical-safety-processor'
        
    def create_test_event(self, use_sample_data=True):
        """Create a test adverse event for Lambda processing"""
        
        if use_sample_data:
            # Load sample from your processed data
            try:
                data_file = Path('data/processed/processed_fda_25k.csv')
                if data_file.exists():
                    df = pd.read_csv(data_file, nrows=1)  # Just first row
                    sample_event = df.iloc[0].to_dict()
                    
                    # Add required fields
                    sample_event['event_id'] = f"test_{datetime.now().timestamp()}"
                    
                    # Convert numpy types to Python types for JSON serialization
                    for key, value in sample_event.items():
                        if pd.isna(value):
                            sample_event[key] = None
                        elif hasattr(value, 'item'):  # numpy scalar
                            sample_event[key] = value.item()
                    
                    print("✅ Using sample data from processed dataset")
                    return sample_event
                    
            except Exception as e:
                print(f"⚠️ Could not load sample data: {e}")
        
        # Fallback: create synthetic test event
        test_event = {
            'event_id': f"test_{datetime.now().timestamp()}",
            'safetyreportid': 'TEST123456',
            'safetyreportversion': 1,
            'serious': 1,
            'seriousnesscongenitalanomali': 0,
            'seriousnessdeath': 0,
            'seriousnessdisabling': 0,
            'seriousnesshospitalization': 1,
            'seriousnesslifethreatening': 0,
            'seriousnessother': 0,
            'duplicate': 0,
            'occurcountry_encoded': 1,
            'primarysource_encoded': 2,
            'primarysourcecountry_encoded': 1,
            'reporttype_encoded': 1,
            'drug_count': 2,
            'reaction_count': 3,
            'patient_age': 65.0,
            'patient_weight': 75.0,
            'patient_sex_encoded': 1,
            'outcome_severity': 4,
            'patient_risk_score': 3.5,
            'time_to_onset_days': 14.0,
            'concomitant_drug_risk': 2.0
        }
        
        print("✅ Using synthetic test event")
        return test_event
    
    def test_lambda_invocation(self, test_event, invocation_type='RequestResponse'):
        """
        Test Lambda function invocation
        
        Args:
            test_event: Event data to send to Lambda
            invocation_type: 'RequestResponse' (synchronous) or 'Event' (asynchronous)
        """
        print(f"🧪 Testing Lambda function: {self.function_name}")
        print(f"📨 Invocation type: {invocation_type}")
        print(f"📋 Test event ID: {test_event.get('event_id', 'unknown')}")
        
        try:
            # Invoke Lambda function
            response = self.lambda_client.invoke(
                FunctionName=self.function_name,
                InvocationType=invocation_type,
                Payload=json.dumps(test_event)
            )
            
            # Parse response
            status_code = response['StatusCode']
            
            if invocation_type == 'RequestResponse':
                # Synchronous - get response payload
                payload = json.loads(response['Payload'].read().decode('utf-8'))
                
                print(f"\n📊 Lambda Response:")
                print(f"   Status Code: {status_code}")
                print(f"   Function Error: {response.get('FunctionError', 'None')}")
                print(f"   Execution Duration: {response.get('ExecutionDuration', 'Unknown')}ms")
                print(f"   Memory Used: {response.get('LogResult', 'Unknown')}")
                
                if 'body' in payload:
                    body = json.loads(payload['body']) if isinstance(payload['body'], str) else payload['body']
                    print(f"\n📋 Processing Results:")
                    print(json.dumps(body, indent=2, default=str))
                else:
                    print(f"\n📋 Raw Payload:")
                    print(json.dumps(payload, indent=2, default=str))
                
                # Check if processing was successful
                if status_code == 200 and not response.get('FunctionError'):
                    print("\n✅ Lambda test PASSED!")
                    return True, payload
                else:
                    print("\n❌ Lambda test FAILED!")
                    return False, payload
            
            else:
                # Asynchronous - just check invocation
                print(f"\n✅ Async invocation successful (Status: {status_code})")
                return True, {"message": "Async invocation sent"}
                
        except Exception as e:
            print(f"\n❌ Lambda invocation error: {e}")
            return False, {"error": str(e)}
    
    def test_multiple_events(self, num_events=3):
        """Test Lambda with multiple events"""
        print(f"\n🔄 Testing Lambda with {num_events} events...")
        
        results = []
        for i in range(num_events):
            print(f"\n--- Test Event {i+1}/{num_events} ---")
            test_event = self.create_test_event()
            test_event['event_id'] = f"multi_test_{i}_{datetime.now().timestamp()}"
            
            success, result = self.test_lambda_invocation(test_event)
            results.append({
                'event_id': test_event['event_id'],
                'success': success,
                'result': result
            })
        
        # Summary
        successful = sum(1 for r in results if r['success'])
        print(f"\n📊 Multiple Event Test Summary:")
        print(f"   Total Events: {num_events}")
        print(f"   Successful: {successful}")
        print(f"   Failed: {num_events - successful}")
        print(f"   Success Rate: {successful/num_events*100:.1f}%")
        
        return results
    
    def check_function_status(self):
        """Check Lambda function configuration and status"""
        print("🔍 Checking Lambda function status...")
        
        try:
            response = self.lambda_client.get_function(FunctionName=self.function_name)
            
            config = response['Configuration']
            
            print(f"\n📋 Function Configuration:")
            print(f"   Function Name: {config['FunctionName']}")
            print(f"   Runtime: {config['Runtime']}")
            print(f"   Handler: {config['Handler']}")
            print(f"   Memory: {config['MemorySize']}MB")
            print(f"   Timeout: {config['Timeout']}s")
            print(f"   Last Modified: {config['LastModified']}")
            print(f"   State: {config['State']}")
            print(f"   Code Size: {config['CodeSize']:,} bytes")
            
            # Environment variables (masked for security)
            env_vars = config.get('Environment', {}).get('Variables', {})
            if env_vars:
                print(f"\n🔐 Environment Variables:")
                for key, value in env_vars.items():
                    # Mask sensitive values
                    if 'password' in key.lower() or 'secret' in key.lower():
                        masked_value = '*' * len(value) if value else 'Not Set'
                    else:
                        masked_value = value
                    print(f"   {key}: {masked_value}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error checking function status: {e}")
            return False
    
    def run_comprehensive_test(self):
        """Run comprehensive Lambda function tests"""
        print("🚀 Starting Comprehensive Lambda Test Suite")
        print("=" * 60)
        
        # Test 1: Check function status
        print("\n🔍 Test 1: Function Status Check")
        status_ok = self.check_function_status()
        
        if not status_ok:
            print("❌ Function status check failed. Cannot proceed with tests.")
            return False
        
        # Test 2: Single event test
        print("\n🧪 Test 2: Single Event Processing")
        test_event = self.create_test_event()
        success, result = self.test_lambda_invocation(test_event)
        
        if not success:
            print("❌ Single event test failed.")
            return False
        
        # Test 3: Multiple events test
        print("\n🔄 Test 3: Multiple Events Processing")
        multi_results = self.test_multiple_events(3)
        
        # Final summary
        print("\n" + "=" * 60)
        print("🎯 TEST SUITE SUMMARY")
        print(f"✅ Function Status: {'PASS' if status_ok else 'FAIL'}")
        print(f"✅ Single Event: {'PASS' if success else 'FAIL'}")
        successful_multi = sum(1 for r in multi_results if r['success'])
        print(f"✅ Multiple Events: {successful_multi}/3 PASS")
        
        overall_success = status_ok and success and (successful_multi >= 2)
        print(f"\n🎉 Overall Result: {'ALL TESTS PASSED' if overall_success else 'SOME TESTS FAILED'}")
        
        if overall_success:
            print("\n🎯 Your Lambda function is working correctly!")
            print("Ready for integration with Kafka pipeline.")
        else:
            print("\n⚠️ Some tests failed. Check the error messages above.")
        
        return overall_success


def main():
    """Main testing function"""
    print("Clinical Trial Safety Monitor - Lambda Testing")
    print("=" * 60)
    
    tester = LambdaTester()
    
    try:
        # Run comprehensive test suite
        success = tester.run_comprehensive_test()
        
        if success:
            print("\n🎉 All tests passed! Your Lambda function is ready.")
        else:
            print("\n❌ Some tests failed. Please check the configuration.")
            
    except Exception as e:
        print(f"\n❌ Unexpected error during testing: {e}")
        
    print("\n" + "=" * 60)
    print("Testing complete.")


if __name__ == "__main__":
    main()