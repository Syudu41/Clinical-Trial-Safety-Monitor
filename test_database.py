"""
Database Connection Test Script
Tests PostgreSQL RDS connection and creates tables
"""

import sys
import os

# Add the project root to Python path
sys.path.append('.')

def test_database():
    """Test database connection and table creation"""
    print("🧪 Testing PostgreSQL Database Connection...")
    print("=" * 50)
    
    try:
        # Import the database manager
        from src.aws.database_utils import DatabaseManager
        
        # Create database manager instance
        print("📡 Connecting to RDS PostgreSQL...")
        db = DatabaseManager()
        
        # Test connection
        print("🔍 Testing connection...")
        success = db.test_connection()
        
        if success:
            print("✅ Database connection successful!")
            print()
            
            # Test table creation
            print("🏗️ Creating database tables...")
            tables_created = db.create_tables()
            
            if tables_created:
                print("✅ Database tables created successfully!")
                print()
                
                # Get system stats
                print("📊 Getting system statistics...")
                stats = db.get_system_stats()
                print("System Stats:")
                for key, value in stats.items():
                    print(f"  - {key}: {value}")
                
                print()
                print("🎉 Database setup complete!")
                return True
                
            else:
                print("❌ Table creation failed!")
                return False
        else:
            print("❌ Database connection failed!")
            print("Check your .env file credentials and RDS status")
            return False
            
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("Make sure src/aws/database_utils.py exists and is properly configured")
        return False
        
    except Exception as e:
        print(f"❌ Connection Error: {e}")
        print()
        print("Common issues:")
        print("1. Check your .env file has the correct RDS credentials")
        print("2. Verify your RDS instance is running and accessible")
        print("3. Make sure psycopg2-binary is installed: pip install psycopg2-binary")
        return False


def test_s3():
    """Test S3 connection"""
    print("\n🧪 Testing S3 Connection...")
    print("=" * 30)
    
    try:
        from src.aws.s3_utils import S3Manager
        
        print("📡 Connecting to S3...")
        s3 = S3Manager()
        
        # List files in processed bucket
        files = s3.list_files()
        print(f"📁 Files in processed bucket: {len(files)}")
        for file in files[:5]:  # Show first 5 files
            print(f"   - {file}")
        
        print("✅ S3 connection successful!")
        return True
        
    except Exception as e:
        print(f"❌ S3 Error: {e}")
        return False


if __name__ == "__main__":
    print("Clinical Trial Safety Monitoring - Database & S3 Test")
    print("=" * 60)
    
    # Test database
    db_success = test_database()
    
    # Test S3
    s3_success = test_s3()
    
    print("\n" + "=" * 60)
    print("FINAL RESULTS:")
    print(f"Database: {'✅ PASS' if db_success else '❌ FAIL'}")
    print(f"S3:       {'✅ PASS' if s3_success else '❌ FAIL'}")
    
    if db_success and s3_success:
        print("\n🎉 All tests passed! Ready for AWS Lambda setup!")
    else:
        print("\n⚠️ Some tests failed. Please check the error messages above.")