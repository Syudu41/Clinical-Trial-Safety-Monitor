"""
AWS S3 Utilities for Clinical Trial Safety Monitoring System
Handles file uploads, downloads, and management in S3
"""

import boto3
import pandas as pd
import pickle
import logging
import os
from typing import Optional, List
from botocore.exceptions import ClientError, NoCredentialsError

# Import project configuration
from config import aws_config

class S3Manager:
    """Manages interactions with AWS S3 for data storage and retrieval"""
    
    def __init__(self):
        """Initialize S3 client with AWS credentials"""
        try:
            self.s3_client = boto3.client('s3', region_name=aws_config.AWS_REGION)
            self.raw_bucket = aws_config.S3_RAW_BUCKET
            self.processed_bucket = aws_config.S3_PROCESSED_BUCKET
            
            # Test connection
            self.s3_client.head_bucket(Bucket=self.raw_bucket)
            logging.info(f"✅ Connected to S3. Raw bucket: {self.raw_bucket}")
            
        except NoCredentialsError:
            logging.error("❌ AWS credentials not found. Run 'aws configure'")
            raise
        except ClientError as e:
            logging.error(f"❌ Error connecting to S3: {e}")
            raise
    
    def upload_file(self, local_file_path: str, s3_key: str, bucket_type: str = 'processed') -> bool:
        """
        Upload a local file to S3
        
        Args:
            local_file_path: Path to local file
            s3_key: S3 object key (file path in bucket)
            bucket_type: 'raw' or 'processed' bucket
            
        Returns:
            bool: True if successful, False otherwise
        """
        bucket_name = self.raw_bucket if bucket_type == 'raw' else self.processed_bucket
        
        try:
            self.s3_client.upload_file(local_file_path, bucket_name, s3_key)
            file_size = os.path.getsize(local_file_path)
            logging.info(f"✅ Uploaded {local_file_path} to s3://{bucket_name}/{s3_key} ({file_size:,} bytes)")
            return True
            
        except FileNotFoundError:
            logging.error(f"❌ Local file not found: {local_file_path}")
            return False
        except ClientError as e:
            logging.error(f"❌ Error uploading to S3: {e}")
            return False
    
    def download_file(self, s3_key: str, local_file_path: str, bucket_type: str = 'processed') -> bool:
        """
        Download a file from S3 to local storage
        
        Args:
            s3_key: S3 object key (file path in bucket)
            local_file_path: Local path to save file
            bucket_type: 'raw' or 'processed' bucket
            
        Returns:
            bool: True if successful, False otherwise
        """
        bucket_name = self.raw_bucket if bucket_type == 'raw' else self.processed_bucket
        
        try:
            # Create local directory if it doesn't exist
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
            
            self.s3_client.download_file(bucket_name, s3_key, local_file_path)
            file_size = os.path.getsize(local_file_path)
            logging.info(f"✅ Downloaded s3://{bucket_name}/{s3_key} to {local_file_path} ({file_size:,} bytes)")
            return True
            
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                logging.error(f"❌ File not found in S3: s3://{bucket_name}/{s3_key}")
            else:
                logging.error(f"❌ Error downloading from S3: {e}")
            return False
    
    def upload_dataframe(self, df: pd.DataFrame, s3_key: str, bucket_type: str = 'processed') -> bool:
        """
        Upload a pandas DataFrame directly to S3 as CSV
        
        Args:
            df: pandas DataFrame to upload
            s3_key: S3 object key (should end with .csv)
            bucket_type: 'raw' or 'processed' bucket
            
        Returns:
            bool: True if successful, False otherwise
        """
        bucket_name = self.raw_bucket if bucket_type == 'raw' else self.processed_bucket
        
        try:
            # Convert DataFrame to CSV string
            csv_buffer = df.to_csv(index=False)
            
            # Upload to S3
            self.s3_client.put_object(
                Bucket=bucket_name,
                Key=s3_key,
                Body=csv_buffer,
                ContentType='text/csv'
            )
            
            logging.info(f"✅ Uploaded DataFrame to s3://{bucket_name}/{s3_key} ({len(df)} rows)")
            return True
            
        except ClientError as e:
            logging.error(f"❌ Error uploading DataFrame to S3: {e}")
            return False
    
    def download_dataframe(self, s3_key: str, bucket_type: str = 'processed') -> Optional[pd.DataFrame]:
        """
        Download a CSV file from S3 directly as pandas DataFrame
        
        Args:
            s3_key: S3 object key (CSV file)
            bucket_type: 'raw' or 'processed' bucket
            
        Returns:
            pandas.DataFrame or None if error
        """
        bucket_name = self.raw_bucket if bucket_type == 'raw' else self.processed_bucket
        
        try:
            response = self.s3_client.get_object(Bucket=bucket_name, Key=s3_key)
            df = pd.read_csv(response['Body'])
            logging.info(f"✅ Downloaded DataFrame from s3://{bucket_name}/{s3_key} ({len(df)} rows)")
            return df
            
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                logging.error(f"❌ File not found in S3: s3://{bucket_name}/{s3_key}")
            else:
                logging.error(f"❌ Error downloading DataFrame from S3: {e}")
            return None
    
    def upload_model(self, model, s3_key: str) -> bool:
        """
        Upload a trained ML model to S3
        
        Args:
            model: Trained scikit-learn model
            s3_key: S3 object key (should end with .pkl)
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Serialize model to bytes
            model_bytes = pickle.dumps(model)
            
            # Upload to processed bucket
            self.s3_client.put_object(
                Bucket=self.processed_bucket,
                Key=s3_key,
                Body=model_bytes,
                ContentType='application/octet-stream'
            )
            
            logging.info(f"✅ Uploaded model to s3://{self.processed_bucket}/{s3_key}")
            return True
            
        except Exception as e:
            logging.error(f"❌ Error uploading model to S3: {e}")
            return False
    
    def download_model(self, s3_key: str):
        """
        Download a trained ML model from S3
        
        Args:
            s3_key: S3 object key (pickle file)
            
        Returns:
            Trained model object or None if error
        """
        try:
            response = self.s3_client.get_object(Bucket=self.processed_bucket, Key=s3_key)
            model = pickle.loads(response['Body'].read())
            logging.info(f"✅ Downloaded model from s3://{self.processed_bucket}/{s3_key}")
            return model
            
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                logging.error(f"❌ Model not found in S3: s3://{self.processed_bucket}/{s3_key}")
            else:
                logging.error(f"❌ Error downloading model from S3: {e}")
            return None
        except Exception as e:
            logging.error(f"❌ Error deserializing model: {e}")
            return None
    
    def list_files(self, prefix: str = "", bucket_type: str = 'processed') -> List[str]:
        """
        List files in S3 bucket with optional prefix filter
        
        Args:
            prefix: Filter files by prefix (folder path)
            bucket_type: 'raw' or 'processed' bucket
            
        Returns:
            List of S3 object keys
        """
        bucket_name = self.raw_bucket if bucket_type == 'raw' else self.processed_bucket
        
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=bucket_name,
                Prefix=prefix
            )
            
            files = []
            if 'Contents' in response:
                files = [obj['Key'] for obj in response['Contents']]
                logging.info(f"✅ Found {len(files)} files in s3://{bucket_name}/{prefix}")
            else:
                logging.info(f"📁 No files found in s3://{bucket_name}/{prefix}")
            
            return files
            
        except ClientError as e:
            logging.error(f"❌ Error listing S3 files: {e}")
            return []
    
    def delete_file(self, s3_key: str, bucket_type: str = 'processed') -> bool:
        """
        Delete a file from S3
        
        Args:
            s3_key: S3 object key to delete
            bucket_type: 'raw' or 'processed' bucket
            
        Returns:
            bool: True if successful, False otherwise
        """
        bucket_name = self.raw_bucket if bucket_type == 'raw' else self.processed_bucket
        
        try:
            self.s3_client.delete_object(Bucket=bucket_name, Key=s3_key)
            logging.info(f"✅ Deleted s3://{bucket_name}/{s3_key}")
            return True
            
        except ClientError as e:
            logging.error(f"❌ Error deleting from S3: {e}")
            return False
    
    def get_file_info(self, s3_key: str, bucket_type: str = 'processed') -> Optional[dict]:
        """
        Get metadata about a file in S3
        
        Args:
            s3_key: S3 object key
            bucket_type: 'raw' or 'processed' bucket
            
        Returns:
            Dictionary with file metadata or None if error
        """
        bucket_name = self.raw_bucket if bucket_type == 'raw' else self.processed_bucket
        
        try:
            response = self.s3_client.head_object(Bucket=bucket_name, Key=s3_key)
            
            info = {
                'size': response['ContentLength'],
                'last_modified': response['LastModified'],
                'content_type': response.get('ContentType', 'unknown'),
                'etag': response['ETag'].strip('"')
            }
            
            logging.info(f"✅ File info for s3://{bucket_name}/{s3_key}: {info['size']:,} bytes")
            return info
            
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                logging.error(f"❌ File not found: s3://{bucket_name}/{s3_key}")
            else:
                logging.error(f"❌ Error getting file info: {e}")
            return None


# Example usage and testing functions
def test_s3_connection():
    """Test S3 connection and basic operations"""
    print("🧪 Testing S3 Connection...")
    
    try:
        s3_manager = S3Manager()
        
        # List existing files
        files = s3_manager.list_files()
        print(f"📁 Files in processed bucket: {len(files)}")
        for file in files[:5]:  # Show first 5 files
            print(f"   - {file}")
        
        # Test model download if it exists
        if 'models/gradient_boosting_model.pkl' in files:
            model = s3_manager.download_model('models/gradient_boosting_model.pkl')
            if model is not None:
                print("✅ Model download successful!")
            else:
                print("❌ Model download failed")
        
        return True
        
    except Exception as e:
        print(f"❌ S3 connection test failed: {e}")
        return False


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Run connection test
    test_s3_connection()