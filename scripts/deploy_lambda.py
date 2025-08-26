"""
AWS Lambda Deployment Script - WITH S3 SUPPORT FOR LARGE PACKAGES
Packages and deploys the Clinical Trial Safety Event Processor to AWS Lambda
"""

import os
import sys
import json
import boto3
import zipfile
import shutil
import subprocess
from pathlib import Path
from datetime import datetime

class LambdaDeployer:
    """Handles packaging and deployment of Lambda function"""
    
    def __init__(self):
        self.project_root = Path.cwd()
        self.deployment_dir = self.project_root / "src" / "aws" / "deployment_package"
        self.zip_file = self.project_root / "clinical_safety_lambda.zip"
        self.lambda_client = boto3.client('lambda')
        self.iam_client = boto3.client('iam')
        
        # Lambda configuration
        self.function_name = 'clinical-safety-processor'
        self.runtime = 'python3.11'
        self.handler = 'lambda_function.lambda_handler'
        self.timeout = 300  # 5 minutes
        self.memory_size = 512  # MB
    
    def clean_deployment_directory(self):
        """Clean deployment directory of installed packages"""
        if not self.deployment_dir.exists():
            return
            
        # Keep these essential files
        keep_files = {'lambda_function.py', 'lambda_handler.py', 'requirements.txt'}
        
        cleaned_count = 0
        for item in self.deployment_dir.iterdir():
            if item.name not in keep_files:
                try:
                    if item.is_file():
                        item.unlink()
                        cleaned_count += 1
                    elif item.is_dir():
                        shutil.rmtree(item)
                        cleaned_count += 1
                except Exception as e:
                    print(f"⚠️ Could not clean {item}: {e}")
        
        if cleaned_count > 0:
            print(f"✅ Cleaned {cleaned_count} package folders/files")
        else:
            print("✅ Deployment directory already clean")
        
    def create_deployment_package(self):
        """Create deployment package directory and files"""
        print("📦 Creating Lambda deployment package...")
        
        # Create deployment directory
        self.deployment_dir.mkdir(parents=True, exist_ok=True)
        
        # Clean existing files first
        print("🧹 Cleaning deployment directory...")
        self.clean_deployment_directory()
        
        # Copy main Lambda files
        source_files = [
            (["src/aws/lambda_handler.py"], "lambda_handler.py"),
            (["src/aws/lambda_function.py"], "lambda_function.py"),
            (["lambda_requirements.txt"], "requirements.txt")
        ]
        
        for possible_sources, dest_name in source_files:
            copied = False
            for src_path in possible_sources:
                src_full = self.project_root / src_path
                if src_full.exists():
                    shutil.copy2(src_full, self.deployment_dir / dest_name)
                    print(f"✅ Copied {src_path} → {dest_name}")
                    copied = True
                    break
            
            if not copied:
                if dest_name == "requirements.txt":
                    print("📝 Creating minimal Lambda requirements.txt...")
                    default_requirements = """pandas
numpy
scikit-learn
psycopg2-binary"""
                    with open(self.deployment_dir / "requirements.txt", "w") as f:
                        f.write(default_requirements)
                    print("✅ Created minimal requirements.txt")
                else:
                    print(f"❌ Could not find {dest_name}")
                    if dest_name == "lambda_function.py":
                        return False
        
        print(f"✅ Deployment package created in {self.deployment_dir}")
        return True
    
    def install_dependencies(self):
        """Install Python dependencies for Lambda deployment package"""
        print("📚 Installing Lambda dependencies...")
        
        requirements_file = self.deployment_dir / "requirements.txt"
        if not requirements_file.exists():
            print("❌ requirements.txt not found")
            return False
        
        # Show what we're installing
        print("📋 Installing these packages for Lambda:")
        with open(requirements_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    print(f"   - {line}")
        
        try:
            print(f"📦 Installing packages to: {self.deployment_dir}")
            result = subprocess.run([
                sys.executable, "-m", "pip", "install",
                "-r", str(requirements_file),
                "-t", str(self.deployment_dir),
                "--only-binary=all",  # Force pre-compiled wheels
                "--quiet"
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"❌ pip install failed: {result.stderr}")
                return False
            
            # Count installed packages
            package_count = 0
            for item in self.deployment_dir.iterdir():
                if item.is_dir() and not item.name.startswith('.') and item.name not in {'__pycache__'}:
                    package_count += 1
            
            print(f"✅ Successfully installed {package_count} packages for Lambda")
            return True
            
        except Exception as e:
            print(f"❌ Failed to install dependencies: {e}")
            return False
    
    def create_zip_package(self):
        """Create ZIP package for Lambda deployment"""
        print("🗜️ Creating ZIP package...")
        
        # Remove existing ZIP file
        if self.zip_file.exists():
            self.zip_file.unlink()
        
        # Create ZIP file
        file_count = 0
        with zipfile.ZipFile(self.zip_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in self.deployment_dir.rglob("*"):
                if file_path.is_file():
                    arcname = file_path.relative_to(self.deployment_dir)
                    zipf.write(file_path, arcname)
                    file_count += 1
        
        # Check ZIP size
        zip_size = self.zip_file.stat().st_size / (1024 * 1024)  # MB
        print(f"✅ ZIP package created: {self.zip_file.name} ({zip_size:.1f} MB, {file_count} files)")
        
        if zip_size > 50:
            print("⚠️ Package exceeds 50MB - will use S3 deployment method")
        
        return True
    
    def upload_to_s3(self):
        """Upload ZIP package to S3 for large deployments"""
        print("📤 Uploading large package to S3...")
        
        # Use your processed bucket for deployment
        deployment_bucket = os.environ.get('AWS_S3_PROCESSED_BUCKET', 'clinical-safety-processed-2025-yourname')
        deployment_key = f"lambda-deployments/{self.function_name}.zip"
        
        try:
            s3_client = boto3.client('s3')
            
            print(f"📦 Uploading to s3://{deployment_bucket}/{deployment_key}")
            with open(self.zip_file, 'rb') as f:
                s3_client.upload_fileobj(
                    f, 
                    deployment_bucket, 
                    deployment_key,
                    ExtraArgs={'ServerSideEncryption': 'AES256'}
                )
            
            print("✅ Package uploaded to S3 successfully")
            return deployment_bucket, deployment_key
            
        except Exception as e:
            print(f"❌ Failed to upload to S3: {e}")
            return None, None
    
    def create_lambda_role(self):
        """Create or get existing IAM role for Lambda function"""
        role_name = f"{self.function_name}-role"
        
        try:
            # Check if role already exists
            try:
                response = self.iam_client.get_role(RoleName=role_name)
                role_arn = response['Role']['Arn']
                print(f"✅ Using existing IAM role: {role_arn}")
                return role_arn
            except self.iam_client.exceptions.NoSuchEntityException:
                pass
            
            # Create new role
            print(f"🔑 Creating IAM role: {role_name}")
            
            trust_policy = {
                "Version": "2012-10-17",
                "Statement": [{
                    "Effect": "Allow",
                    "Principal": {"Service": "lambda.amazonaws.com"},
                    "Action": "sts:AssumeRole"
                }]
            }
            
            response = self.iam_client.create_role(
                RoleName=role_name,
                AssumeRolePolicyDocument=json.dumps(trust_policy)
            )
            role_arn = response['Role']['Arn']
            
            # Attach policies
            policies = [
                'arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole',
                'arn:aws:iam::aws:policy/service-role/AWSLambdaVPCAccessExecutionRole'
            ]
            
            for policy_arn in policies:
                self.iam_client.attach_role_policy(RoleName=role_name, PolicyArn=policy_arn)
            
            # Custom S3 policy
            s3_policy = {
                "Version": "2012-10-17",
                "Statement": [{
                    "Effect": "Allow",
                    "Action": ["s3:GetObject", "s3:PutObject"],
                    "Resource": ["arn:aws:s3:::clinical-safety-*/*"]
                }]
            }
            
            self.iam_client.put_role_policy(
                RoleName=role_name,
                PolicyName=f"{role_name}-s3-policy",
                PolicyDocument=json.dumps(s3_policy)
            )
            
            print(f"✅ IAM role created: {role_arn}")
            
            # Wait for role propagation
            import time
            time.sleep(10)
            
            return role_arn
            
        except Exception as e:
            print(f"❌ Failed to create IAM role: {e}")
            return None
    
    def deploy_lambda_function(self, role_arn):
        """Deploy Lambda function to AWS (handles S3 for large packages)"""
        print(f"🚀 Deploying Lambda function: {self.function_name}")
        
        # Check if we need S3 deployment (for large packages)
        zip_size = self.zip_file.stat().st_size / (1024 * 1024)  # MB
        use_s3_deployment = zip_size > 50
        
        # Environment variables
        environment_vars = {
            'S3_PROCESSED_BUCKET': os.environ.get('AWS_S3_PROCESSED_BUCKET', 'clinical-safety-processed-2025-yourname'),
            'RDS_HOST': os.environ.get('AWS_RDS_HOST', ''),
            'RDS_DATABASE': os.environ.get('AWS_RDS_DATABASE', 'clinical_safety'),
            'RDS_USER': os.environ.get('AWS_RDS_USER', 'postgres'),
            'RDS_PASSWORD': os.environ.get('AWS_RDS_PASSWORD', ''),
            'RDS_PORT': os.environ.get('AWS_RDS_PORT', '5432')
        }
        
        try:
            # Prepare deployment code config
            if use_s3_deployment:
                print("📤 Using S3 deployment for large package...")
                s3_bucket, s3_key = self.upload_to_s3()
                if not s3_bucket:
                    return None
                
                code_config = {
                    'S3Bucket': s3_bucket,
                    'S3Key': s3_key
                }
            else:
                print("📦 Using direct ZIP deployment...")
                with open(self.zip_file, 'rb') as f:
                    zip_content = f.read()
                code_config = {'ZipFile': zip_content}
            
            # Check if function exists
            try:
                self.lambda_client.get_function(FunctionName=self.function_name)
                print("📝 Updating existing Lambda function...")
                
                # Update function code
                if use_s3_deployment:
                    response = self.lambda_client.update_function_code(
                        FunctionName=self.function_name,
                        S3Bucket=s3_bucket,
                        S3Key=s3_key
                    )
                else:
                    response = self.lambda_client.update_function_code(
                        FunctionName=self.function_name,
                        ZipFile=code_config['ZipFile']
                    )
                
                # Update configuration
                self.lambda_client.update_function_configuration(
                    FunctionName=self.function_name,
                    Runtime=self.runtime,
                    Handler=self.handler,
                    Timeout=self.timeout,
                    MemorySize=self.memory_size,
                    Environment={'Variables': environment_vars}
                )
                
            except self.lambda_client.exceptions.ResourceNotFoundException:
                print("🆕 Creating new Lambda function...")
                
                # Create new function
                create_params = {
                    'FunctionName': self.function_name,
                    'Runtime': self.runtime,
                    'Role': role_arn,
                    'Handler': self.handler,
                    'Code': code_config,
                    'Description': 'Clinical Trial Safety Event Processor',
                    'Timeout': self.timeout,
                    'MemorySize': self.memory_size,
                    'Environment': {'Variables': environment_vars},
                    'PackageType': 'Zip'
                }
                
                response = self.lambda_client.create_function(**create_params)
            
            function_arn = response['FunctionArn']
            print(f"✅ Lambda function deployed successfully!")
            print(f"📋 Function ARN: {function_arn}")
            print(f"📊 Deployment method: {'S3 (Large Package)' if use_s3_deployment else 'Direct ZIP'}")
            return function_arn
            
        except Exception as e:
            print(f"❌ Failed to deploy Lambda function: {e}")
            return None
    
    def deploy(self):
        """Main deployment workflow"""
        print("🚀 Starting Lambda deployment process...")
        print("=" * 60)
        
        # Step 1: Create deployment package
        if not self.create_deployment_package():
            return False
        
        # Step 2: Install dependencies
        if not self.install_dependencies():
            return False
        
        # Step 3: Create ZIP package
        if not self.create_zip_package():
            return False
        
        # Step 4: Clean up packages after zipping
        self.clean_deployment_directory()
        
        # Step 5: Create IAM role
        role_arn = self.create_lambda_role()
        if not role_arn:
            return False
        
        # Step 6: Deploy Lambda function
        function_arn = self.deploy_lambda_function(role_arn)
        if not function_arn:
            return False
        
        zip_size = self.zip_file.stat().st_size / (1024 * 1024)
        
        print("\n" + "=" * 60)
        print("🎉 DEPLOYMENT SUCCESSFUL!")
        print(f"📋 Function Name: {self.function_name}")
        print(f"📋 Function ARN: {function_arn}")
        print(f"📁 Package Size: {zip_size:.1f}MB")
        print(f"📊 Method: {'S3 Deployment' if zip_size > 50 else 'Direct Upload'}")
        print(f"⚡ Memory: {self.memory_size}MB, Timeout: {self.timeout}s")
        print("\n🧪 Ready for testing with scripts/test_lambda.py!")
        
        return True
    
    def cleanup(self):
        """Clean up temporary files"""
        if self.zip_file.exists():
            self.zip_file.unlink()
            print(f"✅ Cleaned up ZIP file")
        
        self.clean_deployment_directory()


if __name__ == "__main__":
    print("AWS Lambda Deployment - With S3 Support for Large Packages")
    print("=" * 60)
    
    deployer = LambdaDeployer()
    
    try:
        success = deployer.deploy()
        
        if success:
            print("\n🎯 Next Steps:")
            print("1. Test: python scripts/test_lambda.py")
            print("2. Check AWS Console for function details")
            print("3. Monitor CloudWatch logs")
        else:
            print("\n❌ Deployment failed. Check error messages above.")
            
    except KeyboardInterrupt:
        print("\n⚠️ Deployment cancelled")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        deployer.cleanup()