"""
AWS Lambda Deployment Script
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
        self.runtime = 'python3.9'  # Stable Lambda runtime
        self.handler = 'lambda_function.lambda_handler'
        self.timeout = 300  # 5 minutes
        self.memory_size = 512  # MB
        
    def create_deployment_package(self):
        """Create deployment package directory and files"""
        print("📦 Creating Lambda deployment package...")
        
        # Create deployment directory
        self.deployment_dir.mkdir(parents=True, exist_ok=True)
        
        # Clean existing files
        for file in self.deployment_dir.glob("*"):
            if file.is_file():
                file.unlink()
            elif file.is_dir():
                shutil.rmtree(file)
        
        # Copy main Lambda files - check multiple possible locations
        source_files = [
            # (possible_source_paths, dest_name)
            (["src/aws/lambda_handler.py"], "lambda_handler.py"),
            (["src/aws/deployment_package/lambda_function.py", "src/aws/lambda_function.py"], "lambda_function.py"),
            (["src/aws/deployment_package/requirements.txt", "src/aws/requirements.txt", "requirements.txt"], "requirements.txt")
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
                print(f"⚠️ Warning: Could not find {dest_name} in any of these locations:")
                for src_path in possible_sources:
                    print(f"   - {src_path}")
                
                # Create default requirements.txt if missing
                if dest_name == "requirements.txt":
                    print("📝 Creating default requirements.txt...")
                    default_requirements = """# Lambda Deployment Requirements
scikit-learn==1.3.2
pandas==2.1.4  
numpy==1.24.4
psycopg2-binary==2.9.9
"""
                    with open(self.deployment_dir / "requirements.txt", "w") as f:
                        f.write(default_requirements)
                    print("✅ Created default requirements.txt")
        
        print(f"✅ Deployment package created in {self.deployment_dir}")
    
    def install_dependencies(self):
        """Install Python dependencies for Lambda deployment package"""
        print("📚 Installing Lambda dependencies...")
        print("ℹ️  Note: Installing packages INTO deployment package (separate from your local environment)")
        
        requirements_file = self.deployment_dir / "requirements.txt"
        if not requirements_file.exists():
            print("❌ requirements.txt not found")
            return False
        
        try:
            # Install dependencies directly into deployment directory
            # This creates a local copy of packages that will be zipped with Lambda function
            print(f"📦 Installing packages to: {self.deployment_dir}")
            subprocess.run([
                sys.executable, "-m", "pip", "install",
                "-r", str(requirements_file),
                "-t", str(self.deployment_dir),  # Install TO deployment directory
                "--no-deps"  # Avoid dependency conflicts with Lambda runtime
            ], check=True, capture_output=True, text=True)
            
            # Show what was installed
            installed_packages = []
            for item in self.deployment_dir.iterdir():
                if item.is_dir() and not item.name.startswith('.'):
                    installed_packages.append(item.name)
            
            print(f"✅ Installed {len(installed_packages)} packages for Lambda:")
            for pkg in sorted(installed_packages)[:5]:  # Show first 5
                print(f"   - {pkg}")
            if len(installed_packages) > 5:
                print(f"   ... and {len(installed_packages) - 5} more")
            
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install dependencies: {e}")
            print(f"Error output: {e.stderr}")
            return False
    
    def create_zip_package(self):
        """Create ZIP package for Lambda deployment"""
        print("🗜️ Creating ZIP package...")
        
        # Remove existing ZIP file
        if self.zip_file.exists():
            self.zip_file.unlink()
        
        # Create ZIP file
        with zipfile.ZipFile(self.zip_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in self.deployment_dir.rglob("*"):
                if file_path.is_file():
                    # Add file to ZIP with relative path
                    arcname = file_path.relative_to(self.deployment_dir)
                    zipf.write(file_path, arcname)
        
        # Check ZIP size (Lambda has 50MB limit for direct upload)
        zip_size = self.zip_file.stat().st_size / (1024 * 1024)  # MB
        print(f"✅ ZIP package created: {self.zip_file.name} ({zip_size:.1f} MB)")
        
        if zip_size > 50:
            print("⚠️ Warning: Package size exceeds 50MB. Consider using S3 for deployment.")
        
        return zip_size <= 50
    
    def create_lambda_role(self):
        """Create IAM role for Lambda function"""
        role_name = f"{self.function_name}-role"
        
        # Trust policy for Lambda
        trust_policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {"Service": "lambda.amazonaws.com"},
                    "Action": "sts:AssumeRole"
                }
            ]
        }
        
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
            response = self.iam_client.create_role(
                RoleName=role_name,
                AssumeRolePolicyDocument=json.dumps(trust_policy),
                Description=f"IAM role for {self.function_name} Lambda function"
            )
            
            role_arn = response['Role']['Arn']
            
            # Attach basic Lambda execution policy
            self.iam_client.attach_role_policy(
                RoleName=role_name,
                PolicyArn='arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole'
            )
            
            # Attach VPC execution policy (for RDS access)
            self.iam_client.attach_role_policy(
                RoleName=role_name,
                PolicyArn='arn:aws:iam::aws:policy/service-role/AWSLambdaVPCAccessExecutionRole'
            )
            
            # Create custom policy for S3 and RDS access
            custom_policy = {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": [
                            "s3:GetObject",
                            "s3:PutObject"
                        ],
                        "Resource": [
                            "arn:aws:s3:::clinical-safety-*/*"
                        ]
                    }
                ]
            }
            
            self.iam_client.put_role_policy(
                RoleName=role_name,
                PolicyName=f"{role_name}-policy",
                PolicyDocument=json.dumps(custom_policy)
            )
            
            print(f"✅ IAM role created: {role_arn}")
            
            # Wait for role propagation
            import time
            print("⏳ Waiting for role propagation...")
            time.sleep(10)
            
            return role_arn
            
        except Exception as e:
            print(f"❌ Failed to create IAM role: {e}")
            return None
    
    def deploy_lambda_function(self, role_arn):
        """Deploy Lambda function to AWS"""
        print(f"🚀 Deploying Lambda function: {self.function_name}")
        
        # Read ZIP package
        with open(self.zip_file, 'rb') as f:
            zip_content = f.read()
        
        # Environment variables for Lambda
        environment_vars = {
            'S3_PROCESSED_BUCKET': 'clinical-safety-processed-2025-yourname',  # Update with actual bucket
            'RDS_HOST': os.environ.get('AWS_RDS_HOST', ''),
            'RDS_DATABASE': os.environ.get('AWS_RDS_DATABASE', 'clinical_safety'),
            'RDS_USER': os.environ.get('AWS_RDS_USER', 'postgres'),
            'RDS_PASSWORD': os.environ.get('AWS_RDS_PASSWORD', ''),
            'RDS_PORT': os.environ.get('AWS_RDS_PORT', '5432')
        }
        
        try:
            # Check if function exists
            try:
                self.lambda_client.get_function(FunctionName=self.function_name)
                # Function exists, update it
                print("📝 Updating existing Lambda function...")
                
                response = self.lambda_client.update_function_code(
                    FunctionName=self.function_name,
                    ZipFile=zip_content
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
                # Function doesn't exist, create it
                print("🆕 Creating new Lambda function...")
                
                response = self.lambda_client.create_function(
                    FunctionName=self.function_name,
                    Runtime=self.runtime,
                    Role=role_arn,
                    Handler=self.handler,
                    Code={'ZipFile': zip_content},
                    Description='Clinical Trial Safety Event Processor',
                    Timeout=self.timeout,
                    MemorySize=self.memory_size,
                    Environment={'Variables': environment_vars},
                    PackageType='Zip'
                )
            
            function_arn = response['FunctionArn']
            print(f"✅ Lambda function deployed successfully!")
            print(f"📋 Function ARN: {function_arn}")
            return function_arn
            
        except Exception as e:
            print(f"❌ Failed to deploy Lambda function: {e}")
            return None
    
    def deploy(self):
        """Main deployment workflow"""
        print("🚀 Starting Lambda deployment process...")
        print("=" * 60)
        
        # Step 1: Create deployment package
        self.create_deployment_package()
        
        # Step 2: Install dependencies
        if not self.install_dependencies():
            print("❌ Deployment failed: Could not install dependencies")
            return False
        
        # Step 3: Create ZIP package
        if not self.create_zip_package():
            print("❌ Deployment failed: Package too large")
            return False
        
        # Step 4: Create IAM role
        role_arn = self.create_lambda_role()
        if not role_arn:
            print("❌ Deployment failed: Could not create IAM role")
            return False
        
        # Step 5: Deploy Lambda function
        function_arn = self.deploy_lambda_function(role_arn)
        if not function_arn:
            print("❌ Deployment failed: Could not deploy Lambda function")
            return False
        
        print("\n" + "=" * 60)
        print("🎉 DEPLOYMENT SUCCESSFUL!")
        print(f"📋 Function Name: {self.function_name}")
        print(f"📋 Function ARN: {function_arn}")
        print(f"📋 Runtime: {self.runtime}")
        print(f"📋 Handler: {self.handler}")
        print(f"📋 Memory: {self.memory_size}MB")
        print(f"📋 Timeout: {self.timeout}s")
        print("\n🧪 Ready for testing!")
        
        return True
    
    def cleanup(self):
        """Clean up temporary files"""
        if self.zip_file.exists():
            self.zip_file.unlink()
            print(f"✅ Cleaned up {self.zip_file}")


if __name__ == "__main__":
    print("AWS Lambda Deployment for Clinical Trial Safety Monitor")
    print("=" * 60)
    
    deployer = LambdaDeployer()
    
    try:
        success = deployer.deploy()
        
        if success:
            print("\n🎯 Next Steps:")
            print("1. Test the Lambda function using scripts/test_lambda.py")
            print("2. Integrate with Kafka consumer")
            print("3. Set up CloudWatch monitoring")
        else:
            print("\n❌ Deployment failed. Check error messages above.")
            
    except KeyboardInterrupt:
        print("\n⚠️ Deployment cancelled by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
    finally:
        deployer.cleanup()