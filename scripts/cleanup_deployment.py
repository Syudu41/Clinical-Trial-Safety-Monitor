"""
Manual Cleanup Script for Lambda Deployment
Use this to clean up deployment packages if needed
"""

import shutil
from pathlib import Path

def cleanup_deployment():
    """Clean up deployment directory manually"""
    project_root = Path.cwd()
    deployment_dir = project_root / "src" / "aws" / "deployment_package"
    zip_file = project_root / "clinical_safety_lambda.zip"
    
    print("🧹 Manual Lambda Deployment Cleanup")
    print("=" * 40)
    
    cleaned_count = 0
    
    # Clean deployment directory
    if deployment_dir.exists():
        keep_files = {'lambda_function.py', 'lambda_handler.py', 'requirements.txt'}
        
        for item in deployment_dir.iterdir():
            if item.name not in keep_files:
                if item.is_file():
                    item.unlink()
                    cleaned_count += 1
                elif item.is_dir():
                    shutil.rmtree(item)
                    cleaned_count += 1
        
        print(f"✅ Cleaned {cleaned_count} items from deployment directory")
    else:
        print("📁 Deployment directory doesn't exist")
    
    # Clean ZIP file
    if zip_file.exists():
        zip_file.unlink()
        print(f"✅ Removed ZIP file: {zip_file.name}")
    else:
        print("📦 No ZIP file to clean")
    
    print("\n🎯 Cleanup complete! Ready for fresh deployment.")

if __name__ == "__main__":
    cleanup_deployment()