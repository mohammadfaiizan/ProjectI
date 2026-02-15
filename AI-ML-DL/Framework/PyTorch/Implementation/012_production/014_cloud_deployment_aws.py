import torch
import torch.nn as nn
import json
import base64
from typing import Dict, List, Tuple, Optional, Any

# Note: AWS operations require boto3
# Install with: pip install boto3

try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
    AWS_AVAILABLE = True
except ImportError:
    AWS_AVAILABLE = False
    print("Warning: boto3 not available. Install with: pip install boto3")

# Sample Model for AWS Deployment
class AWSDeploymentModel(nn.Module):
    """Model designed for AWS deployment"""
    
    def __init__(self, num_classes: int = 10):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x

# AWS SageMaker Deployment Manager
class SageMakerDeploymentManager:
    """Manage AWS SageMaker deployments"""
    
    def __init__(self, region_name: str = 'us-east-1'):
        if not AWS_AVAILABLE:
            raise ImportError("boto3 not available. Install with: pip install boto3")
        
        self.region_name = region_name
        self.sagemaker_client = boto3.client('sagemaker', region_name=region_name)
        self.s3_client = boto3.client('s3', region_name=region_name)
        self.iam_client = boto3.client('iam', region_name=region_name)
        
        print(f"✓ SageMaker client initialized for region: {region_name}")
    
    def create_model_artifacts(self, model: nn.Module, 
                              model_name: str,
                              s3_bucket: str,
                              s3_prefix: str = "pytorch-models") -> str:
        """Create and upload model artifacts to S3"""
        
        import tempfile
        import tarfile
        import os
        
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save model
            model_path = os.path.join(temp_dir, "model.pth")
            torch.save(model.state_dict(), model_path)
            
            # Create inference script
            inference_script = self._create_inference_script()
            script_path = os.path.join(temp_dir, "inference.py")
            with open(script_path, 'w') as f:
                f.write(inference_script)
            
            # Create requirements
            requirements = self._create_requirements()
            req_path = os.path.join(temp_dir, "requirements.txt")
            with open(req_path, 'w') as f:
                f.write(requirements)
            
            # Create model archive
            archive_path = os.path.join(temp_dir, "model.tar.gz")
            with tarfile.open(archive_path, "w:gz") as tar:
                tar.add(model_path, arcname="model.pth")
                tar.add(script_path, arcname="code/inference.py")
                tar.add(req_path, arcname="code/requirements.txt")
            
            # Upload to S3
            s3_key = f"{s3_prefix}/{model_name}/model.tar.gz"
            
            try:
                self.s3_client.upload_file(archive_path, s3_bucket, s3_key)
                model_artifacts_uri = f"s3://{s3_bucket}/{s3_key}"
                print(f"✓ Model artifacts uploaded: {model_artifacts_uri}")
                return model_artifacts_uri
            
            except ClientError as e:
                print(f"✗ Failed to upload model artifacts: {e}")
                raise
    
    def _create_inference_script(self) -> str:
        """Create SageMaker inference script"""
        
        script_content = '''
import torch
import torch.nn as nn
import json
import io
import base64
from PIL import Image
import torchvision.transforms as transforms

# Model definition (should match your model architecture)
class AWSDeploymentModel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def model_fn(model_dir):
    """Load model for inference"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = AWSDeploymentModel(num_classes=10)
    model.load_state_dict(torch.load(f"{model_dir}/model.pth", map_location=device))
    model.to(device)
    model.eval()
    
    return model

def input_fn(request_body, request_content_type):
    """Parse input data"""
    if request_content_type == "application/json":
        input_data = json.loads(request_body)
        
        if "image" in input_data:
            # Decode base64 image
            image_data = base64.b64decode(input_data["image"])
            image = Image.open(io.BytesIO(image_data)).convert('RGB')
            
            # Preprocess
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
            
            tensor = transform(image).unsqueeze(0)
            return tensor
        
        elif "instances" in input_data:
            # Batch processing
            tensors = []
            for instance in input_data["instances"]:
                image_data = base64.b64decode(instance["image"])
                image = Image.open(io.BytesIO(image_data)).convert('RGB')
                
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])
                ])
                
                tensor = transform(image)
                tensors.append(tensor)
            
            return torch.stack(tensors)
    
    raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    """Run prediction"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_data = input_data.to(device)
    
    with torch.no_grad():
        outputs = model(input_data)
        probabilities = torch.softmax(outputs, dim=1)
    
    return probabilities.cpu()

def output_fn(prediction, content_type):
    """Format output"""
    if content_type == "application/json":
        # Single prediction
        if prediction.size(0) == 1:
            probs = prediction[0].numpy()
            top_probs, top_indices = torch.topk(torch.from_numpy(probs), k=5)
            
            predictions = []
            for i in range(5):
                predictions.append({
                    "class_id": top_indices[i].item(),
                    "probability": top_probs[i].item()
                })
            
            return json.dumps({"predictions": predictions})
        
        # Batch predictions
        else:
            results = []
            for i in range(prediction.size(0)):
                probs = prediction[i].numpy()
                top_probs, top_indices = torch.topk(torch.from_numpy(probs), k=3)
                
                predictions = []
                for j in range(3):
                    predictions.append({
                        "class_id": top_indices[j].item(),
                        "probability": top_probs[j].item()
                    })
                
                results.append({"predictions": predictions})
            
            return json.dumps({"results": results})
    
    raise ValueError(f"Unsupported content type: {content_type}")
'''
        return script_content
    
    def _create_requirements(self) -> str:
        """Create requirements for SageMaker"""
        
        return '''
torch>=1.12.0
torchvision>=0.13.0
pillow>=8.3.0
numpy>=1.21.0
'''
    
    def create_sagemaker_model(self, model_name: str,
                              model_artifacts_uri: str,
                              execution_role_arn: str,
                              pytorch_version: str = "1.12.0",
                              python_version: str = "py38") -> str:
        """Create SageMaker model"""
        
        # Define container image
        container_image = f"763104351884.dkr.ecr.{self.region_name}.amazonaws.com/pytorch-inference:{pytorch_version}-gpu-{python_version}-ubuntu20.04-sagemaker"
        
        try:
            response = self.sagemaker_client.create_model(
                ModelName=model_name,
                PrimaryContainer={
                    'Image': container_image,
                    'ModelDataUrl': model_artifacts_uri,
                    'Environment': {
                        'SAGEMAKER_PROGRAM': 'inference.py',
                        'SAGEMAKER_SUBMIT_DIRECTORY': model_artifacts_uri,
                        'SAGEMAKER_CONTAINER_LOG_LEVEL': '20',
                        'SAGEMAKER_REGION': self.region_name
                    }
                },
                ExecutionRoleArn=execution_role_arn
            )
            
            print(f"✓ SageMaker model created: {model_name}")
            return response['ModelArn']
        
        except ClientError as e:
            print(f"✗ Failed to create SageMaker model: {e}")
            raise
    
    def create_endpoint_config(self, config_name: str,
                              model_name: str,
                              instance_type: str = "ml.m5.large",
                              instance_count: int = 1) -> str:
        """Create SageMaker endpoint configuration"""
        
        try:
            response = self.sagemaker_client.create_endpoint_config(
                EndpointConfigName=config_name,
                ProductionVariants=[
                    {
                        'VariantName': 'primary',
                        'ModelName': model_name,
                        'InitialInstanceCount': instance_count,
                        'InstanceType': instance_type,
                        'InitialVariantWeight': 1.0
                    }
                ]
            )
            
            print(f"✓ Endpoint configuration created: {config_name}")
            return response['EndpointConfigArn']
        
        except ClientError as e:
            print(f"✗ Failed to create endpoint configuration: {e}")
            raise
    
    def create_endpoint(self, endpoint_name: str,
                       config_name: str) -> str:
        """Create SageMaker endpoint"""
        
        try:
            response = self.sagemaker_client.create_endpoint(
                EndpointName=endpoint_name,
                EndpointConfigName=config_name
            )
            
            print(f"✓ Endpoint creation started: {endpoint_name}")
            print("  (This may take several minutes...)")
            return response['EndpointArn']
        
        except ClientError as e:
            print(f"✗ Failed to create endpoint: {e}")
            raise
    
    def wait_for_endpoint(self, endpoint_name: str, timeout_minutes: int = 15) -> bool:
        """Wait for endpoint to be in service"""
        
        import time
        
        print(f"Waiting for endpoint {endpoint_name} to be in service...")
        
        start_time = time.time()
        timeout_seconds = timeout_minutes * 60
        
        while time.time() - start_time < timeout_seconds:
            try:
                response = self.sagemaker_client.describe_endpoint(
                    EndpointName=endpoint_name
                )
                
                status = response['EndpointStatus']
                
                if status == 'InService':
                    print(f"✓ Endpoint {endpoint_name} is in service")
                    return True
                elif status == 'Failed':
                    print(f"✗ Endpoint {endpoint_name} failed")
                    return False
                else:
                    print(f"  Status: {status}")
                    time.sleep(30)
            
            except ClientError as e:
                print(f"Error checking endpoint status: {e}")
                time.sleep(30)
        
        print(f"✗ Timeout waiting for endpoint {endpoint_name}")
        return False
    
    def invoke_endpoint(self, endpoint_name: str,
                       payload: Dict[str, Any],
                       content_type: str = "application/json") -> Dict[str, Any]:
        """Invoke SageMaker endpoint"""
        
        runtime_client = boto3.client('sagemaker-runtime', region_name=self.region_name)
        
        try:
            response = runtime_client.invoke_endpoint(
                EndpointName=endpoint_name,
                ContentType=content_type,
                Body=json.dumps(payload)
            )
            
            result = json.loads(response['Body'].read().decode())
            return result
        
        except ClientError as e:
            print(f"✗ Failed to invoke endpoint: {e}")
            raise
    
    def delete_endpoint(self, endpoint_name: str) -> bool:
        """Delete SageMaker endpoint"""
        
        try:
            self.sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
            print(f"✓ Endpoint deleted: {endpoint_name}")
            return True
        
        except ClientError as e:
            print(f"✗ Failed to delete endpoint: {e}")
            return False

# AWS Lambda Deployment Manager
class LambdaDeploymentManager:
    """Manage AWS Lambda deployments"""
    
    def __init__(self, region_name: str = 'us-east-1'):
        if not AWS_AVAILABLE:
            raise ImportError("boto3 not available")
        
        self.region_name = region_name
        self.lambda_client = boto3.client('lambda', region_name=region_name)
        self.iam_client = boto3.client('iam', region_name=region_name)
        
    def create_lambda_deployment_package(self, model: nn.Module,
                                       model_name: str) -> str:
        """Create Lambda deployment package"""
        
        import tempfile
        import zipfile
        import os
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save model (CPU only for Lambda)
            model_path = os.path.join(temp_dir, "model.pth")
            torch.save(model.state_dict(), model_path)
            
            # Create Lambda function
            lambda_function = self._create_lambda_function()
            function_path = os.path.join(temp_dir, "lambda_function.py")
            with open(function_path, 'w') as f:
                f.write(lambda_function)
            
            # Create deployment package
            package_path = f"{model_name}_lambda.zip"
            with zipfile.ZipFile(package_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                zipf.write(model_path, "model.pth")
                zipf.write(function_path, "lambda_function.py")
            
            print(f"✓ Lambda deployment package created: {package_path}")
            return package_path
    
    def _create_lambda_function(self) -> str:
        """Create Lambda function code"""
        
        function_code = '''
import json
import torch
import torch.nn as nn
import base64
import io
from PIL import Image
import torchvision.transforms as transforms

# Model definition
class AWSDeploymentModel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Global model (loaded once)
model = None

def load_model():
    global model
    if model is None:
        model = AWSDeploymentModel(num_classes=10)
        model.load_state_dict(torch.load('/opt/ml/model/model.pth', map_location='cpu'))
        model.eval()
    return model

def lambda_handler(event, context):
    try:
        # Load model
        model = load_model()
        
        # Parse input
        if 'body' in event:
            body = json.loads(event['body'])
        else:
            body = event
        
        # Decode image
        image_data = base64.b64decode(body['image'])
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # Preprocess
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        input_tensor = transform(image).unsqueeze(0)
        
        # Inference
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
        
        # Get top predictions
        top_probs, top_indices = torch.topk(probabilities, k=5)
        
        predictions = []
        for i in range(5):
            predictions.append({
                "class_id": top_indices[0][i].item(),
                "probability": top_probs[0][i].item()
            })
        
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'predictions': predictions
            })
        }
        
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e)
            })
        }
'''
        return function_code

# AWS Batch Training Manager
class BatchTrainingManager:
    """Manage AWS Batch training jobs"""
    
    def __init__(self, region_name: str = 'us-east-1'):
        if not AWS_AVAILABLE:
            raise ImportError("boto3 not available")
        
        self.region_name = region_name
        self.batch_client = boto3.client('batch', region_name=region_name)
        self.ecs_client = boto3.client('ecs', region_name=region_name)
    
    def create_training_job_definition(self, job_definition_name: str,
                                     container_image: str,
                                     vcpus: int = 2,
                                     memory: int = 4096) -> str:
        """Create Batch job definition for training"""
        
        try:
            response = self.batch_client.register_job_definition(
                jobDefinitionName=job_definition_name,
                type='container',
                containerProperties={
                    'image': container_image,
                    'vcpus': vcpus,
                    'memory': memory,
                    'jobRoleArn': 'arn:aws:iam::account:role/BatchJobRole',
                    'environment': [
                        {'name': 'AWS_DEFAULT_REGION', 'value': self.region_name}
                    ],
                    'mountPoints': [],
                    'volumes': []
                }
            )
            
            print(f"✓ Job definition created: {job_definition_name}")
            return response['jobDefinitionArn']
        
        except ClientError as e:
            print(f"✗ Failed to create job definition: {e}")
            raise

# Cloud Formation Templates
class CloudFormationTemplates:
    """Generate CloudFormation templates for infrastructure"""
    
    @staticmethod
    def create_sagemaker_infrastructure_template() -> str:
        """Create CloudFormation template for SageMaker infrastructure"""
        
        template = {
            "AWSTemplateFormatVersion": "2010-09-09",
            "Description": "SageMaker infrastructure for PyTorch model deployment",
            "Parameters": {
                "ModelName": {
                    "Type": "String",
                    "Default": "pytorch-model",
                    "Description": "Name for the SageMaker model"
                },
                "InstanceType": {
                    "Type": "String",
                    "Default": "ml.m5.large",
                    "Description": "Instance type for SageMaker endpoint"
                },
                "ExecutionRoleName": {
                    "Type": "String",
                    "Default": "SageMakerExecutionRole",
                    "Description": "Name for SageMaker execution role"
                }
            },
            "Resources": {
                "SageMakerExecutionRole": {
                    "Type": "AWS::IAM::Role",
                    "Properties": {
                        "RoleName": {"Ref": "ExecutionRoleName"},
                        "AssumeRolePolicyDocument": {
                            "Version": "2012-10-17",
                            "Statement": [{
                                "Effect": "Allow",
                                "Principal": {"Service": "sagemaker.amazonaws.com"},
                                "Action": "sts:AssumeRole"
                            }]
                        },
                        "ManagedPolicyArns": [
                            "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
                        ]
                    }
                },
                "S3Bucket": {
                    "Type": "AWS::S3::Bucket",
                    "Properties": {
                        "BucketName": {"Fn::Sub": "${ModelName}-artifacts-${AWS::AccountId}"},
                        "VersioningConfiguration": {"Status": "Enabled"},
                        "PublicAccessBlockConfiguration": {
                            "BlockPublicAcls": True,
                            "BlockPublicPolicy": True,
                            "IgnorePublicAcls": True,
                            "RestrictPublicBuckets": True
                        }
                    }
                }
            },
            "Outputs": {
                "ExecutionRoleArn": {
                    "Description": "ARN of SageMaker execution role",
                    "Value": {"Fn::GetAtt": ["SageMakerExecutionRole", "Arn"]},
                    "Export": {"Name": {"Fn::Sub": "${AWS::StackName}-ExecutionRoleArn"}}
                },
                "S3BucketName": {
                    "Description": "Name of S3 bucket for model artifacts",
                    "Value": {"Ref": "S3Bucket"},
                    "Export": {"Name": {"Fn::Sub": "${AWS::StackName}-S3Bucket"}}
                }
            }
        }
        
        return json.dumps(template, indent=2)

if __name__ == "__main__":
    print("AWS Cloud Deployment")
    print("=" * 25)
    
    if not AWS_AVAILABLE:
        print("AWS SDK (boto3) not available. Install with: pip install boto3")
        print("Also configure AWS credentials using 'aws configure' or environment variables")
        exit(1)
    
    # Create sample model
    model = AWSDeploymentModel(num_classes=10)
    
    print("\n1. SageMaker Deployment")
    print("-" * 26)
    
    try:
        # Initialize SageMaker manager
        sagemaker_manager = SageMakerDeploymentManager(region_name='us-east-1')
        
        print("SageMaker deployment process:")
        print("1. Upload model artifacts to S3")
        print("2. Create SageMaker model")
        print("3. Create endpoint configuration")
        print("4. Create and deploy endpoint")
        print("\nNote: Actual deployment requires AWS credentials and resources")
        
    except Exception as e:
        print(f"SageMaker setup error: {e}")
    
    print("\n2. Lambda Deployment")
    print("-" * 22)
    
    try:
        # Initialize Lambda manager
        lambda_manager = LambdaDeploymentManager(region_name='us-east-1')
        
        # Create deployment package
        package_path = lambda_manager.create_lambda_deployment_package(model, "pytorch_lambda")
        
        print("Lambda deployment package created")
        print("Next steps:")
        print("1. Upload package to Lambda")
        print("2. Configure function settings")
        print("3. Set up API Gateway for HTTP access")
        
    except Exception as e:
        print(f"Lambda setup error: {e}")
    
    print("\n3. CloudFormation Infrastructure")
    print("-" * 36)
    
    # Generate CloudFormation template
    cf_templates = CloudFormationTemplates()
    
    sagemaker_template = cf_templates.create_sagemaker_infrastructure_template()
    
    # Save template
    with open("sagemaker_infrastructure.json", 'w') as f:
        f.write(sagemaker_template)
    
    print("✓ CloudFormation template generated: sagemaker_infrastructure.json")
    
    print("\n4. AWS Deployment Best Practices")
    print("-" * 37)
    
    best_practices = [
        "Use IAM roles with minimal required permissions",
        "Enable CloudTrail for audit logging",
        "Implement proper monitoring with CloudWatch",
        "Use VPC for network isolation",
        "Enable encryption at rest and in transit",
        "Implement auto-scaling for production workloads",
        "Use multiple AZs for high availability",
        "Regular security assessments and updates",
        "Cost optimization with reserved instances",
        "Backup and disaster recovery planning",
        "Use Infrastructure as Code (CloudFormation/CDK)",
        "Implement CI/CD pipelines for deployments"
    ]
    
    print("AWS Deployment Best Practices:")
    for i, practice in enumerate(best_practices, 1):
        print(f"{i:2d}. {practice}")
    
    print("\n5. AWS Service Comparison")
    print("-" * 28)
    
    service_comparison = {
        "SageMaker": {
            "Use case": "Managed ML platform",
            "Pros": "Fully managed, auto-scaling, A/B testing",
            "Cons": "Higher cost, AWS-specific",
            "Best for": "Production ML workloads"
        },
        "Lambda": {
            "Use case": "Serverless inference",
            "Pros": "Pay-per-request, auto-scaling, no server management",
            "Cons": "15-minute timeout, cold starts, size limits",
            "Best for": "Lightweight, infrequent inference"
        },
        "ECS/Fargate": {
            "Use case": "Containerized applications",
            "Pros": "Container orchestration, flexible, cost-effective",
            "Cons": "More management overhead",
            "Best for": "Custom container deployments"
        },
        "EC2": {
            "Use case": "Full control over infrastructure",
            "Pros": "Maximum flexibility, cost control",
            "Cons": "High management overhead",
            "Best for": "Custom requirements, cost optimization"
        }
    }
    
    print("AWS Service Comparison:")
    for service, details in service_comparison.items():
        print(f"\n{service}:")
        for key, value in details.items():
            print(f"  {key.title()}: {value}")
    
    print("\n6. Cost Optimization Strategies")
    print("-" * 34)
    
    cost_strategies = [
        "Use Spot instances for training workloads",
        "Implement auto-scaling to match demand",
        "Choose appropriate instance types for workload",
        "Use reserved instances for predictable workloads",
        "Monitor and optimize data transfer costs",
        "Implement lifecycle policies for S3 storage",
        "Use CloudWatch to monitor resource utilization",
        "Regularly review and cleanup unused resources"
    ]
    
    print("Cost Optimization Strategies:")
    for i, strategy in enumerate(cost_strategies, 1):
        print(f"{i}. {strategy}")
    
    print("\n7. Security Considerations")
    print("-" * 29)
    
    security_considerations = [
        "Use VPC endpoints for secure S3 access",
        "Encrypt data at rest using KMS",
        "Enable VPC Flow Logs for network monitoring",
        "Implement WAF for API protection",
        "Use Secrets Manager for sensitive data",
        "Regular security group audits",
        "Enable GuardDuty for threat detection",
        "Implement least privilege access policies"
    ]
    
    print("Security Considerations:")
    for i, consideration in enumerate(security_considerations, 1):
        print(f"{i}. {consideration}")
    
    print("\n8. Monitoring and Logging")
    print("-" * 27)
    
    monitoring_tools = {
        "CloudWatch": "Metrics, logs, and alarms",
        "X-Ray": "Distributed tracing",
        "CloudTrail": "API call logging",
        "Config": "Configuration change tracking",
        "Systems Manager": "Operational insights",
        "Personal Health Dashboard": "Service health notifications"
    }
    
    print("AWS Monitoring Tools:")
    for tool, description in monitoring_tools.items():
        print(f"  {tool}: {description}")
    
    print("\n9. Deployment Automation")
    print("-" * 27)
    
    deployment_commands = '''
# Deploy using AWS CLI
aws sagemaker create-model --model-name pytorch-model \\
    --primary-container Image=pytorch-image,ModelDataUrl=s3://bucket/model.tar.gz \\
    --execution-role-arn arn:aws:iam::account:role/SageMakerRole

# Deploy using CloudFormation
aws cloudformation create-stack --stack-name pytorch-infrastructure \\
    --template-body file://sagemaker_infrastructure.json \\
    --capabilities CAPABILITY_NAMED_IAM

# Deploy using CDK (TypeScript/Python)
cdk deploy pytorch-infrastructure-stack

# Deploy using SAM (Serverless Application Model)
sam build && sam deploy --guided
'''
    
    print("Deployment Commands:")
    print(deployment_commands)
    
    print("\nAWS cloud deployment demonstration completed!")
    print("Generated files:")
    print("  - sagemaker_infrastructure.json (CloudFormation template)")
    print("  - pytorch_lambda.zip (Lambda deployment package)")
    
    print("\nNext steps for actual deployment:")
    print("1. Configure AWS credentials")
    print("2. Create S3 bucket for model artifacts")
    print("3. Deploy infrastructure using CloudFormation")
    print("4. Upload model and deploy to chosen service")
    print("5. Set up monitoring and alerting")
    print("6. Implement CI/CD pipeline for updates")