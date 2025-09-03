#!/bin/bash
set -e

echo "Setting up MLflow with AWS S3 backend..."

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    echo "AWS CLI not found. Installing..."
    pip install awscli
fi

# Check AWS credentials
echo "Checking AWS credentials..."
if ! aws sts get-caller-identity &> /dev/null; then
    echo "❌ AWS credentials not configured. Please run 'aws configure' first."
    exit 1
fi

# Get AWS Account ID
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
echo "✓ AWS Account ID: $AWS_ACCOUNT_ID"

# Create S3 bucket for MLflow artifacts
BUCKET_NAME="fraud-scoring-models-${AWS_ACCOUNT_ID}"
echo "Creating S3 bucket: $BUCKET_NAME"

if aws s3 ls "s3://$BUCKET_NAME" 2>&1 | grep -q 'NoSuchBucket'; then
    echo "Creating bucket..."
    aws s3 mb "s3://$BUCKET_NAME" --region ${AWS_DEFAULT_REGION:-us-west-2}
    
    # Enable versioning
    echo "Enabling versioning..."
    aws s3api put-bucket-versioning \
        --bucket "$BUCKET_NAME" \
        --versioning-configuration Status=Enabled
    
    # Enable encryption
    echo "Enabling encryption..."
    aws s3api put-bucket-encryption \
        --bucket "$BUCKET_NAME" \
        --server-side-encryption-configuration '{
            "Rules": [
                {
                    "ApplyServerSideEncryptionByDefault": {
                        "SSEAlgorithm": "AES256"
                    }
                }
            ]
        }'
    echo "✓ S3 bucket created and configured"
else
    echo "✓ S3 bucket already exists"
fi

# Create MLflow artifacts folder
echo "Creating MLflow artifacts folder..."
aws s3api put-object --bucket "$BUCKET_NAME" --key mlflow-artifacts/ --body /dev/null
echo "✓ MLflow artifacts folder created"

# Copy AWS environment configuration
echo "Setting up environment file..."
cp .env.mlflow.aws .env.mlflow
sed -i "s/your_aws_account_id/$AWS_ACCOUNT_ID/g" .env.mlflow
echo "✓ Environment file configured"

# Create MLflow server directory
mkdir -p mlflow-server

echo ""
echo "🎉 MLflow AWS setup complete!"
echo ""
echo "To start MLflow server with AWS S3 backend:"
echo "  docker-compose -f docker-compose.mlflow.yml --env-file .env.mlflow up -d"
echo ""
echo "MLflow UI will be available at: http://localhost:5000"
echo "S3 bucket: s3://$BUCKET_NAME/mlflow-artifacts"
echo ""
echo "Don't forget to update .env.mlflow with your actual AWS credentials!"