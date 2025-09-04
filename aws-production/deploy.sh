#!/bin/bash

# MLflow AWS Production Deployment Script
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="fraud-scoring"
AWS_REGION="us-west-2"
TERRAFORM_DIR="terraform"

echo -e "${GREEN}🚀 MLflow Production Deployment on AWS${NC}"
echo "================================================="

# Check prerequisites
echo -e "${YELLOW}Checking prerequisites...${NC}"

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    echo -e "${RED}❌ AWS CLI is not installed. Please install it first.${NC}"
    exit 1
fi

# Check if Terraform is installed
if ! command -v terraform &> /dev/null; then
    echo -e "${RED}❌ Terraform is not installed. Please install it first.${NC}"
    exit 1
fi

# Check AWS credentials
if ! aws sts get-caller-identity &> /dev/null; then
    echo -e "${RED}❌ AWS credentials not configured. Please run 'aws configure'.${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Prerequisites check passed${NC}"

# Navigate to Terraform directory
cd "$(dirname "$0")/$TERRAFORM_DIR"

# Initialize Terraform
echo -e "${YELLOW}Initializing Terraform...${NC}"
terraform init

# Plan deployment
echo -e "${YELLOW}Creating deployment plan...${NC}"
terraform plan \
  -var="aws_region=$AWS_REGION" \
  -var="project_name=$PROJECT_NAME" \
  -out=tfplan

# Confirm deployment
echo -e "${YELLOW}Ready to deploy MLflow infrastructure to AWS.${NC}"
read -p "Do you want to proceed with deployment? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${RED}Deployment cancelled.${NC}"
    exit 1
fi

# Apply Terraform configuration
echo -e "${YELLOW}Deploying infrastructure...${NC}"
terraform apply tfplan

# Get outputs
echo -e "${GREEN}🎉 Deployment completed successfully!${NC}"
echo "================================================="
echo -e "${GREEN}MLflow URL:${NC} $(terraform output -raw mlflow_url)"
echo -e "${GREEN}RDS Endpoint:${NC} $(terraform output -raw rds_endpoint)"
echo -e "${GREEN}ECS Cluster:${NC} $(terraform output -raw ecs_cluster_name)"

echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "1. Wait 5-10 minutes for services to fully start"
echo "2. Access MLflow UI at the URL above"
echo "3. Update your local .env file with the new MLflow URL"
echo "4. Test connection from your local environment"

# Save outputs to file
terraform output -json > ../outputs.json
echo -e "${GREEN}✅ Infrastructure outputs saved to outputs.json${NC}"