#!/bin/bash

# MLflow AWS Infrastructure Destruction Script
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
TERRAFORM_DIR="terraform"

echo -e "${RED}🗑️  MLflow Infrastructure Destruction${NC}"
echo "================================================="
echo -e "${YELLOW}⚠️  WARNING: This will destroy ALL MLflow infrastructure!${NC}"
echo -e "${YELLOW}This includes:${NC}"
echo "- ECS Cluster and Service"
echo "- RDS Database (and all data)"
echo "- Load Balancer"
echo "- VPC and networking"
echo "- Secrets Manager secrets"
echo ""

# Confirm destruction
read -p "Are you absolutely sure you want to destroy the infrastructure? (type 'yes' to confirm): " confirm
if [[ ! $confirm == "yes" ]]; then
    echo -e "${GREEN}Operation cancelled.${NC}"
    exit 0
fi

# Navigate to Terraform directory
cd "$(dirname "$0")/$TERRAFORM_DIR"

# Destroy infrastructure
echo -e "${YELLOW}Destroying infrastructure...${NC}"
terraform destroy -auto-approve

echo -e "${GREEN}✅ Infrastructure destroyed successfully!${NC}"
echo ""
echo -e "${YELLOW}Note:${NC} S3 bucket 'fraud-models-test' was not destroyed."
echo "If you want to delete it and all MLflow artifacts, run:"
echo "aws s3 rm s3://fraud-models-test --recursive"
echo "aws s3 rb s3://fraud-models-test"