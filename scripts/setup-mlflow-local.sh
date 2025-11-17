#!/bin/bash
set -e

echo "Setting up MLflow with local MinIO backend..."

# Copy local environment configuration
echo "Setting up environment file..."
cp .env.mlflow.local .env.mlflow
echo "✓ Environment file configured for local MinIO"

# Create MLflow server directory
mkdir -p mlflow-server

echo ""
echo "🎉 MLflow local setup complete!"
echo ""
echo "To start MLflow server with local MinIO backend:"
echo "  docker-compose -f docker-compose.mlflow.yml --env-file .env.mlflow --profile local up -d"
echo ""
echo "Services will be available at:"
echo "  - MLflow UI: http://localhost:5000"
echo "  - MinIO Console: http://localhost:9001 (minioadmin/minioadmin123)"
echo "  - MinIO API: http://localhost:9000"
echo ""
echo "This setup is perfect for local development and testing!"