#!/bin/bash
set -e

# Configuration
IMAGE_TAG=${1:-latest}  # Use first argument or default to 'latest'
REGION="us-east-1"
SERVICE_NAME="gns-auto-books-api"

echo "🚀 Deploying GnS Auto Books to AWS App Runner..."
echo "📌 Using image tag: $IMAGE_TAG"
echo ""

# Get AWS Account ID
echo "📋 Getting AWS account details..."
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_REPO="$AWS_ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/gns-auto-books"

echo "✓ AWS Account ID: $AWS_ACCOUNT_ID"
echo "✓ ECR Repository: $ECR_REPO"
echo ""

# Build Docker image for AMD64 (AWS App Runner requires x86)
echo "📦 Building Docker image for AMD64 platform..."
echo "⚠️  First build may take 10-15 minutes (downloading ML models for AMD64)..."
echo "💡 Building for linux/amd64 (required for AWS App Runner on Apple Silicon)"
docker buildx build --platform linux/amd64 -t gns-auto-books .
echo "✓ Docker image built successfully for AMD64"
echo ""

# Login to ECR
echo "🔐 Logging into AWS ECR..."
aws ecr get-login-password --region $REGION | \
    docker login --username AWS --password-stdin $ECR_REPO
echo "✓ Logged into ECR"
echo ""

# Tag image
echo "🏷️  Tagging image as: $IMAGE_TAG"
docker tag gns-auto-books:latest $ECR_REPO:$IMAGE_TAG
echo "✓ Image tagged"
echo ""

# Push to ECR
echo "⬆️  Pushing to ECR (this may take 2-5 minutes)..."
docker push $ECR_REPO:$IMAGE_TAG
echo "✓ Image pushed to ECR with tag: $IMAGE_TAG"
echo ""

# Get service ARN
echo "🔍 Finding App Runner service..."
SERVICE_ARN=$(aws apprunner list-services --region $REGION \
    --query "ServiceSummaryList[?ServiceName=='$SERVICE_NAME'].ServiceArn" \
    --output text)

if [ -z "$SERVICE_ARN" ]; then
    echo "⚠️  No App Runner service found with name '$SERVICE_NAME'"
    echo "Please create the service first via AWS Console"
    echo ""
    echo "📌 Image has been pushed to ECR successfully!"
    echo "🔗 Image URI: $ECR_REPO:$IMAGE_TAG"
    echo ""
    echo "Next steps:"
    echo "1. Go to App Runner Console"
    echo "2. Update your service to use image tag: $IMAGE_TAG"
    echo "3. Click 'Deploy'"
    exit 0
fi

echo "✓ Found service: $SERVICE_NAME"
echo ""

# Trigger deployment
echo "🚀 Triggering App Runner deployment..."
aws apprunner start-deployment \
    --service-arn $SERVICE_ARN \
    --region $REGION > /dev/null

echo ""
echo "✅ Deployment started successfully!"
echo "📌 Image tag: $IMAGE_TAG"
echo ""
echo "📊 Monitor progress:"
echo "   - Console: https://console.aws.amazon.com/apprunner/"
echo "   - Or run: aws apprunner describe-service --service-arn $SERVICE_ARN --region $REGION"
echo ""
echo "⏳ Deployment takes ~2-5 minutes"
echo "🎉 Your API will be live soon!"
echo ""
echo "💡 Usage:"
echo "   ./deploy.sh              # Deploys with 'latest' tag (default)"
echo "   ./deploy.sh v2           # Deploys with custom tag 'v2'"
echo "   ./deploy.sh production   # Deploys with custom tag 'production'"
