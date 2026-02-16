# 🚀 AWS App Runner Deployment Guide

**Deploy your FastAPI app with ML models to AWS App Runner**

---

## 📝 Deployment Summary

✅ **Successfully deployed!**

- **Live API**: `https://3rjca2r24m.us-east-1.awsapprunner.com`
- **Platform**: AWS App Runner (fully managed, auto-scaling)
- **CI/CD**: GitHub Actions (automated deployment on push)
- **Cost**: ~$12-15/month for low traffic
- **Features**: HTTPS, health monitoring, CloudWatch logs

### 🔑 Key Learnings & Solutions

During deployment, we solved several critical issues:

1. **Architecture Mismatch** 🏗️
   - **Problem**: Building on Apple Silicon (ARM64) for AWS servers (AMD64)
   - **Solution**: Use `docker buildx build --platform linux/amd64` or GitHub Actions (native AMD64)

2. **Slow Startup** ⏱️
   - **Problem**: ML models downloading at runtime caused 18-minute timeout
   - **Solution**: Pre-download models during Docker build with `RUN python -c "..."`

3. **No Logs Visible** 📋
   - **Problem**: Couldn't see application logs in CloudWatch
   - **Solution**: Enable "Observability" in App Runner service configuration

4. **Environment Variables** 🔧
   - **Problem**: Extra quotes in `.env` file caused HuggingFace model loading errors
   - **Solution**: Remove quotes from values: `emb_model=multi-qa-mpnet-base-dot-v1`

5. **ECR Tag Immutability** 🏷️
   - **Problem**: Couldn't overwrite `latest` tag
   - **Solution**: Use versioned tags (`v1`, `v2`, etc.) or disable tag immutability

---

## 🎯 Quick Start

### For New Deployments
Follow the step-by-step guide below (~30-40 minutes)

### For Updates
```bash
# Just push your code!
git push origin fastapi

# GitHub Actions automatically deploys in ~10-12 minutes
```

---

## 🎯 Console vs CLI - Which Should You Use?

This guide provides **both methods** for each step. Choose based on your preference:

| Method | Best For | Pros | Cons |
|--------|----------|------|------|
| **🖥️ Console (Web UI)** | First-time deployers, learning AWS | Visual feedback, guided process, see all options | Slower for repeated tasks |
| **⌨️ CLI (Terminal)** | Experienced users, automation | Fast, scriptable, repeatable | Need to know exact parameters |

**Our Recommendation**:
- 🆕 **First deployment**: Use **Console** (see what you're doing)
- 🔄 **Future updates**: Use **CLI** or `deploy.sh` script (much faster)

**You can mix and match!** For example:
- Create resources in Console (visual)
- Push Docker images with CLI (copy-paste from Console!)

---

## 📋 What You'll Get

- **Live API URL**: `https://xxxxx.us-east-1.awsapprunner.com`
- **Auto HTTPS**: SSL certificate included
- **Auto-scaling**: Handles traffic spikes automatically
- **Cost**: ~$12-15/month for low traffic
- **No server management**: AWS handles everything

---

## ⚠️ BEFORE YOU START

### Fix Security Issue (5 minutes)
Your Google API key is leaked in `centrix.log`. Fix this NOW:

1. **Delete old keys**:
   - Pinecone: https://app.pinecone.io/ → API Keys → Delete
   - Google: https://console.cloud.google.com/ → Credentials → Delete leaked key

2. **Generate NEW keys**:
   - Get new Pinecone API key
   - Get new Google Gemini API key

3. **Update your `.env`** file with new keys

4. **Verify**: Check that `.env` is in `.gitignore` ✅

---

## 🛠️ Prerequisites (10 minutes)

Install these if you don't have them:

### 1. Docker Desktop
```bash
# Download and install from:
https://www.docker.com/products/docker-desktop/

# Verify installation
docker --version
```

### 2. AWS CLI
```bash
# Mac
brew install awscli

# Windows - download from:
https://aws.amazon.com/cli/

# Verify installation
aws --version
```

### 3. AWS Account Setup
```bash
# Configure AWS credentials
aws configure

# You'll need:
# - AWS Access Key ID (from AWS IAM Console)
# - AWS Secret Access Key
# - Default region: us-east-1
# - Default output format: json
```

---

## 🧪 Step 1: Test Locally (5 minutes)

Make sure everything works before deploying:

```bash
# 1. Build Docker image
docker build -t gns-auto-books .

# 2. Run locally
docker run -p 8027:8027 --env-file .env gns-auto-books

# 3. Test in another terminal
curl http://localhost:8027/health

# Expected response:
# {"status":"healthy","service":"Bank Reconciliation API"}

# 4. Stop the container (Ctrl+C)
```

✅ **If you see the health response, you're ready to deploy!**

---

## ☁️ Step 2: Setup AWS Access (5 minutes)

Since the AWS account was set up by someone else, you need to create YOUR OWN access key.

### Create Your Access Key (Console - SAFE!)

1. **Login to AWS Console**: https://console.aws.amazon.com/
   - Use the username/password your boss shared

2. **Go to Security Credentials**:
   - Click your **username** in top-right corner
   - Select **"Security credentials"**
   - OR go directly to: https://console.aws.amazon.com/iam/home#/security_credentials

3. **Create Access Key**:
   - Scroll to **"Access keys"** section
   - Click **"Create access key"**
   - Select: **"Command Line Interface (CLI)"**
   - Check: ✅ "I understand..."
   - Click **"Next"**

4. **Download Credentials**:
   - (Optional) Description: "Local deployment - [Your Name]"
   - Click **"Create access key"**
   - ⚠️ **Click "Download .csv file"** - SAVE THIS FILE!

5. **Configure AWS CLI**:
   ```bash
   aws configure
   
   # Enter values from your CSV file:
   AWS Access Key ID [None]: AKIA... (from csv)
   AWS Secret Access Key [None]: abc123... (from csv)
   Default region name [None]: us-east-1
   Default output format [None]: json
   ```

6. **Test It Works**:
   ```bash
   aws sts get-caller-identity
   # Should show your account ID
   ```

✅ **Done!** You have your own credentials now.

---

## 🐳 Step 3: Create ECR Repository

ECR = Elastic Container Registry (where you store Docker images)

### Option A: Via AWS Console (Recommended for First Time)

1. **Go to ECR Console**:
   - https://console.aws.amazon.com/ecr/
   - OR search "ECR" in AWS search bar

2. **Create Repository**:
   - Click **"Get Started"** or **"Create repository"**
   
3. **Repository Settings**:
   - **Visibility**: Private (keep it selected)
   - **Repository name**: `gns-auto-books`
   - **Tag immutability**: Disabled (default)
   - **Scan on push**: Disabled (default - optional to enable)
   - **KMS encryption**: Disabled (default is fine)
   
4. **Create**:
   - Click **"Create repository"**
   
5. **Save the URI**:
   - You'll see: `123456789012.dkr.ecr.us-east-1.amazonaws.com/gns-auto-books`
   - ⚠️ **Copy this URI** - you need it next!
> got this 783667412479.dkr.ecr.us-east-1.amazonaws.com/gns-auto-books
✅ **Repository created!**

### Option B: Via AWS CLI (Optional - Faster for Repeat Use)

```bash
# Create ECR repository
aws ecr create-repository \
    --repository-name gns-auto-books \
    --region us-east-1

# Output will show:
# {
#     "repository": {
#         "repositoryUri": "123456789012.dkr.ecr.us-east-1.amazonaws.com/gns-auto-books",
#         ...
#     }
# }

# ⚠️ SAVE the "repositoryUri" from output
```

✅ **Either method works!** Console is easier to see what you're doing.

### 3.2 Push Docker Image to ECR

**Two ways to get the commands:**

#### Option A: Console Provides Commands (Easiest!)

1. **In ECR Console**, click on your repository name: `gns-auto-books`
2. Click **"View push commands"** button (top-right)
3. **Select tab**: "macOS/Linux"
4. **Copy each command** and run them one by one

AWS automatically generates the correct commands for you!

#### Option B: Manual CLI Commands

```bash
# Get your AWS Account ID first
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

# Command 1: Login to ECR
aws ecr get-login-password --region us-east-1 | \
    docker login --username AWS --password-stdin \
    $AWS_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com

# Command 2: Build Docker image
docker build -t gns-auto-books .

# Command 3: Tag the image
docker tag gns-auto-books:latest \
    $AWS_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/gns-auto-books:latest

# Command 4: Push to ECR
docker push $AWS_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/gns-auto-books:latest
```

### 3.3 Run the Commands

```bash
# Stop your running Docker container first (Ctrl+C in that terminal)

# Then run all 4 commands (from Console OR manual above)
# Command 1 - Login to ECR ✅
# Command 2 - Build image ✅  
# Command 3 - Tag image ✅
# Command 4 - Push to ECR (takes 2-5 minutes) ⏳
```

**Push Progress**: You'll see upload progress like:
```
The push refers to repository [123456.dkr.ecr.us-east-1.amazonaws.com/gns-auto-books]
abc123: Pushing [====================>] 500MB/500MB
latest: digest: sha256:abc... size: 1234
```

✅ **Verify**: Refresh ECR repository page → You should see image with tag "latest"

---

## 🏃 Step 4: Create IAM Role for App Runner (3 minutes)

App Runner needs permission to pull your Docker images from ECR.

### Option A: Via AWS Console (Recommended - Visual)

1. **Go to IAM Console**:
   - https://console.aws.amazon.com/iam/
   - OR search "IAM" in AWS search bar

2. **Start Creating Role**:
   - Click **"Roles"** in left sidebar
   - Click **"Create role"** button

3. **Select Trusted Entity**:
   - **Trusted entity type**: AWS service
   - **Use case**: Scroll down and select **"App Runner"**
   - Click **"Next"**

4. **Attach Permissions**:
   - You'll see: `AWSAppRunnerServicePolicyForECRAccess` already selected ✅
   - This gives App Runner permission to pull from ECR
   - Click **"Next"**

5. **Name the Role**:
   - **Role name**: `AppRunnerECRAccessRole`
   - **Description**: "Allows App Runner to access ECR images"
   - Click **"Create role"**

6. **Verify**:
   - You should see: "Role AppRunnerECRAccessRole created"
   - Note the role ARN (optional)

✅ **Done!** You only need to do this once for all App Runner services.

### Option B: Via AWS CLI (Optional - One Command)

```bash
# Create the IAM role
aws iam create-role \
    --role-name AppRunnerECRAccessRole \
    --assume-role-policy-document '{
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "build.apprunner.amazonaws.com"},
            "Action": "sts:AssumeRole"
        }]
    }'

# Attach the ECR access policy
aws iam attach-role-policy \
    --role-name AppRunnerECRAccessRole \
    --policy-arn arn:aws:iam::aws:policy/service-role/AWSAppRunnerServicePolicyForECRAccess

# Verify
aws iam get-role --role-name AppRunnerECRAccessRole
```

✅ **Either method creates the same role!**

---

## 🎯 Step 5: Create App Runner Service (15 minutes)

Now for the main event! This is where your app goes live.

**Recommended**: Use **Console method** for first deployment (visual, easier to understand)

---

### Option A: Via AWS Console (Recommended - Step-by-Step)

**Go to App Runner Console**: https://console.aws.amazon.com/apprunner/

#### 🔹 Page 1: Source and Deployment

1. **Click** "Create service" button

2. **Repository type**: 
   - Select: ⚪ **Container registry**

3. **Provider**: 
   - Select: ⚪ **Amazon ECR**

4. **Container image URI**:
   - Click **"Browse"**
   - Select repository: `gns-auto-books`
   - Select image tag: `latest`
   - Click **"Continue"**

5. **Deployment settings**:
   - Deployment trigger: ⚪ **Manual** (deploy when you tell it to)
   - ECR access role: 
     - Select: ⚪ **Use existing service role**
     - Dropdown: Select `AppRunnerECRAccessRole`

6. **Click "Next"**

---

### 🔹 Page 2: Configure Service

#### Service Settings:
- **Service name**: `gns-auto-books-api`
- **Virtual CPU**: 1 vCPU
- **Memory**: 2 GB
- **Port**: `8027`

#### Environment Variables:
Click **"Add environment variable"** 7 times and enter:

| Key | Value | Notes |
|-----|-------|-------|
| `PINECONE_API_KEY` | `paste your new key` | ⚠️ Use NEW key! |
| `GOOGLE_API_KEY` | `paste your new key` | ⚠️ Use NEW key! |
| `index_name` | `banktransactions` | Exactly as shown |
| `emb_model` | `multi-qa-mpnet-base-dot-v1` | No quotes! |
| `PORT` | `8027` | Exactly as shown |
| `HOST` | `0.0.0.0` | Exactly as shown |
| `CORS_ALLOWED_ORIGINS` | `*` | Allows all origins |

**Important**: 
- ✅ Type values **without quotes**
- ✅ Copy-paste to avoid typos
- ✅ Use your **NEW** API keys (not the leaked ones!)

#### Auto Scaling Configuration:
- **Minimum size**: `1` (keeps 1 instance always ready)
- **Maximum size**: `5` (can scale up to 5 instances)
- **Maximum concurrency**: `100` (requests per instance)

#### Health Check:
- **Protocol**: HTTP
- **Path**: `/health`
- **Interval**: `5` seconds
- **Timeout**: `2` seconds
- **Healthy threshold**: `1` (1 success = healthy)
- **Unhealthy threshold**: `3` (3 failures = unhealthy)

#### Observability:
- **Observability**: ✅ **Enable** (Turn this ON!)
  - This enables CloudWatch logs so you can debug issues
  - Without this, you can't see application logs
  - Required for troubleshooting

#### Security (Optional):
- **Instance role**: None (not needed for now)
- **VPC connector**: None (not needed for public API)
- **AWS X-Ray**: ⚪ Disabled (optional - costs extra)

7. **Click "Next"**

---

### 🔹 Page 3: Review and Create

1. **Review all settings**:
   - Scroll through and verify everything looks correct
   - Pay special attention to:
     - Port: `8027` ✓
     - Environment variables: 7 variables ✓
     - Health check path: `/health` ✓

2. **Create & Deploy**:
   - Click **"Create & deploy"**

3. **Wait for Deployment** ⏳:
   - Status will show: **"Operation in progress"**
   - This takes **5-10 minutes**
   - You'll see progress:
     - ⏳ Provisioning
     - ⏳ Building
     - ⏳ Deploying
     - ✅ Running

☕ **Perfect time for a coffee break!**

---

### 🎉 Deployment Complete!

When status shows: ✅ **"Running"**

1. **Copy Your API URL**:
   - Look for **"Default domain"**: 
   ```
   xxxxxx.us-east-1.awsapprunner.com
   ```
   - ⚠️ **SAVE THIS URL** - This is your live API!

2. **Configuration Tab**:
   - Click "Configuration" to see all settings
   - Can edit environment variables anytime here

---

### Option B: Via AWS CLI (Optional - For Advanced Users)

**Create a configuration file** first:

```bash
# Get your AWS account ID and ECR image URI
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export IMAGE_URI="$AWS_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/gns-auto-books:latest"
export ROLE_ARN=$(aws iam get-role --role-name AppRunnerECRAccessRole --query 'Role.Arn' --output text)

# Create App Runner service
aws apprunner create-service \
    --service-name gns-auto-books-api \
    --region us-east-1 \
    --source-configuration '{
        "ImageRepository": {
            "ImageIdentifier": "'$IMAGE_URI'",
            "ImageRepositoryType": "ECR",
            "ImageConfiguration": {
                "Port": "8027",
                "RuntimeEnvironmentVariables": {
                    "PINECONE_API_KEY": "your-new-pinecone-key",
                    "GOOGLE_API_KEY": "your-new-google-key",
                    "index_name": "banktransactions",
                    "emb_model": "multi-qa-mpnet-base-dot-v1",
                    "PORT": "8027",
                    "HOST": "0.0.0.0",
                    "CORS_ALLOWED_ORIGINS": "*"
                }
            }
        },
        "AutoDeploymentsEnabled": false,
        "AuthenticationConfiguration": {
            "AccessRoleArn": "'$ROLE_ARN'"
        }
    }' \
    --instance-configuration '{
        "Cpu": "1 vCPU",
        "Memory": "2 GB"
    }' \
    --health-check-configuration '{
        "Protocol": "HTTP",
        "Path": "/health",
        "Interval": 5,
        "Timeout": 2,
        "HealthyThreshold": 1,
        "UnhealthyThreshold": 3
    }' \
    --auto-scaling-configuration-arn arn:aws:apprunner:us-east-1:aws:autoscalingconfiguration/DefaultConfiguration

# Get service details and URL
aws apprunner describe-service \
    --service-arn $(aws apprunner list-services --query 'ServiceSummaryList[?ServiceName==`gns-auto-books-api`].ServiceArn' --output text) \
    --query 'Service.ServiceUrl' \
    --output text
```

**Note**: Replace `your-new-pinecone-key` and `your-new-google-key` with actual keys!

---

## 🤖 Step 6: Setup GitHub Actions (Optional but Recommended)

**Why?** Automate deployments - just `git push` and GitHub handles the rest!

### 6.1 Add AWS Secrets to GitHub

1. **Go to your GitHub repository**:
   ```
   https://github.com/YOUR-USERNAME/gns-auto-books/settings/secrets/actions
   ```

2. **Click "New repository secret"**

3. **Add Secret #1**:
   - Name: `AWS_ACCESS_KEY_ID`
   - Secret: Your AWS access key ID (starts with `AKIA...`)
   - Click "Add secret"

4. **Add Secret #2**:
   - Name: `AWS_SECRET_ACCESS_KEY`
   - Secret: Your AWS secret access key
   - Click "Add secret"

### 6.2 Update Workflow Configuration

The workflow file `.github/workflows/deploy.yml` is already configured, but you need to update it with your service ARN:

1. **Get your App Runner Service ARN**:
   ```bash
   aws apprunner list-services --region us-east-1 \
     --query 'ServiceSummaryList[?ServiceName==`gns-auto-books-api`].ServiceArn' \
     --output text
   ```

2. **Update `.github/workflows/deploy.yml`**:
   - Line 13: Update `APP_RUNNER_SERVICE_ARN` with your ARN
   - Line 75: Update API URL with your App Runner URL

### 6.3 Test the Workflow

```bash
# Make a small change
echo "# Updated" >> README.md

# Commit and push
git add .
git commit -m "Test GitHub Actions deployment"
git push origin fastapi

# Watch the deployment
# Go to: https://github.com/YOUR-USERNAME/gns-auto-books/actions
```

✅ **Done!** Now every `git push` automatically deploys to AWS!

---

## ✅ Step 7: Test Your Live Deployment (2 minutes)

Once App Runner status shows ✅ **"Running"**:

### 7.1 Find Your URL

**In App Runner Console**, on the service overview page, you'll see:

```
Default domain: xxxxxx.us-east-1.awsapprunner.com
              👆 This is your live API URL!
```

### 7.2 Test All Endpoints
```bash
# Set your URL (replace with yours)
export API_URL="https://xxxxxx.us-east-1.awsapprunner.com"

# Test 1: Root
curl $API_URL/
# Expected: {"message":"Welcome to Bank Reconcilation"}

# Test 2: Health
curl $API_URL/health
# Expected: {"status":"healthy","service":"Bank Reconciliation API"}

# Test 3: Ping
curl $API_URL/ping
# Expected: {"ping":"pong"}

# Test 4: API Docs (open in browser)
open $API_URL/docs
```

### 7.3 Test from React App
Update your React app:

```javascript
// src/config.js
export const API_BASE_URL = "https://xxxxxx.us-east-1.awsapprunner.com";

// Test the process endpoint
const response = await fetch(`${API_BASE_URL}/api/process`, {
  method: 'POST',
  body: formData
});
```

---

## 🎉 SUCCESS!

Your API is now live at: `https://xxxxxx.us-east-1.awsapprunner.com`

### What You Have:
- ✅ Auto-scaling FastAPI backend
- ✅ HTTPS with SSL certificate
- ✅ Health monitoring
- ✅ CloudWatch logs
- ✅ ~$13/month for low traffic

---

## 📊 Monitor Your App

### View Logs
```bash
# Option 1: AWS Console
# App Runner → Your service → Logs tab

# Option 2: CloudWatch
# CloudWatch → Log Groups → /aws/apprunner/gns-auto-books-api
```

### Check Metrics
```bash
# Go to: App Runner → Your service → Metrics
# Monitor: Requests, Active Instances, 2xx/4xx/5xx responses
```

### Set Cost Alert
```bash
# Get notified if costs exceed $20/month
aws cloudwatch put-metric-alarm \
    --alarm-name "AppRunner-HighCost" \
    --metric-name EstimatedCharges \
    --namespace AWS/Billing \
    --statistic Maximum \
    --period 86400 \
    --evaluation-periods 1 \
    --threshold 20 \
    --comparison-operator GreaterThanThreshold
```

---

## 🔄 Update Your App (Future Deployments)

When you make code changes, you have multiple deployment options:

---

### 🌟 Option A: GitHub Actions (Recommended - Automated!)

**✅ Best for**: Regular deployments, team collaboration, CI/CD

**Setup once, deploy forever!**

#### Initial Setup (One-Time):

1. **Add AWS Secrets to GitHub** (if not done already):
   - Go to: `https://github.com/YOUR-USERNAME/gns-auto-books/settings/secrets/actions`
   - Click **"New repository secret"**
   - Add these two secrets:
     - `AWS_ACCESS_KEY_ID` = Your AWS access key
     - `AWS_SECRET_ACCESS_KEY` = Your AWS secret key

2. **That's it!** The workflow is already configured in `.github/workflows/deploy.yml`

#### Deploy by Pushing Code:

```bash
# Make your code changes, then:
git add .
git commit -m "Your changes description"
git push origin fastapi  # or 'main' branch

# 🎉 GitHub Actions automatically:
# 1. Builds Docker image (on native AMD64 - fast!)
# 2. Pushes to ECR
# 3. Deploys to App Runner
# 4. Takes ~10-12 minutes total
```

**Monitor Progress**:
- Go to: `https://github.com/YOUR-USERNAME/gns-auto-books/actions`
- Click on the latest workflow run
- Watch real-time logs

**Manual Trigger** (optional):
- Go to Actions tab → "Deploy to AWS App Runner"
- Click "Run workflow" → Select branch → "Run workflow"

---

### 🛠️ Option B: Local Script (`deploy.sh`)

**✅ Best for**: Quick local deployments, testing before pushing to GitHub

```bash
# Deploy with 'latest' tag
./deploy.sh

# Or deploy with custom version tag
./deploy.sh v1.0.2

# Takes ~15-20 minutes (slower on Apple Silicon due to cross-compilation)
```

**Note**: On Apple Silicon Macs, local builds are slower due to `linux/amd64` cross-compilation. GitHub Actions is much faster!

---

### 🖥️ Option C: Manual Console + CLI

**✅ Best for**: One-off deployments, troubleshooting

```bash
# 1. Build for AMD64 (required for AWS)
docker buildx build --platform linux/amd64 -t gns-auto-books .

# 2. Get push commands from ECR Console
# ECR → gns-auto-books → "View push commands"
# Copy and run all 4 commands

# 3. Trigger deployment in Console
# App Runner → gns-auto-books-api → Actions → Deploy
```

---

### 📊 Comparison

| Method | Speed | Effort | Best For |
|--------|-------|--------|----------|
| **GitHub Actions** | ⚡ Fast (10-12 min) | 🟢 Just `git push` | Regular deploys, teams |
| **`deploy.sh` script** | 🐌 Slow on M1/M2 (15-20 min) | 🟡 One command | Local testing |
| **Manual Console** | 🐌 Slow (20+ min) | 🔴 Multiple steps | Troubleshooting |

**💡 Recommendation**: Use **GitHub Actions** for all deployments after initial setup!

---

## 🐛 Troubleshooting

### Problem: Build fails locally
**Solution**: Check Docker Desktop is running, verify `.env` file exists

### Problem: "Access Denied" pushing to ECR
**Solution**: Run `aws configure` again, check IAM permissions

### Problem: Health check failing
**Solution**: 
- Verify port is `8027` in App Runner config
- Check CloudWatch logs for errors
- Ensure all environment variables are set

### Problem: "Out of memory" errors
**Solution**: Increase memory to 3-4 GB in service configuration

### Problem: Slow responses
**Solution**: 
- Check Google API key is valid
- Check Pinecone index exists
- Review CloudWatch logs for bottlenecks

---

## 💡 Next Steps

### Optional Enhancements:
1. **Custom Domain**: Add `api.yourdomain.com`
2. **CI/CD**: Auto-deploy on git push (GitHub Actions)
3. **Staging Environment**: Create a test version
4. **API Authentication**: Add JWT tokens
5. **Rate Limiting**: Protect against abuse

---

## 💰 Cost Optimization Tips

**Current setup**: ~$12-15/month

### To Reduce Costs:
1. **Scale to zero** (but slower first request):
   - Set min instances to 0
   - Saves ~$8/month
   - 5-10s cold start delay

2. **Use smaller instance**:
   - 0.5 vCPU, 1GB RAM
   - Saves ~$6/month
   - May be slower for ML models

3. **Different region**: 
   - Some regions are cheaper
   - But check latency to Pinecone/Google APIs

**Current setup is optimal** for your ML workload and low traffic!

---

## 📞 Need Help?

**AWS Documentation**: https://docs.aws.amazon.com/apprunner/

**Common Commands**:
```bash
# List your services
aws apprunner list-services --region us-east-1

# Get service details
aws apprunner describe-service \
    --service-arn <your-arn>

# View recent deployments
aws apprunner list-operations \
    --service-arn <your-arn>
```

---

## 🎓 Summary

You just deployed a production-ready FastAPI app with:
- ✅ ML models (sentence transformers)
- ✅ Auto-scaling infrastructure
- ✅ HTTPS encryption
- ✅ Health monitoring
- ✅ All for ~$13/month

**Your API is ready to use!** 🚀

Share your App Runner URL with your React team and start processing bank transactions!

---

## 📝 Recent Feature: Bank Statement PDF Processing

### What's Included
- ✅ **Complete Pipeline**: PDF → Extract → CSV → AI Categorization
- ✅ **Multi-Bank Support**: Wells Fargo, BMO, Chase, Bank of America
- ✅ **AI Categorization**: 3 prediction models (Pinecone, Gemini LLM, COA matching)
- ✅ **FastAPI Endpoint**: `POST /api/process` - Upload PDF, get categorized transactions

### Key Features
1. **PDF Processing**: PyMuPDF-based transaction extraction
2. **Transaction Parsing**: Automatic detection of dates, descriptions, amounts
3. **AI Categorization**: 
   - Pinecone vector search for semantic similarity
   - Google Gemini LLM for US tax form categories
   - Chart of Accounts (COA) matching
4. **CSV Output**: Both extracted and categorized results saved

### Testing the Feature
```bash
# Test with a bank statement PDF
curl -X POST https://3rjca2r24m.us-east-1.awsapprunner.com/api/process \
  -F "file=@your-bank-statement.pdf"

# Expected: JSON with transactions and categorization
```

---

**Deployment Date**: November 7, 2025

**API URL**: `https://3rjca2r24m.us-east-1.awsapprunner.com`

**AWS Account ID**: `783667412479`

**ECR Repository**: `783667412479.dkr.ecr.us-east-1.amazonaws.com/gns-auto-books`

