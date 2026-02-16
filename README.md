# Bank Categorization & Reconciliation API

## Prerequisites

1. **Python**: Ensure Python 3.8+ is installed
2. **Virtual Environment**: Recommended to use a virtual environment

## Setup Instructions

### 1. Clone and Navigate to Project
```bash
git clone <repository-url>
cd Bank_Categorization
```

### 2. Create Virtual Environment
```bash
# Try python first, if not available use python3 or py
python -m venv venv
# OR if python command not found:
python3 -m venv venv
# OR on Windows:
py -m venv venv
```

### 3. Activate Virtual Environment
**Windows:**
```bash
venv\Scripts\activate
```

**macOS/Linux:**
```bash
source venv/bin/activate
```

### 4. Install Dependencies
```bash
# If you're in the virtual environment:
pip install -r requirements.txt

# If pip is not found, try:
python -m pip install -r requirements.txt
# OR
python3 -m pip install -r requirements.txt
# OR on Windows:
py -m pip install -r requirements.txt
```

### 5. Environment Variables Setup ⚠️ **CRITICAL STEP**
Create a `.env` file in the project root with the following variables:

**Create file:** `.env` (in the same directory as `streamlit_app.py`)

```env
# Pinecone Configuration
PINECONE_API_KEY=your_pinecone_api_key_here
index_name=banktransactions

# Google Gemini API
GOOGLE_API_KEY=your_google_api_key_here

# Embedding Model
emb_model=multi-qa-mpnet-base-dot-v1

# Server Configuration
HOST=0.0.0.0
PORT=8027
CORS_ALLOWED_ORIGINS=*
```

**⚠️ IMPORTANT**: Replace `your_pinecone_api_key_here` and `your_google_api_key_here` with your actual API keys!

## Running the Application

### Option 1: FastAPI Backend (Production)
```bash
# Default port (8027)
python main.py

# Custom port
PORT=8027 python main.py

# Or if python command not found:
python3 main.py
# OR on Windows:
py main.py
```

The API will be available at: `http://localhost:8027`

**API Documentation:**
- Swagger UI: `http://localhost:8027/docs`
- ReDoc: `http://localhost:8027/redoc`

### Option 2: Start Streamlit App (Development/Testing)
```bash
streamlit run streamlit_app_new.py
```

The app will be available at: `http://localhost:8501`

### Option 3: Docker Container (Local Testing)

**Build and run the application in Docker (same as production environment):**

#### Step 1: Build Docker Image
```bash
# Build the image (takes 3-5 minutes due to ML dependencies)
docker build -t gns-auto-books:latest .

# Verify image was created
docker images | grep gns-auto-books
```

#### Step 2: Start Container with Environment Variables
```bash
# Make sure .env file exists in current directory
ls -la .env

# Start container with .env file
docker run -d \
  --name gns-auto-books \
  --restart unless-stopped \
  -p 8027:8027 \
  --env-file .env \
  gns-auto-books:latest

# Verify container is running
docker ps | grep gns-auto-books
```

#### Step 3: Verify Environment Variables (Debugging)
```bash
# Check if GOOGLE_API_KEY is loaded in container
docker exec gns-auto-books env | grep GOOGLE_API_KEY

# Check all environment variables
docker exec gns-auto-books env

# View container logs
docker logs gns-auto-books

# Follow logs in real-time
docker logs -f gns-auto-books
```

#### Step 4: Test the API
```bash
# Health check
curl http://localhost:8027/health

# Open API docs in browser
open http://localhost:8027/docs
```

#### Step 5: Stop and Remove Container
```bash
# Stop the container
docker stop gns-auto-books

# Remove the container
docker rm gns-auto-books

# Remove the image (optional)
docker rmi gns-auto-books:latest
```

#### Troubleshooting Docker Issues

**Problem: Container exits immediately**
```bash
# Check logs for error messages
docker logs gns-auto-books

# Check if .env file is properly formatted (no extra spaces/quotes)
cat .env
```

**Problem: API key not found in container**
```bash
# Verify .env exists and has correct format
cat .env | grep GOOGLE_API_KEY
# Should show: GOOGLE_API_KEY=AIza...UCY (NO QUOTES!)

# Restart container with --env-file
docker stop gns-auto-books && docker rm gns-auto-books
docker run -d --name gns-auto-books -p 8027:8027 --env-file .env gns-auto-books:latest
```

**Problem: Port 8027 already in use**
```bash
# Find what's using the port
lsof -i :8027

# Kill the process or use different port
docker run -d --name gns-auto-books -p 8028:8027 --env-file .env gns-auto-books:latest
```

## Usage

1. **Upload PDF**: Use the file uploader to select a bank statement PDF
2. **Extract Transactions**: The app will automatically parse the PDF and extract transactions
3. **Download Excel**: Download the extracted transactions as an Excel file
4. **Categorize**: Click the "Categorize" button to run transaction categorization using AI
   - ✅ **Fast**: Direct function call (no subprocess overhead)
   - ✅ **Reliable**: No timeout issues
   - ✅ **Better Error Handling**: Clear feedback on missing API keys

## Supported Banks

- Wells Fargo (Business Checking & Credit Cards)
- BMO (Business Checking & Credit Cards)
- Chase (Credit Cards)
- Bank of America (Business Banking)
- JPMorgan Chase Bank

## Troubleshooting

### Common Issues:

1. **Python Command Not Found**: 
   - **Windows**: Install Python from Microsoft Store or python.org (ensure "Add to PATH" is checked)
   - **Try different commands**: `python`, `python3`, or `py`
   - **Alternative**: Use `py -m pip install -r requirements.txt` on Windows

2. **Missing API Keys**: 
   - ⚠️ **CRITICAL**: Ensure all required API keys are set in the `.env` file
   - The app will show debugging info if keys are missing
   - Check the "🔍 Debugging Information" section when categorization fails

3. **PDF Upload Fails**: Check that the PDF is a valid bank statement format

4. **Categorization Errors**: 
   - Verify Pinecone and Google API credentials in `.env` file
   - Check network connectivity
   - The app now provides detailed error information

5. **Port Already in Use**: Change the PORT in `.env` or kill existing processes

### Logs
Check `centrix.log` for detailed application logs and error messages.

## Project Structure

```
gns-auto-books/
├── main.py                   # FastAPI backend entry point
├── streamlit_app_new.py      # Streamlit UI (optional)
├── requirements.txt          # Python dependencies
├── backend/                  # Backend services
│   ├── routes/              # API routes
│   │   └── transaction.py   # Transaction endpoints
│   ├── interactors/         # Business logic
│   │   └── transaction.py   # Transaction processing
│   ├── services/            # External services
│   │   └── pinecone.py      # Vector database operations
│   └── utils/               # Utilities
│       ├── embed.py         # HuggingFace embeddings
│       ├── logger.py        # Logging configuration
│       └── schema.py        # Pydantic schemas
├── utils/                   # Additional utilities
│   └── transaction_extraction.py
├── input/                   # PDF upload directory
├── output/                  # Processed files directory
├── Dockerfile               # Docker container definition
├── .dockerignore            # Docker build exclusions
├── deploy.sh                # Automated deployment script
├── DEPLOYMENT.md            # Complete deployment guide
└── .env                     # Environment variables (not in git)
```

## API Endpoints

### Public Endpoints

#### Health Check
```bash
GET /health
GET /ping
```
Response:
```json
{
  "status": "healthy",
  "service": "Bank Reconciliation API"
}
```

#### Process Transactions (Main Endpoint)
```bash
POST /api/process
```
**Description:** Upload a CSV file containing bank transactions to get AI-powered categorization.

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: CSV file with transaction data

**Response:**
```json
{
  "succeeded": true,
  "message": "Successfully processed transaction",
  "data": "✅ Saved categorized CSV: output/filename_categorized.csv",
  "status_code": 200
}
```

**Example (JavaScript/React):**
```javascript
const uploadFile = async (file) => {
  const formData = new FormData();
  formData.append('file', file);
  
  try {
    const response = await fetch('http://localhost:8027/api/process', {
      method: 'POST',
      body: formData,
    });
    
    const data = await response.json();
    console.log(data);
  } catch (error) {
    console.error('Error:', error);
  }
};
```

**Example (cURL):**
```bash
curl -X POST "http://localhost:8027/api/process" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/transactions.csv"
```

### Internal Endpoints

#### Add GL to Vector Database
```bash
POST /api/Add_GL_in_vecdb_pipeline
```
**Description:** Add General Ledger documents to Pinecone vector database for category matching.

---

For interactive API documentation, visit `http://localhost:8027/docs` when the FastAPI server is running.

---

## 🚀 Production Deployment

This API is deployed on **AWS App Runner** with automated CI/CD via GitHub Actions.

### 🎯 Deployment Options

#### Option 1: Automated GitHub Actions (Recommended)
```bash
# Just push your code - GitHub Actions handles the rest!
git add .
git commit -m "Your changes"
git push origin fastapi

# Automatically builds, tests, and deploys in ~10-12 minutes
# Monitor at: https://github.com/YOUR-USERNAME/gns-auto-books/actions
```

#### Option 2: Local Deployment Script
```bash
# Use the automated deployment script
./deploy.sh          # Deploy with 'latest' tag
./deploy.sh v1.0.2   # Deploy with custom tag
```

### 📖 Complete Guide

For full step-by-step setup and deployment instructions, see **[DEPLOYMENT.md](./DEPLOYMENT.md)**

**What's Included**:
- ✅ AWS App Runner setup (fully managed, auto-scaling)
- ✅ GitHub Actions CI/CD pipeline
- ✅ HTTPS with automatic SSL certificates
- ✅ Health monitoring and CloudWatch logs
- ✅ Cost: ~$12-15/month for low traffic
- ✅ Docker containerization with ML models pre-baked

### 🌐 Live API

**Production URL**: `https://3rjca2r24m.us-east-1.awsapprunner.com`

**Test Endpoints**:
```bash
# Health check
curl https://3rjca2r24m.us-east-1.awsapprunner.com/health

# API documentation
open https://3rjca2r24m.us-east-1.awsapprunner.com/docs
```

---

## 🖥️ EC2 Docker Commands (For Comparison)

**Use these exact commands on EC2 to match local setup:**

```bash
# SSH into EC2
ssh -i gns-auto-books-key.pem ubuntu@52.4.98.255

# Navigate to project
cd ~/projects/gns-auto-books

# Check .env file format (IMPORTANT: No quotes around values!)
cat .env | grep GOOGLE_API_KEY
# Should show: GOOGLE_API_KEY=AIza...UCY (NO QUOTES!)

# Stop and remove old container
docker stop gns-auto-books || true
docker rm gns-auto-books || true

# Build image
docker build -t gns-auto-books:latest .

# Start container with .env
docker run -d \
  --name gns-auto-books \
  --restart unless-stopped \
  -p 8027:8027 \
  --env-file .env \
  gns-auto-books:latest

# Verify environment variables are loaded
docker exec gns-auto-books env | grep GOOGLE_API_KEY

# Check logs
docker logs gns-auto-books

# Test API
curl http://localhost:8027/health
```

**Compare local vs EC2:**
```bash
# Local Mac
docker exec gns-auto-books env | grep GOOGLE_API_KEY

# EC2 (via SSH)
ssh -i gns-auto-books-key.pem ubuntu@52.4.98.255 "docker exec gns-auto-books env | grep GOOGLE_API_KEY"

# Both should show the same API key!
```

---


docker logs -f gns-auto-books

ssh -i gns-auto-books-key.pem ubuntu@52.4.98.255


# Stop and remove old container
docker stop gns-auto-books
docker rm gns-auto-books

# Build image
docker build -t gns-auto-books:latest .

# Start container with .env
docker run -d \
  --name gns-auto-books \
  --restart unless-stopped \
  -p 8027:8027 \
  --env-file .env \
  gns-auto-books:latest