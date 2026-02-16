# Use Python 3.12 slim image for smaller size
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies (if needed for your packages)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies in stages to avoid timeouts
# Stage 1: Install large packages separately with retries
RUN pip install --default-timeout=300 --retries 5 --no-cache-dir \
    torch numpy pandas scipy scikit-learn || \
    pip install --default-timeout=300 --retries 5 --no-cache-dir \
    torch numpy pandas scipy scikit-learn

# Stage 2: Install remaining dependencies
RUN pip install --default-timeout=300 --retries 5 --no-cache-dir -r requirements.txt || \
    pip install --default-timeout=300 --retries 5 --no-cache-dir -r requirements.txt

# Pre-download HuggingFace models (bake into image)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('multi-qa-mpnet-base-dot-v1')" || \
    python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('multi-qa-mpnet-base-dot-v1')"

# Copy the entire application
COPY . .

# Create necessary directories
RUN mkdir -p input output

# Expose port 8027
EXPOSE 8027

# Health check
# HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
#     CMD python -c "import requests; requests.get('http://localhost:8027/health')" || exit 1

# Run the FastAPI application
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8027"]

