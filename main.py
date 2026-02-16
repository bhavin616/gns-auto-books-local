"""
Bank Reconciliation API - FastAPI Backend

This module initializes and configures the FastAPI application responsible for:
- PDF → Text extraction
- Transaction parsing
- CSV generation
- AI-based categorization

Endpoints:
    - POST /api/process : Process PDF and categorize transactions
    - GET  /health      : Health check
    - GET  /ping        : Connectivity test
"""

import os
import sys
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from backend.routes import transaction
from backend.utils.logger import log_message
from backend.services import mongodb
from newversion.routes import loan
from newversion.routes import rule_base, organizer
from newversion.services import rule_base_mongodb

# ---------------------------------------------------------
# Load environment variables
# ---------------------------------------------------------
load_dotenv()


# ---------------------------------------------------------
# Lifespan Event Handlers (Startup/Shutdown)
# ---------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI startup and shutdown events.
    Replaces deprecated @app.on_event() decorators.
    """
    # Startup
    try:
        await mongodb.create_client_indexes()
        await mongodb.create_client_loan_indexes()
        await mongodb.create_client_bank_extract_data_indexes()
        await rule_base_mongodb.create_client_rule_base_indexes()
        await rule_base_mongodb.create_client_bank_statement_data_indexes()
        await rule_base_mongodb.create_statement_run_override_indexes()
        await rule_base_mongodb.create_organizer_categories_indexes()
        log_message("info", "MongoDB indexes initialized successfully")
    except Exception as e:
        error_msg = str(e)
        if "10061" in error_msg or "actively refused" in error_msg.lower():
            log_message("warning", "MongoDB is not running or not accessible. The application will start, but MongoDB features will not be available.")
            log_message("info", "To fix: Start MongoDB service or check MONGODB_URL in .env file")
        else:
            log_message("warning", f"Failed to initialize MongoDB indexes: {e}")
    
    yield  # Application runs here
    
    # Shutdown
    try:
        await mongodb.close_mongodb_connection()
        log_message("info", "MongoDB connection closed")
    except Exception as e:
        # Silently ignore shutdown errors if MongoDB was never connected
        pass


# ---------------------------------------------------------
# FastAPI Initialization
# ---------------------------------------------------------
log_message("info", "Initializing FastAPI application.")

app = FastAPI(
    title="Bank Reconciliation API",
    description="AI-powered bank transaction processing and categorization",
    version="1.0.0",
    root_path="/gns-api",  # Needed when behind an NGINX prefix
    lifespan=lifespan,
)


# ---------------------------------------------------------
# Core Endpoints
# ---------------------------------------------------------
@app.get("/", summary="Welcome message")
def read_root():
    """Return a welcome message to confirm API is running."""
    return {"message": "Welcome to Bank Reconciliation API"}


@app.get("/health", summary="Health check")
def health_check():
    """Health check for monitoring systems and load balancers."""
    return {"status": "healthy", "service": "Bank Reconciliation API"}


@app.get("/ping", summary="Ping test")
def ping():
    """Simple endpoint to verify server responsiveness."""
    return {"ping": "pong"}


# ---------------------------------------------------------
# Route Registration
# ---------------------------------------------------------
log_message("info", "Loading API routers.")
try:
    app.include_router(transaction.router, tags=["Transaction Pipeline"])
    app.include_router(rule_base.router, tags=["Rule Base"])
    app.include_router(loan.router, tags=["Loan"])
    app.include_router(organizer.router, tags=["Organizer"])
except Exception as exc:
    log_message("critical", f"Failed to register router: {exc}")
    sys.exit(1)


# ---------------------------------------------------------
# CORS Configuration
# ---------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# Uvicorn Startup (Development Mode)
# ---------------------------------------------------------
if __name__ == "__main__":
    log_message("info", "Starting Uvicorn server.")

    try:
        host = os.getenv("HOST", "0.0.0.0")
        port = int(os.getenv("PORT", 8027))

        uvicorn.run(app, host=host, port=port)
    except Exception as exc:
        log_message("critical", f"Uvicorn server failed to start: {exc}")
        raise
