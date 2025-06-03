"""
Main FastAPI application for EasyMLOps platform
Entry point with middleware, exception handlers, and route configuration
"""

import logging
import os
import webbrowser
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.base import BaseHTTPMiddleware
import time
import uuid
import threading

from app.config import get_settings, create_directories
from app.database import create_tables, check_db_connection, get_db_info, init_db, close_db
from app.utils.logging import setup_logging, get_logger, log_request
from app.routes import models, deployments, dynamic, schemas, monitoring

# Initialize settings and logging
settings = get_settings()
setup_logging()
logger = get_logger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware to log all HTTP requests"""
    
    async def dispatch(self, request: Request, call_next):
        # Generate request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # Log request
        start_time = time.time()
        log_request(request_id, request.method, str(request.url))
        
        # Process request
        try:
            response = await call_next(request)
            process_time = time.time() - start_time
            
            # Log response
            logger.info(
                f"Request {request_id} completed - "
                f"Status: {response.status_code}, "
                f"Time: {process_time:.4f}s"
            )
            
            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Process-Time"] = str(process_time)
            
            return response
            
        except Exception as e:
            process_time = time.time() - start_time
            logger.error(
                f"Request {request_id} failed after {process_time:.4f}s: {str(e)}"
            )
            raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    # Startup
    logger.info("Starting EasyMLOps application...")
    
    # Create necessary directories
    create_directories()
    logger.info("Created necessary directories")
    
    # Check database connection
    if check_db_connection():
        logger.info("Database connection successful")
        # Create tables
        create_tables()
        logger.info("Database tables created/verified")
    else:
        logger.error("Database connection failed - some features may not work")
    
    # Initialize monitoring service
    from app.services.monitoring_service import monitoring_service
    await monitoring_service.start_monitoring_tasks()
    logger.info("Monitoring service started")
    
    # Open browser to the application
    if not settings.DEBUG and settings.HOST == "0.0.0.0":
        try:
            webbrowser.open(f"http://localhost:{settings.PORT}")
            logger.info(f"Browser opened to http://localhost:{settings.PORT}")
        except Exception as e:
            logger.warning(f"Could not open browser automatically: {e}")
    
    logger.info("EasyMLOps application started successfully")
    
    await init_db()
    yield
    
    # Shutdown
    logger.info("Shutting down EasyMLOps application...")
    await close_db()
    logger.info("EasyMLOps application shutdown complete")


# Create FastAPI application
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="ML Operations platform for no-code model deployment",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Exception handlers
@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Handle HTTP exceptions"""
    request_id = getattr(request.state, 'request_id', 'unknown')
    logger.error(f"HTTP {exc.status_code} error in request {request_id}: {exc.detail}")
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "message": exc.detail,
                "status_code": exc.status_code,
                "request_id": request_id,
                "timestamp": time.time()
            }
        }
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle request validation errors"""
    request_id = getattr(request.state, 'request_id', 'unknown')
    logger.error(f"Validation error in request {request_id}: {exc.errors()}")
    
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "message": "Request validation failed",
                "details": exc.errors(),
                "status_code": 422,
                "request_id": request_id,
                "timestamp": time.time()
            }
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    request_id = getattr(request.state, 'request_id', 'unknown')
    logger.error(f"Unhandled exception in request {request_id}: {str(exc)}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "message": "Internal server error",
                "status_code": 500,
                "request_id": request_id,
                "timestamp": time.time()
            }
        }
    )


# Health check endpoints
@app.get("/health", tags=["Health"])
async def health_check():
    """Basic health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": settings.APP_VERSION,
        "environment": "production" if not settings.DEBUG else "development"
    }


@app.get("/health/detailed", tags=["Health"])
async def detailed_health_check():
    """Detailed health check with database and system info"""
    db_info = get_db_info()
    
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": settings.APP_VERSION,
        "environment": "production" if not settings.DEBUG else "development",
        "database": db_info,
        "directories": {
            "models": os.path.exists(settings.MODELS_DIR),
            "bentos": os.path.exists(settings.BENTOS_DIR),
            "static": os.path.exists(settings.STATIC_DIR),
            "logs": os.path.exists("logs")
        }
    }


@app.get("/", response_class=HTMLResponse, tags=["Frontend"])
async def root():
    """Serve the main HTML interface"""
    try:
        with open(os.path.join(settings.STATIC_DIR, "index.html"), "r") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(
            content="""
            <html>
                <head><title>EasyMLOps</title></head>
                <body>
                    <h1>Welcome to EasyMLOps!</h1>
                    <p>The HTML interface is not yet available.</p>
                    <p>Please visit <a href="/docs">/docs</a> for the API documentation.</p>
                </body>
            </html>
            """,
            status_code=200
        )


@app.get("/info", tags=["System"])
async def app_info():
    """Get application information"""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "debug": settings.DEBUG,
        "api_prefix": settings.API_V1_PREFIX,
        "docs_url": "/docs",
        "redoc_url": "/redoc"
    }


# Mount static files
if os.path.exists(settings.STATIC_DIR):
    app.mount("/static", StaticFiles(directory=settings.STATIC_DIR), name="static")

# Include routers
app.include_router(
    models.router,
    prefix=f"{settings.API_V1_PREFIX}/models",
    tags=["models"]
)

app.include_router(
    schemas.router,
    prefix=f"{settings.API_V1_PREFIX}/models",
    tags=["schemas"]
)

app.include_router(
    deployments.router,
    prefix=f"{settings.API_V1_PREFIX}/deployments",
    tags=["deployments"]
)

app.include_router(
    dynamic.router,
    prefix=f"{settings.API_V1_PREFIX}",
    tags=["predictions"]
)

app.include_router(
    monitoring.router,
    prefix=f"{settings.API_V1_PREFIX}/monitoring",
    tags=["monitoring"]
)


def open_browser():
    """Open browser to the application"""
    time.sleep(2)  # Wait for server to start
    webbrowser.open(f"http://localhost:{settings.PORT}/docs")


def main():
    """Main entry point"""
    if settings.DEBUG:
        # Open browser in debug mode
        threading.Thread(target=open_browser, daemon=True).start()
    
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info(f"Server will start on {settings.HOST}:{settings.PORT}")
    
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.RELOAD,
        log_level=settings.LOG_LEVEL.lower(),
        access_log=True
    )


if __name__ == "__main__":
    main() 