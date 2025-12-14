"""
FastAPI Application Factory
Creates and configures the FastAPI application with middleware and exception handlers
"""

import logging
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.base import BaseHTTPMiddleware

from app.config import Settings


logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging HTTP requests"""
    
    async def dispatch(self, request: Request, call_next):
        """Process request and log details"""
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        start_time = time.time()
        
        try:
            response = await call_next(request)
            process_time = time.time() - start_time
            
            logger.info(
                f"Request {request_id} {request.method} {request.url.path} "
                f"completed in {process_time:.4f}s with status {response.status_code}"
            )
            
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Process-Time"] = str(process_time)
            
            return response
            
        except Exception as e:
            process_time = time.time() - start_time
            logger.error(
                f"Request {request_id} failed after {process_time:.4f}s: {str(e)}"
            )
            raise


def create_lifespan(app_settings: Settings, app_logger: logging.Logger):
    """Create a lifespan context manager with the given settings and logger"""
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Application lifespan events"""
        # Import database functions here after configuration is set
        from app.database import (
            check_async_db_connection,
            get_db_info,
            init_db,
            close_db
        )
        
        # Startup
        if app_logger:
            app_logger.info("Starting EasyMLOps application...")
            app_logger.info(f"Database type: {app_settings.get_db_type()}")
            app_logger.info(f"Mode: {('Demo/Development' if app_settings.is_sqlite() else 'Production')}")
        
        # Check database connection asynchronously
        if await check_async_db_connection():
            if app_logger:
                app_logger.info(f"Async database connection successful ({app_settings.get_db_type()})")
        else:
            if app_logger:
                app_logger.error("Database connection failed - some features may not work")
                if not app_settings.is_sqlite():
                    app_logger.error("Make sure PostgreSQL is running and accessible")
                    app_logger.info("💡 Tip: Use --demo flag for demo mode without PostgreSQL")
        
        # Initialize monitoring service
        try:
            # Check if monitoring is disabled (e.g., during tests)
            disable_monitoring = os.environ.get("DISABLE_MONITORING", "false").lower() == "true"
            
            if not disable_monitoring:
                from app.services.monitoring_service import monitoring_service
                await monitoring_service.start_monitoring_tasks()
                if app_logger:
                    app_logger.info("Monitoring service started")
            else:
                if app_logger:
                    app_logger.info("Monitoring service disabled via DISABLE_MONITORING environment variable")
        except Exception as e:
            if app_logger:
                app_logger.warning(f"Could not start monitoring service: {e}")
        
        # Initialize database
        await init_db()
        
        # Check if we're in a test environment (don't open browser during tests)
        is_testing = (
            "pytest" in sys.modules or 
            "PYTEST_CURRENT_TEST" in os.environ or
            any("test" in arg.lower() for arg in sys.argv)
        )
        
        # Open browser to the application (but not during tests)
        if not is_testing and not app_settings.DEBUG and app_settings.HOST == "0.0.0.0" and not getattr(app_settings, 'no_browser', False):
            try:
                import webbrowser
                webbrowser.open(f"http://localhost:{app_settings.PORT}")
                if app_logger:
                    app_logger.info(f"Browser opened to http://localhost:{app_settings.PORT}")
            except Exception as e:
                if app_logger:
                    app_logger.warning(f"Could not open browser automatically: {e}")
        
        if app_logger:
            app_logger.info("EasyMLOps application started successfully")
            if not is_testing:
                app_logger.info(f"🚀 Server running at http://{app_settings.HOST}:{app_settings.PORT}")
                app_logger.info(f"📖 API Documentation at http://localhost:{app_settings.PORT}/docs")
                if app_settings.is_sqlite():
                    app_logger.info("🎯 Running in demo mode - perfect for development and testing!")
        
        yield
        
        # Shutdown
        if app_logger:
            app_logger.info("Shutting down EasyMLOps application...")
        await close_db()
        if app_logger:
            app_logger.info("EasyMLOps application shutdown complete")
    
    return lifespan


def create_app(app_settings: Settings = None, app_logger: logging.Logger = None) -> FastAPI:
    """Create and configure the FastAPI application"""
    # Use provided settings/logger or fall back to defaults
    _settings = app_settings
    _logger = app_logger or logger
    
    # Ensure we have settings
    if _settings is None:
        from app.config import get_settings
        _settings = get_settings()
    
    # Ensure we have logger 
    if _logger is None:
        from app.utils.logging import get_logger
        _logger = get_logger(__name__)
    
    # Create FastAPI application
    app = FastAPI(
        title=_settings.APP_NAME,
        version=_settings.APP_VERSION,
        description="ML Operations platform for no-code model deployment",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=create_lifespan(_settings, _logger)
    )

    # Add middleware
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(GZipMiddleware, minimum_size=1000)

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_settings.BACKEND_CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Exception handlers
    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        """Handle HTTP exceptions"""
        request_id = getattr(request.state, 'request_id', 'unknown')
        if _logger:
            _logger.error(f"HTTP {exc.status_code} error in request {request_id}: {exc.detail}")
        
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
        if _logger:
            _logger.error(f"Validation error in request {request_id}: {exc.errors()}")
        
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
        if _logger:
            _logger.error(f"Unhandled exception in request {request_id}: {str(exc)}", exc_info=True)
        
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

    # Mount static files
    if os.path.exists(_settings.STATIC_DIR):
        app.mount("/static", StaticFiles(directory=_settings.STATIC_DIR), name="static")

    # Register all routes
    from app.core.routes import register_routes
    register_routes(app, _settings)

    return app

