"""
Main FastAPI application for EasyMLOps platform
Entry point with middleware, exception handlers, and route configuration
"""

import logging
import os
import sys
import argparse
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

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="EasyMLOps - ML Operations Platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m app.main                          # Run with PostgreSQL (production)
  python -m app.main --demo                   # Run with SQLite (demo mode)
  python -m app.main --sqlite --db-path demo.db   # Run with custom SQLite file
  python -m app.main --host 127.0.0.1 --port 8080 # Custom host and port
        """
    )
    
    # Mode selection (mutually exclusive)
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--demo", 
        action="store_true", 
        help="Run in demo mode with SQLite database (no PostgreSQL required)"
    )
    mode_group.add_argument(
        "--sqlite", 
        action="store_true", 
        help="Use SQLite database instead of PostgreSQL"
    )
    
    # Database options
    parser.add_argument(
        "--db-path", 
        type=str, 
        default="easymlops.db",
        help="Path to SQLite database file (only used with --sqlite or --demo)"
    )
    
    # Server options
    parser.add_argument(
        "--host", 
        type=str, 
        help="Host to bind the server to (default: from config)"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        help="Port to bind the server to (default: from config)"
    )
    parser.add_argument(
        "--debug", 
        action="store_true", 
        help="Enable debug mode with auto-reload"
    )
    parser.add_argument(
        "--no-browser", 
        action="store_true", 
        help="Don't automatically open browser"
    )
    
    return parser.parse_args()

def configure_database_mode(args):
    """Configure database mode based on arguments - complete separation between SQLite and PostgreSQL"""
    use_sqlite = args.demo or args.sqlite
    
    if use_sqlite:
        # ============ SQLITE/DEMO MODE ============
        # Force SQLite configuration
        os.environ["USE_SQLITE"] = "true"
        os.environ["SQLITE_PATH"] = args.db_path
        os.environ["DATABASE_URL"] = f"sqlite:///{args.db_path}"
        
        # Clear any PostgreSQL configuration
        for pg_var in ["DB_HOST", "DB_PORT", "DB_USER", "DB_PASSWORD", "DB_NAME"]:
            os.environ.pop(pg_var, None)
        
        mode_name = "Demo" if args.demo else "SQLite Development"
        print(f"\n🎯 Configuring {mode_name} Mode")
        print(f"📁 SQLite Database: {args.db_path}")
        print("✅ No external database required")
        if args.demo:
            print("🎮 Perfect for demonstrations and quick testing")
        else:
            print("🔧 Ideal for development and local testing")
        
    else:
        # =========== POSTGRESQL/PRODUCTION MODE ===========
        # Clear SQLite configuration completely
        os.environ.pop("USE_SQLITE", None)
        os.environ.pop("SQLITE_PATH", None)
        
        # Ensure PostgreSQL mode (DATABASE_URL will be constructed from individual vars)
        if "DATABASE_URL" in os.environ and "sqlite" in os.environ["DATABASE_URL"].lower():
            del os.environ["DATABASE_URL"]  # Remove SQLite URL to force PostgreSQL
        
        print(f"\n🏭 Configuring Production Mode")
        print("🐘 PostgreSQL Database Required")
        print("⚠️  Ensure PostgreSQL is installed and running")
        print("🔧 Database connection will be established on startup")
        
    return use_sqlite

def setup_application_config(args, is_demo_mode):
    """Set up application configuration after database mode is configured"""
    # Import here after environment variables are set
    from app.config import get_settings, create_directories, init_sqlite_database
    from app.utils.logging import setup_logging, get_logger
    
    # Initialize settings with the correct database configuration
    settings = get_settings()
    
    # Override server settings from command line
    if args.host:
        settings.HOST = args.host
    if args.port:
        settings.PORT = args.port
    if args.debug:
        settings.DEBUG = True
        settings.RELOAD = True
    
    # Set no_browser flag
    if args.no_browser:
        settings.no_browser = True
    
    # Initialize logging
    setup_logging()
    logger = get_logger(__name__)
    
    # Create directories
    create_directories()
    
    # Initialize SQLite database if in demo mode
    if is_demo_mode:
        init_sqlite_database()
        logger.info(f"SQLite database initialized: {settings.SQLITE_PATH}")
    
    return settings, logger

# Global variables (will be set in main())
settings = None
logger = None

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware to log all HTTP requests"""
    
    async def dispatch(self, request: Request, call_next):
        # Generate request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # Log request
        start_time = time.time()
        if logger:
            logger.debug(f"Request {request_id}: {request.method} {request.url}")
        
        # Process request
        try:
            response = await call_next(request)
            process_time = time.time() - start_time
            
            # Log response
            if logger:
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
            if logger:
                logger.error(
                    f"Request {request_id} failed after {process_time:.4f}s: {str(e)}"
                )
            raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    # Import database functions here after configuration is set
    from app.database import create_tables, check_db_connection, get_db_info, init_db, close_db
    
    # Startup
    if logger:
        logger.info("Starting EasyMLOps application...")
        logger.info(f"Database type: {settings.get_db_type()}")
        logger.info(f"Mode: {'Demo/Development' if settings.is_sqlite() else 'Production'}")
    
    # Check database connection
    if check_db_connection():
        if logger:
            logger.info(f"Database connection successful ({settings.get_db_type()})")
        # Create tables
        create_tables()
        if logger:
            logger.info("Database tables created/verified")
    else:
        if logger:
            logger.error("Database connection failed - some features may not work")
            if not settings.is_sqlite():
                logger.error("Make sure PostgreSQL is running and accessible")
                logger.info("💡 Tip: Use --demo flag for demo mode without PostgreSQL")
    
    # Initialize monitoring service
    try:
        # Check if monitoring is disabled (e.g., during tests)
        disable_monitoring = os.environ.get("DISABLE_MONITORING", "false").lower() == "true"
        
        if not disable_monitoring:
            from app.services.monitoring_service import monitoring_service
            await monitoring_service.start_monitoring_tasks()
            if logger:
                logger.info("Monitoring service started")
        else:
            if logger:
                logger.info("Monitoring service disabled via DISABLE_MONITORING environment variable")
    except Exception as e:
        if logger:
            logger.warning(f"Could not start monitoring service: {e}")
    
    # Check if we're in a test environment (don't open browser during tests)
    is_testing = (
        "pytest" in sys.modules or 
        "PYTEST_CURRENT_TEST" in os.environ or
        any("test" in arg.lower() for arg in sys.argv)
    )
    
    # Open browser to the application (but not during tests)
    if not is_testing and not settings.DEBUG and settings.HOST == "0.0.0.0" and not getattr(settings, 'no_browser', False):
        try:
            webbrowser.open(f"http://localhost:{settings.PORT}")
            if logger:
                logger.info(f"Browser opened to http://localhost:{settings.PORT}")
        except Exception as e:
            if logger:
                logger.warning(f"Could not open browser automatically: {e}")
    
    if logger:
        logger.info("EasyMLOps application started successfully")
        if not is_testing:
            logger.info(f"🚀 Server running at http://{settings.HOST}:{settings.PORT}")
            logger.info(f"📖 API Documentation at http://localhost:{settings.PORT}/docs")
            if settings.is_sqlite():
                logger.info("🎯 Running in demo mode - perfect for development and testing!")
    
    await init_db()
    yield
    
    # Shutdown
    if logger:
        logger.info("Shutting down EasyMLOps application...")
    await close_db()
    if logger:
        logger.info("EasyMLOps application shutdown complete")


# Create FastAPI application (will be configured in main())
app = None

def create_app():
    """Create and configure the FastAPI application"""
    global app
    
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
        if logger:
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
        if logger:
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
        if logger:
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
            "environment": "production" if not settings.DEBUG else "development",
            "database_type": settings.get_db_type(),
            "mode": "demo" if settings.is_sqlite() else "production"
        }

    @app.get("/health/detailed", tags=["Health"])
    async def detailed_health_check():
        """Detailed health check with database and system info"""
        from app.database import get_db_info
        db_info = get_db_info()
        
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "version": settings.APP_VERSION,
            "environment": "production" if not settings.DEBUG else "development",
            "database": db_info,
            "mode": "demo" if settings.is_sqlite() else "production",
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
            mode_info = "demo mode with SQLite" if settings.is_sqlite() else "production mode with PostgreSQL"
            return HTMLResponse(
                content=f"""
                <html>
                    <head>
                        <title>EasyMLOps - {settings.get_db_type()}</title>
                        <style>
                            body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
                            .container {{ max-width: 800px; margin: 0 auto; background: white; padding: 40px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                            .mode-badge {{ background: {'#4CAF50' if settings.is_sqlite() else '#2196F3'}; color: white; padding: 8px 16px; border-radius: 20px; display: inline-block; margin-bottom: 20px; }}
                            .links {{ margin-top: 30px; }}
                            .links a {{ display: inline-block; margin-right: 20px; padding: 10px 20px; background: #007bff; color: white; text-decoration: none; border-radius: 5px; }}
                            .links a:hover {{ background: #0056b3; }}
                        </style>
                    </head>
                    <body>
                        <div class="container">
                            <h1>Welcome to EasyMLOps!</h1>
                            <div class="mode-badge">Running in {mode_info}</div>
                            <p>The ML Operations platform for no-code model deployment.</p>
                            <p>The HTML interface is not yet available, but you can explore the API documentation.</p>
                            <div class="links">
                                <a href="/docs">📖 API Documentation</a>
                                <a href="/health">🔍 Health Check</a>
                                <a href="/health/detailed">📊 Detailed Status</a>
                            </div>
                        </div>
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
            "redoc_url": "/redoc",
            "database_type": settings.get_db_type(),
            "mode": "demo" if settings.is_sqlite() else "production"
        }

    # Mount static files
    if os.path.exists(settings.STATIC_DIR):
        app.mount("/static", StaticFiles(directory=settings.STATIC_DIR), name="static")

    # Include routers
    from app.routes import models, deployments, dynamic, schemas, monitoring
    
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

    return app


def open_browser():
    """Open browser to the application"""
    time.sleep(2)  # Wait for server to start
    webbrowser.open(f"http://localhost:{settings.PORT}/docs")


def main():
    """Main entry point"""
    global settings, logger
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Configure database mode first (before any imports)
    is_demo_mode = configure_database_mode(args)
    
    # Set up application configuration
    settings, logger = setup_application_config(args, is_demo_mode)
    
    # Create the FastAPI app (this was missing!)
    app = create_app()
    
    # Check if we're in a test environment
    is_testing = (
        "pytest" in sys.modules or 
        "PYTEST_CURRENT_TEST" in os.environ or
        any("test" in arg.lower() for arg in sys.argv)
    )
    
    if settings.DEBUG and not is_testing:
        # Open browser in debug mode (unless disabled or testing)
        if not getattr(settings, 'no_browser', False):
            threading.Thread(target=open_browser, daemon=True).start()
    
    # Enhanced startup messaging
    print("\n" + "="*60)
    print(f"🚀 Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    print("="*60)
    
    if settings.is_sqlite():
        print(f"📊 Database Mode: SQLite ({settings.get_db_type()})")
        print(f"📁 Database File: {settings.SQLITE_PATH}")
        if is_demo_mode:
            print("🎯 Running in DEMO mode - perfect for testing!")
        else:
            print("🔧 Running in DEVELOPMENT mode")
        print("✅ No PostgreSQL installation required")
    else:
        print(f"📊 Database Mode: PostgreSQL ({settings.get_db_type()})")
        print("🏭 Running in PRODUCTION mode")
        print("⚠️  Ensure PostgreSQL is running and accessible")
        
    print(f"🌐 Server: http://{settings.HOST}:{settings.PORT}")
    print(f"📖 API Docs: http://{settings.HOST}:{settings.PORT}/docs")
    print(f"🔍 Health Check: http://{settings.HOST}:{settings.PORT}/health")
    print("="*60)
    
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info(f"Database: {settings.get_db_type()}")
    if settings.is_sqlite():
        logger.info(f"SQLite file: {settings.SQLITE_PATH}")
    logger.info(f"Server will start on {settings.HOST}:{settings.PORT}")
    
    # Start the server
    import uvicorn
    uvicorn.run(
        app,  # Pass the actual app object instead of the string
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.RELOAD,
        log_level=settings.LOG_LEVEL.lower(),
        access_log=True
    )


if __name__ == "__main__":
    main() 