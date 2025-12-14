"""
Main FastAPI application entry point for EasyMLOps platform
Handles command-line arguments, configuration, and server startup
"""

import logging
import os
import sys
import argparse
import webbrowser
import threading
import time

from app.core.app_factory import create_app
from app.config import Settings


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


def open_browser(port: int):
    """Open browser to the application"""
    time.sleep(2)  # Wait for server to start
    webbrowser.open(f"http://localhost:{port}/docs")


def main():
    """Main entry point"""
    # Parse command line arguments
    args = parse_arguments()
    
    # Configure database mode first (before any imports)
    is_demo_mode = configure_database_mode(args)
    
    # Set up application configuration
    settings, logger = setup_application_config(args, is_demo_mode)
    
    # Create the FastAPI app with settings and logger
    app = create_app(settings, logger)
    
    # Check if we're in a test environment
    is_testing = (
        "pytest" in sys.modules or 
        "PYTEST_CURRENT_TEST" in os.environ or
        any("test" in arg.lower() for arg in sys.argv)
    )
    
    if settings.DEBUG and not is_testing:
        # Open browser in debug mode (unless disabled or testing)
        if not getattr(settings, 'no_browser', False):
            threading.Thread(target=open_browser, args=(settings.PORT,), daemon=True).start()
    
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
