"""
Route Registration
Registers all API routes and endpoints with the FastAPI application
"""

import os
import time
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse

from app.config import Settings


def register_routes(app: FastAPI, settings: Settings):
    """Register all routes with the FastAPI application"""
    
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

    # Add schemas router also under /schemas prefix for general schema operations
    app.include_router(
        schemas.router,
        prefix=f"{settings.API_V1_PREFIX}/schemas",
        tags=["schemas"]
    )

    app.include_router(
        deployments.router,
        prefix=f"{settings.API_V1_PREFIX}/deployments",
        tags=["deployments"]
    )

    app.include_router(
        monitoring.router,
        prefix=f"{settings.API_V1_PREFIX}/monitoring",
        tags=["monitoring"]
    )

    app.include_router(
        dynamic.router,
        prefix=f"{settings.API_V1_PREFIX}",
        tags=["dynamic"]
    )

