"""
Core Application Components
Exports app factory and route registration
"""

from app.core.app_factory import create_app
from app.core.routes import register_routes

__all__ = [
    "create_app",
    "register_routes",
]

