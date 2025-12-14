"""
Dynamic routes package
Exports router and DynamicRouteManager
"""

from app.routes.dynamic.dynamic_routes import router
from app.routes.dynamic.route_manager import DynamicRouteManager

__all__ = [
    "router",
    "DynamicRouteManager",
]

