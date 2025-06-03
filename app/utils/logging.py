"""
Logging configuration and utilities for EasyMLOps
Provides structured logging with different levels and formatters
"""

import logging
import logging.handlers
import os
import sys
from datetime import datetime
from typing import Optional
import json

from app.config import get_settings

settings = get_settings()


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging"""
    
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields if present
        if hasattr(record, 'user_id'):
            log_entry["user_id"] = record.user_id
        if hasattr(record, 'request_id'):
            log_entry["request_id"] = record.request_id
        if hasattr(record, 'model_id'):
            log_entry["model_id"] = record.model_id
            
        return json.dumps(log_entry)


class ColoredFormatter(logging.Formatter):
    """Colored formatter for console output"""
    
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'ENDC': '\033[0m'       # End color
    }
    
    def format(self, record: logging.LogRecord) -> str:
        log_color = self.COLORS.get(record.levelname, '')
        record.levelname = f"{log_color}{record.levelname}{self.COLORS['ENDC']}"
        return super().format(record)


def setup_logging(
    log_level: str = None,
    log_file: Optional[str] = None,
    enable_json: bool = False
) -> None:
    """
    Setup application logging configuration
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
        enable_json: Whether to use JSON formatting
    """
    
    # Determine log level
    if log_level is None:
        log_level = settings.LOG_LEVEL
    
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create logs directory
    os.makedirs("logs", exist_ok=True)
    
    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    
    if enable_json:
        console_formatter = JSONFormatter()
    else:
        console_formatter = ColoredFormatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (if specified or default)
    if log_file is None:
        log_file = f"logs/easymlops_{datetime.now().strftime('%Y%m%d')}.log"
    
    # Rotating file handler
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(numeric_level)
    
    # Always use JSON format for file logs
    file_formatter = JSONFormatter()
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)
    
    # Error file handler (separate file for errors)
    error_handler = logging.handlers.RotatingFileHandler(
        f"logs/easymlops_errors_{datetime.now().strftime('%Y%m%d')}.log",
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(file_formatter)
    root_logger.addHandler(error_handler)
    
    # Silence some noisy loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)    
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
    
    # Log startup message
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized - Level: {log_level}, File: {log_file}")


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the specified name"""
    return logging.getLogger(name)


class LoggerMixin:
    """Mixin class to add logging functionality to any class"""
    
    @property
    def logger(self) -> logging.Logger:
        """Get logger for this class"""
        return logging.getLogger(self.__class__.__name__)


def log_function_call(func):
    """Decorator to log function calls with parameters and execution time"""
    import functools
    import time
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        start_time = time.time()
        
        # Log function entry
        logger.debug(f"Calling {func.__name__} with args: {args}, kwargs: {kwargs}")
        
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.debug(f"{func.__name__} completed in {execution_time:.4f}s")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"{func.__name__} failed after {execution_time:.4f}s: {str(e)}")
            raise
    
    return wrapper


def log_model_operation(operation: str, model_id: str, user_id: Optional[str] = None):
    """Log model-specific operations"""
    logger = logging.getLogger("model_operations")
    extra = {"model_id": model_id}
    if user_id:
        extra["user_id"] = user_id
    
    logger.info(f"Model operation: {operation}", extra=extra)


def log_request(request_id: str, method: str, path: str, user_id: Optional[str] = None):
    """Log HTTP requests"""
    logger = logging.getLogger("requests")
    extra = {"request_id": request_id}
    if user_id:
        extra["user_id"] = user_id
    
    logger.info(f"{method} {path}", extra=extra) 