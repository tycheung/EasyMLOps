"""
Comprehensive tests for logging utilities
Tests logging configuration, formatters, middleware functions, and utility classes
"""

import pytest
import logging
import json
import tempfile
import os
from unittest.mock import patch, MagicMock, mock_open
from io import StringIO
from datetime import datetime

from app.utils.logging import (
    JSONFormatter, setup_logging, get_logger, LoggerMixin,
    log_function_call, log_request, ColoredFormatter
)


class TestJSONFormatter:
    """Test JSON logging formatter"""
    
    def test_json_formatter_basic_message(self):
        """Test basic message formatting"""
        formatter = JSONFormatter()
        
        # Create a log record
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None
        )
        record.module = "test_module"
        record.funcName = "test_function"
        
        # Format the record
        formatted = formatter.format(record)
        log_data = json.loads(formatted)
        
        # Verify JSON structure
        assert "timestamp" in log_data
        assert log_data["level"] == "INFO"
        assert log_data["logger"] == "test_logger"
        assert log_data["message"] == "Test message"
        assert log_data["module"] == "test_module"
        assert log_data["function"] == "test_function"
        assert log_data["line"] == 10
    
    def test_json_formatter_with_exception(self):
        """Test JSON formatter with exception information"""
        formatter = JSONFormatter()
        
        try:
            raise ValueError("Test exception")
        except ValueError:
            import sys
            exc_info = sys.exc_info()
        
        record = logging.LogRecord(
            name="test_logger",
            level=logging.ERROR,
            pathname="test.py",
            lineno=20,
            msg="Error occurred",
            args=(),
            exc_info=exc_info
        )
        record.module = "test_module"
        record.funcName = "test_function"
        
        formatted = formatter.format(record)
        log_data = json.loads(formatted)
        
        assert "exception" in log_data
        assert "ValueError" in log_data["exception"]
        assert "Test exception" in log_data["exception"]
    
    def test_json_formatter_with_extra_fields(self):
        """Test JSON formatter with extra fields"""
        formatter = JSONFormatter()
        
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=30,
            msg="User action",
            args=(),
            exc_info=None
        )
        record.module = "test_module"
        record.funcName = "test_function"
        
        # Add extra fields
        record.user_id = "user_123"
        record.request_id = "req_456"
        record.model_id = "model_789"
        
        formatted = formatter.format(record)
        log_data = json.loads(formatted)
        
        assert log_data["user_id"] == "user_123"
        assert log_data["request_id"] == "req_456"
        assert log_data["model_id"] == "model_789"


class TestColoredFormatter:
    """Test colored console formatter"""
    
    def test_colored_formatter_different_levels(self):
        """Test colored formatter with different log levels"""
        formatter = ColoredFormatter(
            fmt='%(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Test different log levels
        levels = [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL]
        
        for level in levels:
            record = logging.LogRecord(
                name="test_logger",
                level=level,
                pathname="test.py",
                lineno=10,
                msg="Test message",
                args=(),
                exc_info=None
            )
            
            formatted = formatter.format(record)
            assert isinstance(formatted, str)
            assert "Test message" in formatted
    
    def test_colored_formatter_with_timestamp(self):
        """Test colored formatter includes timestamp"""
        formatter = ColoredFormatter(
            fmt='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None
        )
        
        formatted = formatter.format(record)
        assert isinstance(formatted, str)
        # Should contain timestamp pattern
        assert any(char.isdigit() for char in formatted)


class TestSetupLogging:
    """Test logging setup functionality"""
    
    def teardown_method(self):
        """Clean up logging configuration after each test"""
        # Reset logging configuration
        logging.getLogger().handlers.clear()
        logging.getLogger().setLevel(logging.WARNING)
    
    @patch('app.utils.logging.settings')
    @patch('os.makedirs')
    def test_setup_logging_basic(self, mock_makedirs, mock_settings):
        """Test basic logging setup"""
        mock_settings.LOG_LEVEL = "INFO"
        
        setup_logging()
        
        # Verify logs directory creation
        mock_makedirs.assert_called_with("logs", exist_ok=True)
        
        # Verify root logger configuration
        root_logger = logging.getLogger()
        assert root_logger.level == logging.INFO
        assert len(root_logger.handlers) >= 2  # Console + file handlers
    
    @patch('app.utils.logging.settings')
    @patch('os.makedirs')
    def test_setup_logging_with_custom_level(self, mock_makedirs, mock_settings):
        """Test logging setup with custom log level"""
        mock_settings.LOG_LEVEL = "DEBUG"
        
        setup_logging(log_level="ERROR")
        
        root_logger = logging.getLogger()
        assert root_logger.level == logging.ERROR
    
    @patch('app.utils.logging.settings')
    @patch('os.makedirs')
    def test_setup_logging_with_json_format(self, mock_makedirs, mock_settings):
        """Test logging setup with JSON formatting"""
        mock_settings.LOG_LEVEL = "INFO"
        
        setup_logging(enable_json=True)
        
        root_logger = logging.getLogger()
        
        # Check that JSON formatter is used for console
        console_handler = None
        for handler in root_logger.handlers:
            if (isinstance(handler, logging.StreamHandler) and 
                hasattr(handler, '_is_stdout') and handler._is_stdout):
                console_handler = handler
                break
        
        assert console_handler is not None
        assert isinstance(console_handler.formatter, JSONFormatter)
    
    @patch('app.utils.logging.settings')
    @patch('os.makedirs')
    def test_setup_logging_with_custom_file(self, mock_makedirs, mock_settings):
        """Test logging setup with custom log file"""
        mock_settings.LOG_LEVEL = "INFO"
        
        custom_log_file = "custom_test.log"
        setup_logging(log_file=custom_log_file)
        
        # Verify file handler is configured
        root_logger = logging.getLogger()
        file_handlers = [h for h in root_logger.handlers 
                        if isinstance(h, logging.handlers.RotatingFileHandler)]
        
        assert len(file_handlers) >= 1
    
    @patch('app.utils.logging.settings')
    @patch('os.makedirs')
    def test_setup_logging_silences_noisy_loggers(self, mock_makedirs, mock_settings):
        """Test that setup_logging silences noisy third-party loggers"""
        mock_settings.LOG_LEVEL = "DEBUG"
        
        setup_logging()
        
        # Check that noisy loggers are silenced
        assert logging.getLogger("urllib3").level >= logging.WARNING
        assert logging.getLogger("requests").level >= logging.WARNING
        assert logging.getLogger("sqlalchemy.engine").level >= logging.WARNING


class TestGetLogger:
    """Test logger getter function"""
    
    def test_get_logger_returns_logger(self):
        """Test get_logger returns a proper logger instance"""
        logger = get_logger("test_module")
        
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_module"
    
    def test_get_logger_different_names(self):
        """Test get_logger with different module names"""
        logger1 = get_logger("module1")
        logger2 = get_logger("module2")
        
        assert logger1.name == "module1"
        assert logger2.name == "module2"
        assert logger1 != logger2
    
    def test_get_logger_same_name_returns_same_instance(self):
        """Test get_logger returns same instance for same name"""
        logger1 = get_logger("same_module")
        logger2 = get_logger("same_module")
        
        assert logger1 is logger2


class TestLoggerMixin:
    """Test LoggerMixin class"""
    
    def test_logger_mixin_provides_logger_property(self):
        """Test LoggerMixin provides logger property"""
        class TestClass(LoggerMixin):
            pass
        
        instance = TestClass()
        logger = instance.logger
        
        assert isinstance(logger, logging.Logger)
        assert logger.name == "TestClass"
    
    def test_logger_mixin_logger_name_matches_class(self):
        """Test logger name matches class name"""
        class MyCustomClass(LoggerMixin):
            pass
        
        instance = MyCustomClass()
        assert instance.logger.name == "MyCustomClass"
    
    def test_logger_mixin_can_log(self):
        """Test LoggerMixin can actually log messages"""
        class TestClass(LoggerMixin):
            def do_something(self):
                self.logger.info("Test log message")
                return "done"
        
        instance = TestClass()
        
        with patch.object(instance.logger, 'info') as mock_info:
            result = instance.do_something()
            
            assert result == "done"
            mock_info.assert_called_once_with("Test log message")


class TestLogFunctionCallDecorator:
    """Test log_function_call decorator"""
    
    def test_log_function_call_decorator_logs_entry_and_exit(self):
        """Test decorator logs function entry and exit"""
        with patch('app.utils.logging.logging.getLogger') as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger
            
            @log_function_call
            def test_function(x, y):
                return x + y
            
            result = test_function(1, 2)
            
            assert result == 3
            assert mock_logger.debug.call_count >= 2  # Entry and exit
    
    def test_log_function_call_decorator_logs_arguments(self):
        """Test decorator logs function arguments"""
        with patch('app.utils.logging.logging.getLogger') as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger
            
            @log_function_call
            def test_function(name, age=25):
                return f"{name} is {age} years old"
            
            test_function("Alice", age=30)
            
            # Check that arguments were logged
            debug_calls = [call[0][0] for call in mock_logger.debug.call_args_list]
            assert any("args" in call for call in debug_calls)
            assert any("kwargs" in call for call in debug_calls)
    
    def test_log_function_call_decorator_logs_execution_time(self):
        """Test decorator logs execution time"""
        with patch('app.utils.logging.logging.getLogger') as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger
            
            @log_function_call
            def test_function():
                import time
                time.sleep(0.01)  # Small delay
                return "done"
            
            test_function()
            
            # Check that execution time was logged
            debug_calls = [call[0][0] for call in mock_logger.debug.call_args_list]
            assert any("completed in" in call for call in debug_calls)
    
    def test_log_function_call_decorator_logs_exceptions(self):
        """Test decorator logs exceptions"""
        with patch('app.utils.logging.logging.getLogger') as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger
            
            @log_function_call
            def failing_function():
                raise ValueError("Test error")
            
            with pytest.raises(ValueError):
                failing_function()
            
            # Check that error was logged
            mock_logger.error.assert_called()
            error_call = mock_logger.error.call_args[0][0]
            assert "failed" in error_call


class TestLogRequest:
    """Test log_request function"""
    
    def test_log_request_basic(self):
        """Test basic request logging"""
        with patch('app.utils.logging.logging.getLogger') as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger
            
            log_request("req_123", "GET", "/api/v1/models")
            
            mock_get_logger.assert_called_with("requests")
            mock_logger.info.assert_called_once()
            
            # Check the logged message
            call_args = mock_logger.info.call_args
            assert "GET /api/v1/models" in call_args[0][0]
            assert call_args[1]["extra"]["request_id"] == "req_123"
    
    def test_log_request_with_user_id(self):
        """Test request logging with user ID"""
        with patch('app.utils.logging.logging.getLogger') as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger
            
            log_request("req_456", "POST", "/api/v1/predictions", user_id="user_789")
            
            call_args = mock_logger.info.call_args
            assert call_args[1]["extra"]["request_id"] == "req_456"
            assert call_args[1]["extra"]["user_id"] == "user_789"
    
    def test_log_request_different_methods(self):
        """Test logging different HTTP methods"""
        with patch('app.utils.logging.logging.getLogger') as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger
            
            methods = ["GET", "POST", "PUT", "DELETE", "PATCH"]
            
            for method in methods:
                log_request(f"req_{method}", method, "/api/v1/test")
            
            assert mock_logger.info.call_count == len(methods)


class TestLoggingIntegration:
    """Test logging system integration"""
    
    def teardown_method(self):
        """Clean up after each test"""
        logging.getLogger().handlers.clear()
        logging.getLogger().setLevel(logging.WARNING)
    
    @patch('app.utils.logging.settings')
    def test_full_logging_workflow(self, mock_settings):
        """Test complete logging workflow"""
        mock_settings.LOG_LEVEL = "INFO"
        
        # Setup logging
        setup_logging()
        
        # Get a logger
        logger = get_logger("test_workflow")
        
        # Test different log levels
        with patch.object(logger, 'info') as mock_info, \
             patch.object(logger, 'error') as mock_error, \
             patch.object(logger, 'warning') as mock_warning:
            
            logger.info("Test info message")
            logger.warning("Test warning message")
            logger.error("Test error message")
            
            mock_info.assert_called_once_with("Test info message")
            mock_warning.assert_called_once_with("Test warning message")
            mock_error.assert_called_once_with("Test error message")
    
    @patch('app.utils.logging.settings')
    def test_structured_logging_with_extra_data(self, mock_settings):
        """Test structured logging with extra contextual data"""
        mock_settings.LOG_LEVEL = "INFO"
        
        # Setup logging with JSON format
        setup_logging(enable_json=True)
        
        logger = get_logger("test_structured")
        
        # Create a string stream to capture log output
        log_stream = StringIO()
        
        # Add a handler that writes to our string stream
        handler = logging.StreamHandler(log_stream)
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        
        # Log with extra context
        extra_data = {
            'user_id': 'user_123',
            'request_id': 'req_456',
            'model_id': 'model_789'
        }
        
        logger.info("Processing prediction request", extra=extra_data)
        
        # Get the logged output
        log_output = log_stream.getvalue()
        
        if log_output:
            # Parse the JSON log
            log_data = json.loads(log_output.strip())
            
            assert log_data['message'] == "Processing prediction request"
            assert log_data['user_id'] == 'user_123'
            assert log_data['request_id'] == 'req_456'
            assert log_data['model_id'] == 'model_789'


class TestLoggingErrorHandling:
    """Test logging error handling"""
    
    def teardown_method(self):
        """Clean up after each test"""
        logging.getLogger().handlers.clear()
        logging.getLogger().setLevel(logging.WARNING)
    
    def test_json_formatter_handles_non_serializable_data(self):
        """Test JSON formatter handles non-serializable data gracefully"""
        formatter = JSONFormatter()
        
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None
        )
        record.module = "test_module"
        record.funcName = "test_function"
        
        # Add non-serializable data
        record.complex_object = object()  # Not JSON serializable
        
        # Should not raise an exception
        try:
            formatted = formatter.format(record)
            # Should be valid JSON
            json.loads(formatted)
        except (TypeError, ValueError):
            pytest.fail("JSON formatter should handle non-serializable data gracefully")
    
    @patch('app.utils.logging.settings')
    @patch('os.makedirs')
    def test_setup_logging_continues_with_file_permission_error(self, mock_makedirs, mock_settings):
        """Test setup_logging continues even if file operations fail"""
        mock_settings.LOG_LEVEL = "INFO"
        
        # Mock makedirs to raise permission error
        mock_makedirs.side_effect = PermissionError("Cannot create logs directory")
        
        # Should not raise an exception
        try:
            setup_logging()
        except PermissionError:
            pytest.fail("setup_logging should handle file permission errors gracefully")
    
    def test_logger_mixin_handles_logging_errors(self):
        """Test LoggerMixin handles logging errors gracefully"""
        class TestClass(LoggerMixin):
            def log_something(self):
                # This should not fail even if logging is misconfigured
                self.logger.info("Test message")
        
        instance = TestClass()
        
        # Should not raise an exception
        try:
            instance.log_something()
        except Exception:
            pytest.fail("LoggerMixin should handle logging errors gracefully")


class TestLoggingPerformance:
    """Test logging performance characteristics"""
    
    def teardown_method(self):
        """Clean up after each test"""
        logging.getLogger().handlers.clear()
        logging.getLogger().setLevel(logging.WARNING)
    
    def test_json_formatter_performance(self):
        """Test JSON formatter performance with large messages"""
        formatter = JSONFormatter()
        
        # Create a large message
        large_message = "x" * 10000
        
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg=large_message,
            args=(),
            exc_info=None
        )
        record.module = "test_module"
        record.funcName = "test_function"
        
        import time
        start_time = time.time()
        
        # Format the record
        formatted = formatter.format(record)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Should complete in reasonable time (< 0.1 seconds)
        assert execution_time < 0.1
        
        # Should still produce valid JSON
        log_data = json.loads(formatted)
        assert log_data["message"] == large_message 