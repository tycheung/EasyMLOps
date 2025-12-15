"""
Tests for logging utility decorators and helper functions
Tests log_function_call, log_model_operation, and log_request
"""

import pytest
import logging
from unittest.mock import patch, MagicMock, call
import time

from app.utils.logging import (
    log_function_call,
    log_model_operation,
    log_request,
    LoggerMixin
)


class TestLogFunctionCall:
    """Test log_function_call decorator"""
    
    def test_log_function_call_success(self, caplog):
        """Test decorator logs successful function call"""
        @log_function_call
        def test_function(x, y):
            return x + y
        
        with caplog.at_level(logging.DEBUG):
            result = test_function(2, 3)
        
        assert result == 5
        assert len(caplog.records) >= 2  # Entry and exit logs
        assert any("Calling test_function" in record.message for record in caplog.records)
        assert any("test_function completed" in record.message for record in caplog.records)
    
    def test_log_function_call_with_kwargs(self, caplog):
        """Test decorator logs function call with keyword arguments"""
        @log_function_call
        def test_function(x, y=10):
            return x + y
        
        with caplog.at_level(logging.DEBUG):
            result = test_function(5, y=20)
        
        assert result == 25
        assert any("Calling test_function" in record.message for record in caplog.records)
    
    def test_log_function_call_error(self, caplog):
        """Test decorator logs function errors"""
        @log_function_call
        def test_function():
            raise ValueError("Test error")
        
        with caplog.at_level(logging.ERROR):
            with pytest.raises(ValueError):
                test_function()
        
        assert any("test_function failed" in record.message for record in caplog.records)
        assert any("Test error" in record.message for record in caplog.records)
    
    def test_log_function_call_execution_time(self, caplog):
        """Test decorator logs execution time"""
        @log_function_call
        def slow_function():
            time.sleep(0.1)
            return "done"
        
        with caplog.at_level(logging.DEBUG):
            result = slow_function()
        
        assert result == "done"
        assert any("completed in" in record.message for record in caplog.records)
    
    def test_log_function_call_preserves_function_metadata(self):
        """Test decorator preserves function name and docstring"""
        @log_function_call
        def documented_function():
            """This is a test function"""
            return "test"
        
        assert documented_function.__name__ == "documented_function"
        assert documented_function.__doc__ == "This is a test function"
    
    def test_log_function_call_with_class_method(self, caplog):
        """Test decorator works with class methods"""
        class TestClass:
            @log_function_call
            def instance_method(self, value):
                return value * 2
        
        with caplog.at_level(logging.DEBUG):
            obj = TestClass()
            result = obj.instance_method(5)
        
        assert result == 10
        assert any("Calling instance_method" in record.message for record in caplog.records)


class TestLogModelOperation:
    """Test log_model_operation function"""
    
    def test_log_model_operation_basic(self, caplog):
        """Test logging model operation without user_id"""
        with caplog.at_level(logging.INFO):
            log_model_operation("upload", "model_123")
        
        assert len(caplog.records) == 1
        assert "Model operation: upload" in caplog.records[0].message
        assert caplog.records[0].model_id == "model_123"
    
    def test_log_model_operation_with_user_id(self, caplog):
        """Test logging model operation with user_id"""
        with caplog.at_level(logging.INFO):
            log_model_operation("deploy", "model_456", user_id="user_789")
        
        assert len(caplog.records) == 1
        assert "Model operation: deploy" in caplog.records[0].message
        assert caplog.records[0].model_id == "model_456"
        assert caplog.records[0].user_id == "user_789"
    
    def test_log_model_operation_different_operations(self, caplog):
        """Test logging different model operations"""
        operations = ["upload", "deploy", "delete", "update", "validate"]
        
        with caplog.at_level(logging.INFO):
            for op in operations:
                log_model_operation(op, "model_123")
        
        assert len(caplog.records) == len(operations)
        for i, op in enumerate(operations):
            assert f"Model operation: {op}" in caplog.records[i].message


class TestLogRequest:
    """Test log_request function"""
    
    def test_log_request_basic(self, caplog):
        """Test logging HTTP request without user_id"""
        with caplog.at_level(logging.INFO):
            log_request("req_123", "GET", "/api/v1/models")
        
        assert len(caplog.records) == 1
        assert "GET /api/v1/models" in caplog.records[0].message
        assert caplog.records[0].request_id == "req_123"
    
    def test_log_request_with_user_id(self, caplog):
        """Test logging HTTP request with user_id"""
        with caplog.at_level(logging.INFO):
            log_request("req_456", "POST", "/api/v1/models/upload", user_id="user_789")
        
        assert len(caplog.records) == 1
        assert "POST /api/v1/models/upload" in caplog.records[0].message
        assert caplog.records[0].request_id == "req_456"
        assert caplog.records[0].user_id == "user_789"
    
    def test_log_request_different_methods(self, caplog):
        """Test logging different HTTP methods"""
        methods = ["GET", "POST", "PUT", "DELETE", "PATCH"]
        
        with caplog.at_level(logging.INFO):
            for method in methods:
                log_request(f"req_{method}", method, f"/api/v1/{method.lower()}")
        
        assert len(caplog.records) == len(methods)
        for i, method in enumerate(methods):
            assert f"{method} /api/v1/{method.lower()}" in caplog.records[i].message


class TestLoggerMixin:
    """Test LoggerMixin class"""
    
    def test_logger_mixin_provides_logger(self):
        """Test that LoggerMixin provides a logger"""
        class TestClass(LoggerMixin):
            pass
        
        obj = TestClass()
        logger = obj.logger
        
        assert isinstance(logger, logging.Logger)
        assert logger.name == "TestClass"
    
    def test_logger_mixin_logger_name(self):
        """Test that logger name matches class name"""
        class MyCustomClass(LoggerMixin):
            pass
        
        obj = MyCustomClass()
        assert obj.logger.name == "MyCustomClass"
    
    def test_logger_mixin_inheritance(self):
        """Test LoggerMixin works with inheritance"""
        class BaseClass(LoggerMixin):
            pass
        
        class DerivedClass(BaseClass):
            pass
        
        base_obj = BaseClass()
        derived_obj = DerivedClass()
        
        assert base_obj.logger.name == "BaseClass"
        assert derived_obj.logger.name == "DerivedClass"
    
    def test_logger_mixin_usage(self, caplog):
        """Test using logger from mixin"""
        class ServiceClass(LoggerMixin):
            def do_work(self):
                self.logger.info("Doing work")
                return "done"
        
        with caplog.at_level(logging.INFO):
            service = ServiceClass()
            result = service.do_work()
        
        assert result == "done"
        assert len(caplog.records) == 1
        assert "Doing work" in caplog.records[0].message
        assert caplog.records[0].name == "ServiceClass"


class TestLoggingIntegration:
    """Integration tests for logging utilities"""
    
    def test_combined_logging_workflow(self, caplog):
        """Test combining multiple logging utilities"""
        @log_function_call
        def process_model(model_id, user_id):
            log_model_operation("process", model_id, user_id=user_id)
            log_request("req_123", "POST", f"/api/v1/models/{model_id}/process", user_id=user_id)
            return "processed"
        
        with caplog.at_level(logging.DEBUG):
            result = process_model("model_123", "user_456")
        
        assert result == "processed"
        # Should have logs from all three utilities
        assert len(caplog.records) >= 3

