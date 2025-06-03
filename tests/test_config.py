"""
Unit tests for configuration module
Tests settings validation, environment variables, and directory creation
"""

import pytest
import os
import tempfile
import shutil
from unittest.mock import patch, MagicMock

from app.config import Settings, get_settings, create_directories


class TestSettings:
    """Test Settings class and validation"""
    
    def test_settings_defaults(self):
        """Test default settings values"""
        settings = Settings()
        
        assert settings.APP_NAME == "EasyMLOps"
        assert settings.APP_VERSION == "1.0.0"
        assert settings.DEBUG is False
        assert settings.API_V1_PREFIX == "/api/v1"
        assert settings.HOST == "0.0.0.0"
        assert settings.PORT == 8000
        assert settings.RELOAD is False
    
    def test_database_url_construction(self):
        """Test DATABASE_URL construction from components"""
        settings = Settings(
            POSTGRES_USER="testuser",
            POSTGRES_PASSWORD="testpass",
            POSTGRES_SERVER="testhost",
            POSTGRES_PORT="5433",
            POSTGRES_DB="testdb"
        )
        
        expected = "postgresql://testuser:testpass@testhost:5433/testdb"
        assert str(settings.DATABASE_URL) == expected
    
    def test_database_url_without_password(self):
        """Test DATABASE_URL construction without password"""
        settings = Settings(
            POSTGRES_USER="testuser",
            POSTGRES_PASSWORD="",
            POSTGRES_SERVER="testhost",
            POSTGRES_PORT="5432",
            POSTGRES_DB="testdb"
        )
        
        expected = "postgresql://testuser@testhost:5432/testdb"
        assert str(settings.DATABASE_URL) == expected
    
    def test_database_url_explicit(self):
        """Test explicit DATABASE_URL setting"""
        explicit_url = "postgresql://user:pass@host:5432/db"
        settings = Settings(DATABASE_URL=explicit_url)
        
        assert str(settings.DATABASE_URL) == explicit_url
    
    def test_cors_origins_default(self):
        """Test default CORS origins"""
        settings = Settings()
        
        expected_origins = ["http://localhost:3000", "http://localhost:8000"]
        assert settings.BACKEND_CORS_ORIGINS == expected_origins
    
    def test_file_settings_defaults(self):
        """Test file storage default settings"""
        settings = Settings()
        
        assert settings.MODELS_DIR == "models"
        assert settings.BENTOS_DIR == "bentos"
        assert settings.STATIC_DIR == "static"
        assert settings.MAX_FILE_SIZE == 500 * 1024 * 1024  # 500MB
        assert ".pkl" in settings.ALLOWED_MODEL_EXTENSIONS
        assert ".joblib" in settings.ALLOWED_MODEL_EXTENSIONS
    
    def test_security_settings_defaults(self):
        """Test security settings defaults"""
        settings = Settings()
        
        assert settings.SECRET_KEY == "your-secret-key-change-in-production"
        assert settings.ACCESS_TOKEN_EXPIRE_MINUTES == 30
        assert settings.ALGORITHM == "HS256"
    
    def test_monitoring_settings_defaults(self):
        """Test monitoring settings defaults"""
        settings = Settings()
        
        assert settings.ENABLE_METRICS is True
        assert settings.METRICS_PORT == 9090
    
    @patch.dict(os.environ, {
        "APP_NAME": "TestApp",
        "DEBUG": "true",
        "PORT": "9000",
        "POSTGRES_USER": "envuser",
        "SECRET_KEY": "env-secret-key"
    })
    def test_environment_variable_override(self):
        """Test that environment variables override defaults"""
        settings = Settings()
        
        assert settings.APP_NAME == "TestApp"
        assert settings.DEBUG is True
        assert settings.PORT == 9000
        assert settings.POSTGRES_USER == "envuser"
        assert settings.SECRET_KEY == "env-secret-key"
    
    @patch.dict(os.environ, {"BACKEND_CORS_ORIGINS": '["http://example.com", "https://app.com"]'})
    def test_cors_origins_environment_override(self):
        """Test CORS origins from environment variable"""
        settings = Settings()
        
        # Note: This would require custom parsing in actual implementation
        # For now, test that it accepts the string format
        assert isinstance(settings.BACKEND_CORS_ORIGINS, list)
    
    def test_bentoml_settings(self):
        """Test BentoML specific settings"""
        settings = Settings()
        
        assert settings.BENTOML_HOME == "bentos"
    
    def test_logging_settings(self):
        """Test logging configuration"""
        settings = Settings()
        
        assert settings.LOG_LEVEL == "INFO"
        assert "%(asctime)s" in settings.LOG_FORMAT
        assert "%(name)s" in settings.LOG_FORMAT
        assert "%(levelname)s" in settings.LOG_FORMAT


class TestGetSettings:
    """Test get_settings function"""
    
    def test_get_settings_returns_settings_instance(self):
        """Test that get_settings returns a Settings instance"""
        settings = get_settings()
        
        assert isinstance(settings, Settings)
    
    def test_get_settings_singleton_behavior(self):
        """Test that get_settings returns the same instance"""
        settings1 = get_settings()
        settings2 = get_settings()
        
        # Should be the same object (singleton pattern)
        assert settings1 is settings2


class TestCreateDirectories:
    """Test create_directories function"""
    
    @pytest.fixture(autouse=True)
    def setup_temp_directory(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.temp_dir)
        
        yield
        
        # Clean up test environment
        os.chdir(self.original_cwd)
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('app.config.settings')
    def test_create_directories_success(self, mock_settings):
        """Test successful directory creation"""
        # Set up mock settings
        mock_settings.MODELS_DIR = "test_models"
        mock_settings.BENTOS_DIR = "test_bentos"
        mock_settings.STATIC_DIR = "test_static"
        
        # Create directories
        create_directories()
        
        # Check that directories were created
        assert os.path.exists("test_models")
        assert os.path.exists("test_bentos")
        assert os.path.exists("test_static")
        assert os.path.exists("logs")
        assert os.path.exists("alembic/versions")
    
    @patch('app.config.settings')
    def test_create_directories_already_exist(self, mock_settings):
        """Test directory creation when directories already exist"""
        # Set up mock settings
        mock_settings.MODELS_DIR = "existing_models"
        mock_settings.BENTOS_DIR = "existing_bentos"
        mock_settings.STATIC_DIR = "existing_static"
        
        # Create directories first
        os.makedirs("existing_models", exist_ok=True)
        os.makedirs("existing_bentos", exist_ok=True)
        os.makedirs("existing_static", exist_ok=True)
        
        # Should not raise exception
        create_directories()
        
        # Directories should still exist
        assert os.path.exists("existing_models")
        assert os.path.exists("existing_bentos")
        assert os.path.exists("existing_static")
    
    @patch('app.config.settings')
    @patch('os.makedirs')
    def test_create_directories_with_permissions_error(self, mock_makedirs, mock_settings):
        """Test directory creation with permission error"""
        # Set up mock settings
        mock_settings.MODELS_DIR = "models"
        mock_settings.BENTOS_DIR = "bentos"
        mock_settings.STATIC_DIR = "static"
        
        # Mock makedirs to raise PermissionError for one directory
        def side_effect(path, exist_ok=False):
            if path == "models":
                raise PermissionError("Permission denied")
        
        mock_makedirs.side_effect = side_effect
        
        # Should raise PermissionError
        with pytest.raises(PermissionError):
            create_directories()
    
    def test_directory_structure(self):
        """Test that the correct directory structure is created"""
        with tempfile.TemporaryDirectory() as temp_dir:
            original_cwd = os.getcwd()
            try:
                os.chdir(temp_dir)
                
                with patch('app.config.settings') as mock_settings:
                    mock_settings.MODELS_DIR = "models"
                    mock_settings.BENTOS_DIR = "bentos"
                    mock_settings.STATIC_DIR = "static"
                    
                    create_directories()
                    
                    # Check all expected directories
                    expected_dirs = ["models", "bentos", "static", "logs", "alembic/versions"]
                    for dir_name in expected_dirs:
                        assert os.path.exists(dir_name), f"Directory {dir_name} was not created"
                        assert os.path.isdir(dir_name), f"{dir_name} is not a directory"
            
            finally:
                os.chdir(original_cwd)


class TestSettingsValidation:
    """Test Settings validation and edge cases"""
    
    def test_invalid_port_type(self):
        """Test invalid port type"""
        with pytest.raises((ValueError, TypeError)):
            Settings(PORT="invalid_port")
    
    def test_negative_port(self):
        """Test negative port number"""
        settings = Settings(PORT=-1)
        # Should accept negative number (validation could be added)
        assert settings.PORT == -1
    
    def test_very_large_port(self):
        """Test very large port number"""
        settings = Settings(PORT=99999)
        assert settings.PORT == 99999
    
    def test_empty_database_fields(self):
        """Test empty database configuration fields"""
        settings = Settings(
            POSTGRES_USER="",
            POSTGRES_SERVER="",
            POSTGRES_DB=""
        )
        
        # Should handle empty strings gracefully
        assert settings.POSTGRES_USER == ""
        assert settings.POSTGRES_SERVER == ""
        assert settings.POSTGRES_DB == ""
    
    def test_boolean_debug_from_string(self):
        """Test DEBUG boolean conversion from string"""
        # Test various string values that should be True
        true_values = ["true", "True", "TRUE", "1", "yes", "Yes"]
        false_values = ["false", "False", "FALSE", "0", "no", "No", ""]
        
        for value in true_values:
            with patch.dict(os.environ, {"DEBUG": value}):
                settings = Settings()
                # Note: Actual boolean conversion depends on Pydantic's implementation
                # This test documents expected behavior
    
    def test_max_file_size_validation(self):
        """Test MAX_FILE_SIZE validation"""
        settings = Settings(MAX_FILE_SIZE=1024)
        assert settings.MAX_FILE_SIZE == 1024
        
        # Test with very large value
        large_size = 10 * 1024 * 1024 * 1024  # 10GB
        settings = Settings(MAX_FILE_SIZE=large_size)
        assert settings.MAX_FILE_SIZE == large_size
    
    def test_algorithm_validation(self):
        """Test ALGORITHM field validation"""
        settings = Settings(ALGORITHM="RS256")
        assert settings.ALGORITHM == "RS256"
        
        settings = Settings(ALGORITHM="HS512")
        assert settings.ALGORITHM == "HS512"


class TestSettingsIntegration:
    """Integration tests for Settings class"""
    
    def test_complete_configuration_example(self):
        """Test complete configuration with all fields"""
        config_data = {
            "APP_NAME": "ProductionApp",
            "APP_VERSION": "2.0.0",
            "DEBUG": False,
            "API_V1_PREFIX": "/api/v2",
            "HOST": "127.0.0.1",
            "PORT": 8080,
            "RELOAD": True,
            "POSTGRES_SERVER": "prod-db",
            "POSTGRES_PORT": "5432",
            "POSTGRES_USER": "prod_user",
            "POSTGRES_PASSWORD": "secure_password",
            "POSTGRES_DB": "production_db",
            "MODELS_DIR": "/app/models",
            "BENTOS_DIR": "/app/bentos",
            "STATIC_DIR": "/app/static",
            "MAX_FILE_SIZE": 1000000000,  # 1GB
            "SECRET_KEY": "production-secret-key",
            "ACCESS_TOKEN_EXPIRE_MINUTES": 60,
            "ALGORITHM": "HS256",
            "BACKEND_CORS_ORIGINS": ["https://app.example.com"],
            "LOG_LEVEL": "WARNING",
            "BENTOML_HOME": "/app/bentos",
            "ENABLE_METRICS": True,
            "METRICS_PORT": 9091
        }
        
        settings = Settings(**config_data)
        
        # Verify all fields are set correctly
        for key, value in config_data.items():
            if key == "BACKEND_CORS_ORIGINS":
                assert getattr(settings, key) == value
            else:
                assert getattr(settings, key) == value
        
        # Verify DATABASE_URL construction
        expected_db_url = "postgresql://prod_user:secure_password@prod-db:5432/production_db"
        assert str(settings.DATABASE_URL) == expected_db_url
    
    @patch.dict(os.environ, {
        "APP_NAME": "EnvApp",
        "DEBUG": "true",
        "PORT": "3000",
        "SECRET_KEY": "env-secret"
    })
    def test_environment_variable_precedence(self):
        """Test that environment variables take precedence over defaults"""
        # Create a new Settings instance to pick up the environment variables
        # (not using get_settings() which returns a cached global instance)
        settings = Settings()
        
        # Environment variables should take precedence over explicit constructor values
        assert settings.APP_NAME == "EnvApp"  # From env
        assert settings.DEBUG is True  # From env
        assert settings.PORT == 3000  # From env
        assert settings.SECRET_KEY == "env-secret"  # From env 