"""
Comprehensive tests for model utility functions
Tests model validation, framework detection, file handling, and metadata extraction
"""

import pytest
import tempfile
import os
import json
import zipfile
import hashlib
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path

from app.utils.model_utils import (
    ModelValidator, ModelFileManager, 
    TENSORFLOW_AVAILABLE, PYTORCH_AVAILABLE
)
from app.schemas.model import ModelFramework, ModelType, ModelValidationResult


@pytest.fixture
def temp_model_file():
    """Create a temporary model file for testing"""
    with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as f:
        # Create a simple sklearn model and save it
        try:
            from sklearn.linear_model import LogisticRegression
            import joblib
            
            model = LogisticRegression()
            # Fit with dummy data to make it a valid model
            import numpy as np
            X = np.array([[1, 2], [3, 4], [5, 6]])
            y = np.array([0, 1, 0])
            model.fit(X, y)
            
            joblib.dump(model, f.name)
        except ImportError:
            # If sklearn not available, create a dummy file
            f.write(b"dummy model data")
        
        yield f.name
        
        # Cleanup
        if os.path.exists(f.name):
            os.unlink(f.name)


@pytest.fixture
def temp_json_file():
    """Create a temporary JSON file for testing"""
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        json.dump({"xgboost": "model_config", "objective": "binary:logistic"}, f)
        f.flush()
        yield f.name
        
        if os.path.exists(f.name):
            os.unlink(f.name)


@pytest.fixture
def temp_zip_file():
    """Create a temporary ZIP file for testing"""
    with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as f:
        with zipfile.ZipFile(f.name, 'w') as zip_file:
            zip_file.writestr('saved_model.pb', b'dummy tensorflow model')
            zip_file.writestr('variables/variables.data-00000-of-00001', b'dummy variables')
        
        yield f.name
        
        if os.path.exists(f.name):
            os.unlink(f.name)


class TestModelValidator:
    """Test ModelValidator class functionality"""
    
    def test_calculate_file_hash(self, temp_model_file):
        """Test file hash calculation"""
        hash_value = ModelValidator.calculate_file_hash(temp_model_file)
        
        assert isinstance(hash_value, str)
        assert len(hash_value) == 64  # SHA-256 produces 64 character hex string
        
        # Verify hash is consistent
        hash_value2 = ModelValidator.calculate_file_hash(temp_model_file)
        assert hash_value == hash_value2
    
    def test_calculate_file_hash_nonexistent_file(self):
        """Test file hash calculation with non-existent file"""
        with pytest.raises(Exception):
            ModelValidator.calculate_file_hash("nonexistent_file.txt")
    
    @patch('app.utils.model_utils.settings')
    def test_validate_file_extension_valid(self, mock_settings):
        """Test file extension validation for valid extensions"""
        mock_settings.ALLOWED_MODEL_EXTENSIONS = ['.pkl', '.joblib', '.h5', '.pt']
        
        assert ModelValidator.validate_file_extension("model.pkl") is True
        assert ModelValidator.validate_file_extension("model.joblib") is True
        assert ModelValidator.validate_file_extension("model.h5") is True
        assert ModelValidator.validate_file_extension("model.pt") is True
    
    @patch('app.utils.model_utils.settings')
    def test_validate_file_extension_invalid(self, mock_settings):
        """Test file extension validation for invalid extensions"""
        mock_settings.ALLOWED_MODEL_EXTENSIONS = ['.pkl', '.joblib']
        
        assert ModelValidator.validate_file_extension("model.txt") is False
        assert ModelValidator.validate_file_extension("model.pdf") is False
        assert ModelValidator.validate_file_extension("model.exe") is False
    
    @patch('app.utils.model_utils.settings')
    def test_validate_file_size_valid(self, mock_settings):
        """Test file size validation for valid sizes"""
        mock_settings.MAX_FILE_SIZE = 1024 * 1024 * 100  # 100MB
        
        assert ModelValidator.validate_file_size(1024) is True
        assert ModelValidator.validate_file_size(1024 * 1024) is True
        assert ModelValidator.validate_file_size(1024 * 1024 * 50) is True
    
    @patch('app.utils.model_utils.settings')
    def test_validate_file_size_invalid(self, mock_settings):
        """Test file size validation for invalid sizes"""
        mock_settings.MAX_FILE_SIZE = 1024 * 1024  # 1MB
        
        assert ModelValidator.validate_file_size(1024 * 1024 * 2) is False
        assert ModelValidator.validate_file_size(1024 * 1024 * 10) is False
    
    def test_detect_framework_from_joblib_file(self, temp_model_file):
        """Test framework detection from joblib file"""
        framework = ModelValidator.detect_framework_from_file(temp_model_file)
        
        # Should detect sklearn or return None if sklearn not available
        assert framework in [ModelFramework.SKLEARN, ModelFramework.CUSTOM, None]
    
    def test_detect_framework_from_json_file(self, temp_json_file):
        """Test framework detection from JSON file"""
        framework = ModelValidator.detect_framework_from_file(temp_json_file)
        
        assert framework == ModelFramework.XGBOOST
    
    def test_detect_framework_from_zip_file(self, temp_zip_file):
        """Test framework detection from ZIP file"""
        framework = ModelValidator.detect_framework_from_file(temp_zip_file)
        
        assert framework == ModelFramework.TENSORFLOW
    
    def test_detect_framework_tensorflow_extensions(self):
        """Test framework detection for TensorFlow file extensions"""
        test_files = ["model.h5", "model.pb", "model.tflite"]
        
        for file_path in test_files:
            with patch('os.path.isdir', return_value=False):
                framework = ModelValidator.detect_framework_from_file(file_path)
                assert framework == ModelFramework.TENSORFLOW
    
    def test_detect_framework_pytorch_extensions(self):
        """Test framework detection for PyTorch file extensions"""
        test_files = ["model.pt", "model.pth"]
        
        for file_path in test_files:
            framework = ModelValidator.detect_framework_from_file(file_path)
            assert framework == ModelFramework.PYTORCH
    
    def test_detect_framework_onnx_extension(self):
        """Test framework detection for ONNX file extension"""
        framework = ModelValidator.detect_framework_from_file("model.onnx")
        assert framework == ModelFramework.ONNX
    
    def test_detect_framework_unknown_extension(self):
        """Test framework detection for unknown file extension"""
        framework = ModelValidator.detect_framework_from_file("model.unknown")
        assert framework is None
    
    @patch('app.utils.model_utils.joblib.load')
    def test_detect_pickle_framework_sklearn(self, mock_joblib_load):
        """Test detecting sklearn from pickle file"""
        # Mock sklearn model
        mock_model = MagicMock()
        mock_model.__module__ = 'sklearn.linear_model'
        mock_joblib_load.return_value = mock_model
        
        framework = ModelValidator._detect_pickle_framework("dummy_path")
        assert framework == ModelFramework.SKLEARN
    
    @patch('app.utils.model_utils.joblib.load')
    def test_detect_pickle_framework_xgboost(self, mock_joblib_load):
        """Test detecting XGBoost from pickle file"""
        # Mock XGBoost model
        mock_model = MagicMock()
        mock_model.__module__ = 'xgboost.sklearn'
        mock_joblib_load.return_value = mock_model
        
        framework = ModelValidator._detect_pickle_framework("dummy_path")
        assert framework == ModelFramework.XGBOOST
    
    @patch('app.utils.model_utils.joblib.load')
    def test_detect_pickle_framework_custom(self, mock_joblib_load):
        """Test detecting custom framework from pickle file"""
        # Mock unknown model
        mock_model = MagicMock()
        mock_model.__module__ = 'custom.model'
        mock_joblib_load.return_value = mock_model
        
        framework = ModelValidator._detect_pickle_framework("dummy_path")
        assert framework == ModelFramework.CUSTOM
    
    def test_detect_json_framework_patterns(self):
        """Test JSON framework detection with different patterns"""
        # Create temporary JSON files with different patterns
        test_cases = [
            ({"xgboost": "config"}, ModelFramework.XGBOOST),
            ({"lightgbm": "config"}, ModelFramework.LIGHTGBM),
            ({"h2o": "config"}, ModelFramework.H2O),
            ({"tensorflow": "config"}, ModelFramework.TENSORFLOW),
            ({"pytorch": "config"}, ModelFramework.PYTORCH),
            ({"custom": "config"}, ModelFramework.CUSTOM)
        ]
        
        for data, expected_framework in test_cases:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(data, f)
                f.flush()
                
                framework = ModelValidator._detect_json_framework(f.name)
                assert framework == expected_framework
                
                os.unlink(f.name)
    
    @patch('app.utils.model_utils.joblib.load')
    def test_detect_model_type_sklearn_classification(self, mock_joblib_load):
        """Test detecting sklearn classification model type"""
        # Mock sklearn classifier
        mock_model = MagicMock()
        mock_model.__class__.__name__ = 'LogisticRegression'
        mock_joblib_load.return_value = mock_model
        
        model_type = ModelValidator.detect_model_type("dummy_path", ModelFramework.SKLEARN)
        assert model_type == ModelType.CLASSIFICATION
    
    @patch('app.utils.model_utils.joblib.load')
    def test_detect_model_type_sklearn_regression(self, mock_joblib_load):
        """Test detecting sklearn regression model type"""
        # Mock sklearn regressor
        mock_model = MagicMock()
        mock_model.__class__.__name__ = 'LinearRegression'
        mock_joblib_load.return_value = mock_model
        
        model_type = ModelValidator.detect_model_type("dummy_path", ModelFramework.SKLEARN)
        assert model_type == ModelType.REGRESSION
    
    @patch('app.utils.model_utils.joblib.load')
    def test_detect_model_type_sklearn_clustering(self, mock_joblib_load):
        """Test detecting sklearn clustering model type"""
        # Mock sklearn clusterer
        mock_model = MagicMock()
        mock_model.__class__.__name__ = 'KMeans'
        mock_joblib_load.return_value = mock_model
        
        model_type = ModelValidator.detect_model_type("dummy_path", ModelFramework.SKLEARN)
        assert model_type == ModelType.CLUSTERING
    
    def test_validate_model_file_comprehensive(self, temp_model_file):
        """Test comprehensive model file validation"""
        result = ModelValidator.validate_model_file(temp_model_file)
        
        assert isinstance(result, ModelValidationResult)
        assert hasattr(result, 'is_valid')
        assert hasattr(result, 'framework_detected')
        assert hasattr(result, 'model_type_detected')
        assert hasattr(result, 'errors')
        assert hasattr(result, 'warnings')
        assert hasattr(result, 'metadata')
    
    def test_validate_model_file_nonexistent(self):
        """Test validation of non-existent file"""
        result = ModelValidator.validate_model_file("nonexistent_file.pkl")
        
        assert result.is_valid is False
        assert len(result.errors) > 0
        assert any("not found" in error.lower() for error in result.errors)


class TestModelFileManager:
    """Test ModelFileManager class functionality"""
    
    @patch('app.utils.model_utils.settings')
    def test_get_model_storage_path(self, mock_settings):
        """Test getting model storage path"""
        mock_settings.MODELS_DIR = "/tmp/models"
        
        storage_path = ModelFileManager.get_model_storage_path("abc123", "model.pkl")
        
        assert "/tmp/models" in storage_path
        assert "ab" in storage_path  # Subdirectory based on first 2 chars
        assert "abc123_model.pkl" in storage_path
    
    @patch('app.utils.model_utils.settings')
    @patch('os.makedirs')
    @patch('builtins.open', new_callable=mock_open)
    def test_save_uploaded_file(self, mock_file_open, mock_makedirs, mock_settings):
        """Test saving uploaded file content"""
        mock_settings.MODELS_DIR = "/tmp/models"
        
        file_content = b"model data content"
        storage_path = ModelFileManager.save_uploaded_file(file_content, "abc123", "model.pkl")
        
        assert "/tmp/models" in storage_path
        assert "abc123_model.pkl" in storage_path
        mock_file_open.assert_called_once()
        mock_makedirs.assert_called_once()
    
    @patch('app.utils.model_utils.settings')
    @patch('os.makedirs')
    @patch('shutil.copytree')
    @patch('os.path.exists', return_value=False)
    def test_save_directory_model(self, mock_exists, mock_copytree, mock_makedirs, mock_settings):
        """Test saving directory-based model"""
        mock_settings.MODELS_DIR = "/tmp/models"
        
        storage_path = ModelFileManager.save_directory_model("/source/dir", "abc123", "savedmodel")
        
        assert "/tmp/models" in storage_path
        assert "abc123_savedmodel" in storage_path
        mock_copytree.assert_called_once()
    
    @patch('os.path.exists', return_value=True)
    @patch('os.path.isdir', return_value=False)
    @patch('os.remove')
    def test_delete_model_file(self, mock_remove, mock_isdir, mock_exists):
        """Test deleting model file"""
        result = ModelFileManager.delete_model_file("/path/to/model.pkl")
        
        assert result is True
        mock_remove.assert_called_once_with("/path/to/model.pkl")
    
    @patch('os.path.exists', return_value=True)
    @patch('os.path.isdir', return_value=True)
    @patch('shutil.rmtree')
    def test_delete_model_directory(self, mock_rmtree, mock_isdir, mock_exists):
        """Test deleting model directory"""
        result = ModelFileManager.delete_model_file("/path/to/model_dir")
        
        assert result is True
        mock_rmtree.assert_called_once_with("/path/to/model_dir")
    
    @patch('os.path.exists', return_value=False)
    def test_delete_nonexistent_file(self, mock_exists):
        """Test deleting non-existent file"""
        result = ModelFileManager.delete_model_file("/path/to/nonexistent.pkl")
        
        assert result is False
    
    @patch('os.path.exists', return_value=True)
    @patch('os.path.isdir', return_value=False)
    @patch('os.stat')
    def test_get_file_info_file(self, mock_stat, mock_isdir, mock_exists):
        """Test getting file information for regular file"""
        # Mock file stats
        mock_stat_result = MagicMock()
        mock_stat_result.st_size = 1024
        mock_stat_result.st_ctime = 1234567890
        mock_stat_result.st_mtime = 1234567890
        mock_stat.return_value = mock_stat_result
        
        file_info = ModelFileManager.get_file_info("/path/to/model.pkl")
        
        assert file_info['exists'] is True
        assert file_info['is_directory'] is False
        assert file_info['size'] == 1024
        assert 'created' in file_info
        assert 'modified' in file_info
    
    @patch('os.path.exists', return_value=True)
    @patch('os.path.isdir', return_value=True)
    @patch('os.walk')
    @patch('os.stat')
    @patch('os.path.getsize')
    def test_get_file_info_directory(self, mock_getsize, mock_stat, mock_walk, mock_isdir, mock_exists):
        """Test getting file information for directory"""
        # Mock directory walk
        mock_walk.return_value = [
            ('/dir', [], ['file1.txt', 'file2.txt'])
        ]
        mock_getsize.side_effect = [100, 200]  # File sizes
        
        # Mock directory stats
        mock_stat_result = MagicMock()
        mock_stat_result.st_ctime = 1234567890
        mock_stat_result.st_mtime = 1234567890
        mock_stat.return_value = mock_stat_result
        
        file_info = ModelFileManager.get_file_info("/path/to/model_dir")
        
        assert file_info['exists'] is True
        assert file_info['is_directory'] is True
        assert file_info['size'] == 300  # Sum of file sizes
    
    @patch('os.path.exists', return_value=False)
    def test_get_file_info_nonexistent(self, mock_exists):
        """Test getting file information for non-existent file"""
        file_info = ModelFileManager.get_file_info("/path/to/nonexistent.pkl")
        
        assert file_info['exists'] is False


class TestFrameworkAvailability:
    """Test framework availability flags"""
    
    def test_tensorflow_availability_flag(self):
        """Test TensorFlow availability flag"""
        assert isinstance(TENSORFLOW_AVAILABLE, bool)
    
    def test_pytorch_availability_flag(self):
        """Test PyTorch availability flag"""
        assert isinstance(PYTORCH_AVAILABLE, bool)


class TestModelValidationIntegration:
    """Integration tests for model validation workflow"""
    
    def test_complete_validation_workflow(self, temp_model_file):
        """Test complete model validation workflow"""
        # Calculate hash
        file_hash = ModelValidator.calculate_file_hash(temp_model_file)
        assert isinstance(file_hash, str)
        
        # Validate extension
        is_valid_ext = ModelValidator.validate_file_extension(temp_model_file)
        
        # Get file size
        file_size = os.path.getsize(temp_model_file)
        
        # Validate size (assuming reasonable limits)
        with patch('app.utils.model_utils.settings') as mock_settings:
            mock_settings.MAX_FILE_SIZE = file_size * 2  # Set limit above actual size
            is_valid_size = ModelValidator.validate_file_size(file_size)
            assert is_valid_size is True
        
        # Detect framework
        framework = ModelValidator.detect_framework_from_file(temp_model_file)
        
        # Full validation
        result = ModelValidator.validate_model_file(temp_model_file)
        assert isinstance(result, ModelValidationResult)
    
    def test_file_management_workflow(self):
        """Test complete file management workflow"""
        model_id = "test123"
        filename = "test_model.pkl"
        file_content = b"test model content"
        
        with patch('app.utils.model_utils.settings') as mock_settings, \
             patch('os.makedirs') as mock_makedirs, \
             patch('builtins.open', mock_open()) as mock_file_open:
            
            mock_settings.MODELS_DIR = "/tmp/models"
            
            # Save file
            storage_path = ModelFileManager.save_uploaded_file(file_content, model_id, filename)
            assert storage_path is not None
            
            # Get file info
            with patch('os.path.exists', return_value=True), \
                 patch('os.path.isdir', return_value=False), \
                 patch('os.stat') as mock_stat:
                
                mock_stat_result = MagicMock()
                mock_stat_result.st_size = len(file_content)
                mock_stat_result.st_ctime = 1234567890
                mock_stat_result.st_mtime = 1234567890
                mock_stat.return_value = mock_stat_result
                
                file_info = ModelFileManager.get_file_info(storage_path)
                assert file_info['exists'] is True
            
            # Delete file
            with patch('os.path.exists', return_value=True), \
                 patch('os.path.isdir', return_value=False), \
                 patch('os.remove') as mock_remove:
                
                result = ModelFileManager.delete_model_file(storage_path)
                assert result is True
                mock_remove.assert_called_once()


class TestErrorHandling:
    """Test error handling in model utilities"""
    
    def test_model_validator_error_handling(self):
        """Test ModelValidator handles errors gracefully"""
        # Test with invalid file path
        result = ModelValidator.validate_model_file("/invalid/path/model.pkl")
        assert result.is_valid is False
        assert len(result.errors) > 0
    
    @patch('app.utils.model_utils.joblib.load', side_effect=Exception("Load error"))
    def test_framework_detection_error_handling(self, mock_joblib_load):
        """Test framework detection handles load errors gracefully"""
        framework = ModelValidator._detect_pickle_framework("dummy_path")
        assert framework is None
    
    @patch('builtins.open', side_effect=PermissionError("Permission denied"))
    def test_file_manager_permission_error(self, mock_open):
        """Test ModelFileManager handles permission errors"""
        with pytest.raises(PermissionError):
            ModelFileManager.save_uploaded_file(b"content", "model123", "test.pkl")
    
    @patch('os.remove', side_effect=PermissionError("Permission denied"))
    @patch('os.path.exists', return_value=True)
    @patch('os.path.isdir', return_value=False)
    def test_delete_file_permission_error(self, mock_isdir, mock_exists, mock_remove):
        """Test file deletion handles permission errors gracefully"""
        result = ModelFileManager.delete_model_file("/path/to/model.pkl")
        assert result is False 