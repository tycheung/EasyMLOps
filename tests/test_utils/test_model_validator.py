"""
Tests for ModelValidator class
Tests model validation, framework detection, and validation-related error handling
"""

import pytest
import tempfile
import os
import json
import zipfile
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path

from app.utils.model_utils import (
    ModelValidator,
    TENSORFLOW_AVAILABLE, PYTORCH_AVAILABLE, SKLEARN_AVAILABLE
)
from app.schemas.model import ModelFramework, ModelType, ModelValidationResult


@pytest.fixture
def temp_model_file():
    """Create a temporary model file for testing"""
    import tempfile
    import time
    
    # Create temp file with explicit close to avoid Windows file locking
    f = tempfile.NamedTemporaryFile(suffix='.joblib', delete=False)
    temp_path = f.name
    f.close()  # Close immediately to release file handle
    
    try:
        if SKLEARN_AVAILABLE:
            from sklearn.linear_model import LogisticRegression
            # Optional joblib import
            try:
                import joblib
                JOBLIB_AVAILABLE = True
            except ImportError:
                joblib = None
                JOBLIB_AVAILABLE = False
            
            model = LogisticRegression()
            import numpy as np
            X = np.array([[1, 2], [3, 4], [5, 6]])
            y = np.array([0, 1, 0])
            model.fit(X, y)
            
            if JOBLIB_AVAILABLE:
                joblib.dump(model, temp_path)
            else:
                # Fallback to pickle
                import pickle
                with open(temp_path, 'wb') as pf:
                    pickle.dump(model, pf)
        else:
            with open(temp_path, 'wb') as f:
                f.write(b"dummy model data for non-sklearn env")
        
        yield temp_path
        
    finally:
        # Clean up with retry logic for Windows
        if os.path.exists(temp_path):
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    os.unlink(temp_path)
                    break
                except (PermissionError, OSError) as e:
                    if attempt < max_retries - 1:
                        time.sleep(0.1 * (attempt + 1))  # Exponential backoff
                    else:
                        # Last attempt - try to delete on Windows with a delay
                        import time
                        time.sleep(0.5)
                        try:
                            os.unlink(temp_path)
                        except:
                            pass  # Ignore final failure


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
    
    @pytest.mark.asyncio
    async def test_calculate_file_hash_async(self, temp_model_file):
        """Test file hash calculation asynchronously"""
        hash_value = await ModelValidator.calculate_file_hash_async(temp_model_file)
        
        assert isinstance(hash_value, str)
        assert len(hash_value) == 64
        
        hash_value2 = await ModelValidator.calculate_file_hash_async(temp_model_file)
        assert hash_value == hash_value2
    
    @pytest.mark.asyncio
    async def test_calculate_file_hash_nonexistent_file_async(self):
        """Test file hash calculation with non-existent file asynchronously"""
        with pytest.raises(Exception):
            await ModelValidator.calculate_file_hash_async("nonexistent_file.txt")
    
    @patch('app.utils.model_utils.validator.settings')
    def test_validate_file_extension_valid(self, mock_settings):
        """Test file extension validation for valid extensions"""
        mock_settings.ALLOWED_MODEL_EXTENSIONS = ['.pkl', '.joblib', '.h5', '.pt']
        
        assert ModelValidator.validate_file_extension("model.pkl") is True
        assert ModelValidator.validate_file_extension("model.joblib") is True
        assert ModelValidator.validate_file_extension("model.h5") is True
        assert ModelValidator.validate_file_extension("model.pt") is True
    
    @patch('app.utils.model_utils.validator.settings')
    def test_validate_file_extension_invalid(self, mock_settings):
        """Test file extension validation for invalid extensions"""
        mock_settings.ALLOWED_MODEL_EXTENSIONS = ['.pkl', '.joblib']
        
        assert ModelValidator.validate_file_extension("model.txt") is False
        assert ModelValidator.validate_file_extension("model.pdf") is False
        assert ModelValidator.validate_file_extension("model.exe") is False
    
    @patch('app.utils.model_utils.validator.settings')
    def test_validate_file_size_valid(self, mock_settings):
        """Test file size validation for valid sizes"""
        mock_settings.MAX_FILE_SIZE = 1024 * 1024 * 100
        
        assert ModelValidator.validate_file_size(1024) is True
        assert ModelValidator.validate_file_size(1024 * 1024) is True
        assert ModelValidator.validate_file_size(1024 * 1024 * 50) is True
    
    @patch('app.utils.model_utils.validator.settings')
    def test_validate_file_size_invalid(self, mock_settings):
        """Test file size validation for invalid sizes"""
        mock_settings.MAX_FILE_SIZE = 1024 * 1024
        
        assert ModelValidator.validate_file_size(1024 * 1024 * 2) is False
        assert ModelValidator.validate_file_size(1024 * 1024 * 10) is False
    
    @pytest.mark.asyncio
    async def test_detect_framework_from_joblib_file_async(self, temp_model_file):
        """Test framework detection from joblib file asynchronously"""
        from app.utils.model_utils.frameworks import FrameworkDetector
        framework = await FrameworkDetector.detect_framework_from_file_async(temp_model_file)
        
        if SKLEARN_AVAILABLE:
            assert framework == ModelFramework.SKLEARN
        else:
            assert framework is None or framework == ModelFramework.CUSTOM
    
    @pytest.mark.asyncio
    async def test_detect_framework_from_json_file_async(self, temp_json_file):
        """Test framework detection from JSON file asynchronously"""
        from app.utils.model_utils.frameworks import FrameworkDetector
        framework = await FrameworkDetector.detect_framework_from_file_async(temp_json_file)
        assert framework == ModelFramework.XGBOOST
    
    @pytest.mark.asyncio
    async def test_detect_framework_from_zip_file_async(self, temp_zip_file):
        """Test framework detection from ZIP file asynchronously"""
        from app.utils.model_utils.frameworks import FrameworkDetector
        framework = await FrameworkDetector.detect_framework_from_file_async(temp_zip_file)
        assert framework == ModelFramework.TENSORFLOW
    
    @pytest.mark.asyncio
    async def test_detect_framework_tensorflow_extensions_async(self):
        """Test framework detection for TensorFlow file extensions asynchronously"""
        from app.utils.model_utils.frameworks import FrameworkDetector
        test_files = ["model.h5", "model.pb", "model.tflite"]
        
        for file_path in test_files:
            with patch('app.utils.model_utils.frameworks.aios.path.isdir', return_value=False):
                with patch('app.utils.model_utils.frameworks.aios.path.exists', return_value=True):
                    framework = await FrameworkDetector.detect_framework_from_file_async(file_path)
                    assert framework == ModelFramework.TENSORFLOW
    
    @pytest.mark.asyncio
    async def test_detect_framework_pytorch_extensions_async(self):
        """Test framework detection for PyTorch file extensions asynchronously"""
        from app.utils.model_utils.frameworks import FrameworkDetector
        test_files = ["model.pt", "model.pth"]
        
        for file_path in test_files:
            with patch('app.utils.model_utils.frameworks.aios.path.isdir', return_value=False):
                with patch('app.utils.model_utils.frameworks.aios.path.exists', return_value=True):
                    framework = await FrameworkDetector.detect_framework_from_file_async(file_path)
                    assert framework == ModelFramework.PYTORCH
    
    @pytest.mark.asyncio
    async def test_detect_framework_onnx_extension_async(self):
        """Test framework detection for ONNX file extension asynchronously"""
        from app.utils.model_utils.frameworks import FrameworkDetector
        with patch('app.utils.model_utils.frameworks.aios.path.isdir', return_value=False):
            with patch('app.utils.model_utils.frameworks.aios.path.exists', return_value=True):
                framework = await FrameworkDetector.detect_framework_from_file_async("model.onnx")
                assert framework == ModelFramework.ONNX
    
    @pytest.mark.asyncio
    async def test_detect_framework_unknown_extension_async(self):
        """Test framework detection for unknown file extension asynchronously"""
        from app.utils.model_utils.frameworks import FrameworkDetector
        with patch('app.utils.model_utils.frameworks.aios.path.isdir', return_value=False):
            with patch('app.utils.model_utils.frameworks.aios.path.exists', return_value=True):
                framework = await FrameworkDetector.detect_framework_from_file_async("model.unknown")
                assert framework is None
    
    @pytest.mark.asyncio
    @patch('app.utils.model_utils.frameworks.asyncio.to_thread')
    async def test_detect_pickle_framework_sklearn_async(self, mock_to_thread):
        """Test detecting sklearn from pickle file asynchronously"""
        from app.utils.model_utils.frameworks import FrameworkDetector
        mock_model = MagicMock()
        mock_model.__module__ = 'sklearn.linear_model'
        
        async def fake_joblib_load(*args, **kwargs):
            return mock_model
        mock_to_thread.side_effect = fake_joblib_load

        framework = await FrameworkDetector._detect_pickle_framework_async("dummy_path.pkl")
        assert framework == ModelFramework.SKLEARN
        mock_to_thread.assert_called_once()

    @pytest.mark.asyncio
    @patch('app.utils.model_utils.frameworks.asyncio.to_thread')
    async def test_detect_pickle_framework_xgboost_async(self, mock_to_thread):
        """Test detecting XGBoost from pickle file asynchronously"""
        from app.utils.model_utils.frameworks import FrameworkDetector
        mock_model = MagicMock()
        mock_model.__module__ = 'xgboost.sklearn'
        async def fake_joblib_load(*args, **kwargs):
            return mock_model
        mock_to_thread.side_effect = fake_joblib_load

        framework = await FrameworkDetector._detect_pickle_framework_async("dummy_path.pkl")
        assert framework == ModelFramework.XGBOOST

    @pytest.mark.asyncio
    @patch('app.utils.model_utils.frameworks.asyncio.to_thread')
    @patch('app.utils.model_utils.frameworks.aiofiles.open', new_callable=mock_open)
    async def test_detect_pickle_framework_custom_async(self, mock_aio_open, mock_to_thread):
        """Test detecting custom (default sklearn) from pickle file if unknown type"""
        from app.utils.model_utils.frameworks import FrameworkDetector
        mock_model = MagicMock()
        mock_model.__module__ = 'some_unknown_module.model'
        
        async def fake_joblib_load(*args, **kwargs):
            return mock_model
        mock_to_thread.side_effect = fake_joblib_load

        framework = await FrameworkDetector._detect_pickle_framework_async("dummy_path.pkl")
        assert framework == ModelFramework.SKLEARN

    @pytest.mark.asyncio
    async def test_detect_json_framework_patterns_async(self):
        """Test JSON framework detection with different patterns asynchronously"""
        from app.utils.model_utils.frameworks import FrameworkDetector
        test_cases = [
            ({"xgboost": "config"}, ModelFramework.XGBOOST),
            ({"lightgbm": "config"}, ModelFramework.LIGHTGBM),
            ({"h2o": "config"}, ModelFramework.H2O),
            ({"tensorflow": "config"}, ModelFramework.TENSORFLOW),
            ({"pytorch": "config"}, ModelFramework.PYTORCH),
            ({"custom": "config"}, ModelFramework.CUSTOM)
        ]
        
        for data, expected_framework in test_cases:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f_sync:
                json.dump(data, f_sync)
                temp_file_name = f_sync.name
            
            try:
                framework = await FrameworkDetector._detect_json_framework_async(temp_file_name)
                assert framework == expected_framework
            finally:
                if os.path.exists(temp_file_name):
                    os.unlink(temp_file_name)

    @pytest.mark.asyncio
    @patch('app.utils.model_utils.frameworks.asyncio.to_thread')
    async def test_detect_model_type_sklearn_classification_async(self, mock_to_thread):
        """Test detecting sklearn classification model type asynchronously"""
        from app.utils.model_utils.frameworks import FrameworkDetector
        mock_model = MagicMock()
        mock_model.__class__.__name__ = 'LogisticRegression'
        async def fake_joblib_load(*args, **kwargs): return mock_model
        mock_to_thread.side_effect = fake_joblib_load
        
        model_type = await FrameworkDetector.detect_model_type_async("dummy.pkl", ModelFramework.SKLEARN)
        assert model_type == ModelType.CLASSIFICATION

    @pytest.mark.asyncio
    @patch('app.utils.model_utils.frameworks.asyncio.to_thread')
    async def test_detect_model_type_sklearn_regression_async(self, mock_to_thread):
        """Test detecting sklearn regression model type asynchronously"""
        from app.utils.model_utils.frameworks import FrameworkDetector
        mock_model = MagicMock()
        mock_model.__class__.__name__ = 'LinearRegression'
        async def fake_joblib_load(*args, **kwargs): return mock_model
        mock_to_thread.side_effect = fake_joblib_load

        model_type = await FrameworkDetector.detect_model_type_async("dummy.pkl", ModelFramework.SKLEARN)
        assert model_type == ModelType.REGRESSION

    @pytest.mark.asyncio
    @patch('app.utils.model_utils.frameworks.asyncio.to_thread')
    async def test_detect_model_type_sklearn_clustering_async(self, mock_to_thread):
        """Test detecting sklearn clustering model type asynchronously"""
        from app.utils.model_utils.frameworks import FrameworkDetector
        mock_model = MagicMock()
        mock_model.__class__.__name__ = 'KMeans'
        async def fake_joblib_load(*args, **kwargs): return mock_model
        mock_to_thread.side_effect = fake_joblib_load
        
        model_type = await FrameworkDetector.detect_model_type_async("dummy.pkl", ModelFramework.SKLEARN)
        assert model_type == ModelType.CLUSTERING

    @pytest.mark.asyncio
    @patch('app.utils.model_utils.validator.aios.path.exists')
    @patch('app.utils.model_utils.validator.aios.stat')
    @patch('app.utils.model_utils.validator.ModelValidator.calculate_file_hash_async')
    @patch('app.utils.model_utils.frameworks.FrameworkDetector.detect_framework_from_file_async')
    @patch('app.utils.model_utils.frameworks.FrameworkDetector.detect_model_type_async')
    async def test_validate_model_file_comprehensive_async(self,
                                                        mock_detect_type, mock_detect_framework, 
                                                        mock_calc_hash, mock_stat, mock_exists, 
                                                        temp_model_file):
        """Test comprehensive model file validation asynchronously"""
        mock_exists.return_value = True
        mock_stat_result = MagicMock()
        mock_stat_result.st_size = 1024
        mock_stat.return_value = mock_stat_result
        mock_calc_hash.return_value = "dummyhash123"
        mock_detect_framework.return_value = ModelFramework.SKLEARN
        mock_detect_type.return_value = ModelType.CLASSIFICATION

        result = await ModelValidator.validate_model_file_async(temp_model_file)
        
        assert result.is_valid is True
        assert result.framework_detected == ModelFramework.SKLEARN
        assert result.model_type_detected == ModelType.CLASSIFICATION
        assert result.metadata['file_hash'] == "dummyhash123"

    @pytest.mark.asyncio
    @patch('app.utils.model_utils.validator.aios.path.exists')
    async def test_validate_model_file_nonexistent_async(self, mock_exists):
        """Test model file validation for a non-existent file asynchronously"""
        mock_exists.return_value = False
        result = await ModelValidator.validate_model_file_async("nonexistent_file.pkl")
        
        assert result.is_valid is False
        assert "File not found" in result.errors[0]


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
    
    @pytest.mark.asyncio
    async def test_complete_validation_workflow_async(self, temp_model_file):
        """Test the complete validation workflow using a real file (if possible)"""
        if not SKLEARN_AVAILABLE:
            pytest.skip("Skipping validation workflow test as scikit-learn is not available")

        result = await ModelValidator.validate_model_file_async(temp_model_file)

        assert isinstance(result, ModelValidationResult)
        assert result.is_valid is True
        assert result.framework_detected == ModelFramework.SKLEARN
        assert result.model_type_detected is not None
        assert result.metadata['file_hash'] is not None
        assert result.metadata['file_size'] > 0


class TestValidatorErrorHandling:
    """Test error handling in model validator"""
    
    @pytest.mark.asyncio
    @patch('app.utils.model_utils.validator.aios.path.exists', return_value=False)
    async def test_model_validator_error_handling_async(self, mock_exists):
        """Test general error handling in ModelValidator"""
        result = await ModelValidator.validate_model_file_async("/invalid/path/model.pkl")
        assert result.is_valid is False
        assert len(result.errors) > 0 
        assert result.framework_detected in [None, ModelFramework.CUSTOM]
        mock_exists.assert_called_once_with("/invalid/path/model.pkl")

    @pytest.mark.asyncio
    @patch('app.utils.model_utils.frameworks.asyncio.to_thread', side_effect=Exception("Load error"))
    async def test_framework_detection_error_handling_async(self, mock_to_thread):
        """Test error handling during framework detection"""
        result = await ModelValidator._detect_pickle_framework_async("dummy_error.pkl")
        assert result is None

