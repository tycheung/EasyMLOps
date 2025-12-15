"""
Comprehensive tests for model utilities
Tests validator, file manager, and framework detection
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import tempfile
import os
from pathlib import Path

from app.utils.model_utils import ModelValidator, ModelFileManager
from app.utils.model_utils.frameworks import FrameworkDetector
from app.schemas.model import ModelFramework, ModelType


class TestModelValidator:
    """Test model validator utilities"""
    
    @pytest.mark.asyncio
    async def test_calculate_file_hash_async(self):
        """Test calculating file hash asynchronously"""
        with tempfile.NamedTemporaryFile(delete=False, mode='wb') as f:
            f.write(b"test content")
            temp_path = f.name
        
        try:
            hash_value = await ModelValidator.calculate_file_hash_async(temp_path)
            assert hash_value is not None
            assert isinstance(hash_value, str)
            assert len(hash_value) == 64  # SHA-256 hex digest length
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    @pytest.mark.asyncio
    async def test_calculate_file_hash_async_file_not_found(self):
        """Test calculating hash for non-existent file"""
        with pytest.raises(Exception):
            await ModelValidator.calculate_file_hash_async("/nonexistent/path/file.pkl")
    
    def test_validate_file_extension_valid(self):
        """Test validating valid file extension"""
        assert ModelValidator.validate_file_extension("model.pkl") is True
        assert ModelValidator.validate_file_extension("model.joblib") is True
        assert ModelValidator.validate_file_extension("model.h5") is True
        assert ModelValidator.validate_file_extension("model.pt") is True
    
    def test_validate_file_extension_invalid(self):
        """Test validating invalid file extension"""
        assert ModelValidator.validate_file_extension("model.txt") is False
        assert ModelValidator.validate_file_extension("model.exe") is False
    
    def test_validate_file_size_valid(self):
        """Test validating valid file size"""
        assert ModelValidator.validate_file_size(1024) is True
        assert ModelValidator.validate_file_size(1024 * 1024) is True
    
    def test_validate_file_size_invalid(self):
        """Test validating invalid file size"""
        # Assuming MAX_FILE_SIZE is reasonable (e.g., 100MB)
        large_size = 200 * 1024 * 1024 * 1024  # 200GB
        assert ModelValidator.validate_file_size(large_size) is False
    
    @pytest.mark.asyncio
    @patch('app.utils.model_utils.frameworks.FrameworkDetector.detect_framework_from_file_async', new_callable=AsyncMock)
    @patch('app.utils.model_utils.frameworks.FrameworkDetector.detect_model_type_async', new_callable=AsyncMock)
    @patch('app.utils.model_utils.frameworks.FrameworkDetector.get_framework_metadata_async', new_callable=AsyncMock)
    async def test_validate_model_file_async_success(self, mock_metadata, mock_type, mock_framework):
        """Test validating model file successfully"""
        mock_framework.return_value = ModelFramework.SKLEARN
        mock_type.return_value = ModelType.CLASSIFICATION
        mock_metadata.return_value = {"n_features": 10}
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl', mode='wb') as f:
            f.write(b"fake model content")
            temp_path = f.name
        
        try:
            result = await ModelValidator.validate_model_file_async(temp_path)
            assert result.is_valid is True
            assert result.framework_detected == ModelFramework.SKLEARN
            assert result.model_type_detected == ModelType.CLASSIFICATION
            assert "file_hash" in result.metadata
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    @pytest.mark.asyncio
    async def test_validate_model_file_async_file_not_found(self):
        """Test validating non-existent file"""
        result = await ModelValidator.validate_model_file_async("/nonexistent/file.pkl")
        assert result.is_valid is False
        assert len(result.errors) > 0
        assert "not found" in result.errors[0].lower()
    
    @pytest.mark.asyncio
    async def test_validate_model_file_async_invalid_extension(self):
        """Test validating file with invalid extension"""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.txt', mode='wb') as f:
            f.write(b"content")
            temp_path = f.name
        
        try:
            result = await ModelValidator.validate_model_file_async(temp_path)
            assert result.is_valid is False
            assert any("unsupported" in error.lower() for error in result.errors)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)


class TestModelFileManager:
    """Test model file manager utilities"""
    
    @pytest.mark.asyncio
    async def test_get_model_storage_path_async(self):
        """Test getting model storage path"""
        model_id = "test_model_123"
        filename = "model.pkl"
        
        path = await ModelFileManager.get_model_storage_path_async(model_id, filename)
        
        assert path is not None
        assert isinstance(path, str)
        assert model_id in path
        assert filename in path or Path(filename).name in path
    
    @pytest.mark.asyncio
    async def test_save_uploaded_file_async(self):
        """Test saving uploaded file"""
        model_id = "test_model_123"
        filename = "model.pkl"
        file_content = b"test model content"
        
        try:
            path = await ModelFileManager.save_uploaded_file_async(file_content, model_id, filename)
            
            assert path is not None
            assert os.path.exists(path)
            
            # Verify content
            with open(path, 'rb') as f:
                saved_content = f.read()
            assert saved_content == file_content
        finally:
            # Cleanup
            if 'path' in locals() and os.path.exists(path):
                os.remove(path)
                # Try to remove parent directory if empty
                parent_dir = os.path.dirname(path)
                try:
                    os.rmdir(parent_dir)
                except OSError:
                    pass
    
    @pytest.mark.asyncio
    async def test_delete_model_file_async(self):
        """Test deleting model file"""
        model_id = "test_model_123"
        filename = "model.pkl"
        file_content = b"test content"
        
        # Create file first
        path = await ModelFileManager.save_uploaded_file_async(file_content, model_id, filename)
        
        try:
            # Delete file
            result = await ModelFileManager.delete_model_file_async(path)
            
            assert result is True
            assert not os.path.exists(path)
        finally:
            # Cleanup if still exists
            if os.path.exists(path):
                os.remove(path)
    
    @pytest.mark.asyncio
    async def test_delete_model_file_async_not_found(self):
        """Test deleting non-existent file"""
        result = await ModelFileManager.delete_model_file_async("/nonexistent/path/file.pkl")
        assert result is False
    
    @pytest.mark.asyncio
    async def test_get_file_info_async(self):
        """Test getting file information"""
        model_id = "test_model_123"
        filename = "model.pkl"
        file_content = b"test content"
        
        # Create file first
        path = await ModelFileManager.save_uploaded_file_async(file_content, model_id, filename)
        
        try:
            info = await ModelFileManager.get_file_info_async(path)
            
            assert info is not None
            assert info['exists'] is True
            assert info['size'] == len(file_content)
            assert 'created_at' in info
            assert 'updated_at' in info
            assert info['is_directory'] is False
        finally:
            # Cleanup
            if os.path.exists(path):
                os.remove(path)
                parent_dir = os.path.dirname(path)
                try:
                    os.rmdir(parent_dir)
                except OSError:
                    pass
    
    @pytest.mark.asyncio
    async def test_get_file_info_async_not_found(self):
        """Test getting info for non-existent file"""
        info = await ModelFileManager.get_file_info_async("/nonexistent/path/file.pkl")
        assert info['exists'] is False


class TestFrameworkDetector:
    """Test framework detection utilities"""
    
    @pytest.mark.asyncio
    async def test_detect_framework_from_file_async_pkl(self):
        """Test detecting framework from .pkl file"""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl', mode='wb') as f:
            import pickle
            pickle.dump({"test": "data"}, f)
            temp_path = f.name
        
        try:
            framework = await FrameworkDetector.detect_framework_from_file_async(temp_path)
            # May return None if framework can't be detected from pickle
            assert framework is None or isinstance(framework, ModelFramework)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    @pytest.mark.asyncio
    async def test_detect_framework_from_file_async_h5(self):
        """Test detecting framework from .h5 file"""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.h5', mode='wb') as f:
            f.write(b"fake h5 content")
            temp_path = f.name
        
        try:
            framework = await FrameworkDetector.detect_framework_from_file_async(temp_path)
            assert framework == ModelFramework.TENSORFLOW
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    @pytest.mark.asyncio
    async def test_detect_framework_from_file_async_pt(self):
        """Test detecting framework from .pt file"""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pt', mode='wb') as f:
            f.write(b"fake pytorch content")
            temp_path = f.name
        
        try:
            framework = await FrameworkDetector.detect_framework_from_file_async(temp_path)
            assert framework == ModelFramework.PYTORCH
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    @pytest.mark.asyncio
    async def test_detect_framework_from_file_async_onnx(self):
        """Test detecting framework from .onnx file"""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.onnx', mode='wb') as f:
            f.write(b"fake onnx content")
            temp_path = f.name
        
        try:
            framework = await FrameworkDetector.detect_framework_from_file_async(temp_path)
            assert framework == ModelFramework.ONNX
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    @pytest.mark.asyncio
    async def test_detect_framework_from_file_async_unknown(self):
        """Test detecting framework from unknown file type"""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.unknown', mode='wb') as f:
            f.write(b"content")
            temp_path = f.name
        
        try:
            framework = await FrameworkDetector.detect_framework_from_file_async(temp_path)
            assert framework is None
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    @pytest.mark.asyncio
    async def test_detect_framework_from_file_async_not_found(self):
        """Test detecting framework from non-existent file"""
        framework = await FrameworkDetector.detect_framework_from_file_async("/nonexistent/file.pkl")
        assert framework is None

