"""
Tests for ModelFileManager class
Tests model file loading, storage, and file management operations
"""

import pytest
import tempfile
import os
from unittest.mock import patch, MagicMock

from app.utils.model_utils import ModelFileManager


class TestModelFileManager:
    """Test ModelFileManager class functionality"""
    
    @pytest.mark.asyncio
    @patch('app.utils.model_utils.loaders.settings')
    @patch('app.utils.model_utils.loaders.aios.makedirs')
    async def test_get_model_storage_path_async(self, mock_aios_makedirs, mock_settings):
        """Test getting model storage path asynchronously"""
        mock_settings.MODELS_DIR = "/tmp/models_test"
        model_id = "test_model_123"
        filename = "model.pkl"
        
        expected_subdir = model_id[:2]
        expected_path = f"/tmp/models_test/{expected_subdir}/{model_id}_{filename}"
        
        storage_path = await ModelFileManager.get_model_storage_path_async(model_id, filename)
        
        import os
        assert os.path.normpath(storage_path) == os.path.normpath(expected_path)
        mock_aios_makedirs.assert_called_once()
        call_args = mock_aios_makedirs.call_args
        assert call_args[1]['exist_ok'] is True
        actual_path = call_args[0][0]
        assert expected_subdir in actual_path

    @pytest.mark.asyncio
    @patch('app.utils.model_utils.loaders.settings')
    @patch('app.utils.model_utils.loaders.aios.makedirs')
    @patch('app.utils.model_utils.loaders.aiofiles.open')
    async def test_save_uploaded_file_async(self, mock_aio_open, mock_aios_makedirs, mock_settings):
        """Test saving uploaded file asynchronously"""
        mock_settings.MODELS_DIR = "/tmp/models_test_save"
        model_id = "save_test_456"
        filename = "uploaded_model.dat"
        file_content = b"dummy file content"
        
        expected_storage_path = f"/tmp/models_test_save/{model_id[:2]}/{model_id}_{filename}"
        with patch('app.utils.model_utils.loaders.ModelFileManager.get_model_storage_path_async', return_value=expected_storage_path) as mock_get_path:
        
            storage_path = await ModelFileManager.save_uploaded_file_async(file_content, model_id, filename)
            
            assert storage_path == expected_storage_path
            mock_get_path.assert_called_once_with(model_id, filename)
            mock_aio_open.assert_called_once_with(expected_storage_path, 'wb')
            mock_file_context = mock_aio_open.return_value.__aenter__.return_value
            mock_file_context.write.assert_called_once_with(file_content)

    @pytest.mark.asyncio
    @patch('app.utils.model_utils.loaders.settings')
    @patch('app.utils.model_utils.loaders.aios.makedirs')
    @patch('app.utils.model_utils.loaders.asyncio.to_thread')
    @patch('app.utils.model_utils.loaders.aios.path.exists', return_value=False)
    async def test_save_directory_model_async(self, mock_aios_exists, mock_to_thread, mock_aios_makedirs, mock_settings):
        """Test saving directory model asynchronously"""
        mock_settings.MODELS_DIR = "/tmp/models_test_dir_save"
        model_id = "dir_model_789"
        source_dir = "/path/to/source_model_dir"
        dirname = "my_tf_model"

        expected_storage_path = f"/tmp/models_test_dir_save/{model_id[:2]}/{model_id}_{dirname}"
        with patch('app.utils.model_utils.loaders.ModelFileManager.get_model_storage_path_async', return_value=expected_storage_path) as mock_get_path:

            storage_path = await ModelFileManager.save_directory_model_async(source_dir, model_id, dirname)
            
            assert storage_path == expected_storage_path
            mock_get_path.assert_called_once_with(model_id, dirname)
            mock_aios_exists.assert_called_once_with(expected_storage_path)
            mock_to_thread.assert_called()
            assert len(mock_to_thread.call_args_list) > 0

    @pytest.mark.asyncio
    @patch('app.utils.model_utils.loaders.aios.path.exists', return_value=True)
    @patch('app.utils.model_utils.loaders.aios.path.isdir', return_value=False)
    @patch('app.utils.model_utils.loaders.aios.remove')
    async def test_delete_model_file_async(self, mock_aios_remove, mock_aios_isdir, mock_aios_exists):
        """Test deleting a model file asynchronously"""
        file_path = "/tmp/models_test/test_model.pkl"
        
        result = await ModelFileManager.delete_model_file_async(file_path)
        
        assert result is True
        mock_aios_exists.assert_called_once_with(file_path)
        mock_aios_isdir.assert_called_once_with(file_path)
        mock_aios_remove.assert_called_once_with(file_path)

    @pytest.mark.asyncio
    @patch('app.utils.model_utils.loaders.aios.path.exists', return_value=True)
    @patch('app.utils.model_utils.loaders.aios.path.isdir', return_value=True)
    @patch('app.utils.model_utils.loaders.asyncio.to_thread')
    async def test_delete_model_directory_async(self, mock_to_thread, mock_aios_isdir, mock_aios_exists):
        """Test deleting a model directory asynchronously"""
        dir_path = "/tmp/models_test/test_model_dir"
        
        result = await ModelFileManager.delete_model_file_async(dir_path)
        
        assert result is True
        mock_aios_exists.assert_called_once_with(dir_path)
        mock_aios_isdir.assert_called_once_with(dir_path)
        mock_to_thread.assert_called_once()
        assert len(mock_to_thread.call_args[0]) > 0

    @pytest.mark.asyncio
    @patch('app.utils.model_utils.loaders.aios.path.exists', return_value=False)
    async def test_delete_nonexistent_file_async(self, mock_aios_exists):
        """Test deleting a non-existent file asynchronously"""
        file_path = "/tmp/models_test/non_existent.pkl"
        result = await ModelFileManager.delete_model_file_async(file_path)
        assert result is False
        mock_aios_exists.assert_called_once_with(file_path)

    @pytest.mark.asyncio
    @patch('app.utils.model_utils.loaders.aios.path.exists', return_value=True)
    @patch('app.utils.model_utils.loaders.aios.path.isdir', return_value=False)
    @patch('app.utils.model_utils.loaders.aios.stat')
    async def test_get_file_info_file_async(self, mock_aios_stat, mock_aios_isdir, mock_aios_exists):
        """Test getting file info for a file asynchronously"""
        file_path = "/tmp/models_test/some_file.dat"
        
        mock_stat_result = MagicMock()
        mock_stat_result.st_size = 1024
        mock_stat_result.st_ctime = 1234567890.0
        mock_stat_result.st_mtime = 1234567891.0
        mock_aios_stat.return_value = mock_stat_result
        
        info = await ModelFileManager.get_file_info_async(file_path)
        
        assert info['exists'] is True
        assert info['size'] == 1024
        assert info['is_directory'] is False
        assert info['created_at'] is not None
        assert info['updated_at'] is not None
        mock_aios_exists.assert_called_once_with(file_path)
        mock_aios_isdir.assert_called_once_with(file_path)
        mock_aios_stat.assert_called_once_with(file_path)

    @pytest.mark.asyncio  
    @patch('app.utils.model_utils.loaders.aios.path.exists', return_value=True)
    @patch('app.utils.model_utils.loaders.aios.path.isdir', return_value=True)
    @patch('app.utils.model_utils.loaders.aios.stat')
    async def test_get_file_info_directory_async(self, mock_aios_stat_dir, mock_aios_isdir, mock_aios_exists):
        """Test getting file info for a directory asynchronously"""
        dir_path = "/tmp/models_test/some_dir"

        mock_stat_dir_main = MagicMock()
        mock_stat_dir_main.st_size = 1024
        mock_stat_dir_main.st_ctime = 1234500000.0
        mock_stat_dir_main.st_mtime = 1234500001.0
        mock_aios_stat_dir.return_value = mock_stat_dir_main
        
        info = await ModelFileManager.get_file_info_async(dir_path)
        
        assert info['exists'] is True
        assert info['is_directory'] is True
        assert info['size'] >= 0
        mock_aios_exists.assert_called_once_with(dir_path)
        mock_aios_isdir.assert_called_once_with(dir_path)
        mock_aios_stat_dir.assert_called_once_with(dir_path)

    @pytest.mark.asyncio
    @patch('app.utils.model_utils.loaders.aios.path.exists', return_value=False)
    async def test_get_file_info_nonexistent_async(self, mock_aios_exists):
        """Test getting file info for a non-existent file asynchronously"""
        file_path = "/tmp/models_test/no_such_file.xyz"
        info = await ModelFileManager.get_file_info_async(file_path)
        assert info['exists'] is False
        mock_aios_exists.assert_called_once_with(file_path)


class TestFileManagerErrorHandling:
    """Test error handling in ModelFileManager"""
    
    @pytest.mark.asyncio
    @patch('app.utils.model_utils.loaders.ModelFileManager.get_model_storage_path_async', side_effect=PermissionError("Permission denied for path creation"))
    async def test_file_manager_save_permission_error_async(self, mock_get_path):
        """Test ModelFileManager handles permission errors during save"""
        with pytest.raises(PermissionError, match="Permission denied for path creation"):
            await ModelFileManager.save_uploaded_file_async(b"content", "model123", "test.pkl")

    @pytest.mark.asyncio
    @patch('app.utils.model_utils.loaders.aios.path.exists', return_value=True)
    @patch('app.utils.model_utils.loaders.aios.path.isdir', return_value=False)
    @patch('app.utils.model_utils.loaders.aios.remove', side_effect=PermissionError("Permission denied for delete"))
    async def test_delete_file_permission_error_async(self, mock_aios_remove, mock_aios_isdir, mock_aios_exists):
        """Test file deletion handles permission errors gracefully"""
        result = await ModelFileManager.delete_model_file_async("/path/to/model.pkl")
        assert result is False
        mock_aios_exists.assert_called_once_with("/path/to/model.pkl")
        mock_aios_isdir.assert_called_once_with("/path/to/model.pkl")
        mock_aios_remove.assert_called_once_with("/path/to/model.pkl")

