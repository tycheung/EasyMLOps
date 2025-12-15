"""
Comprehensive tests for model utility functions
Tests model validation, framework detection, file handling, and metadata extraction

NOTE: This file has been refactored. Test classes have been moved to:
- tests/test_utils/test_model_validator.py (validation and framework detection tests)
- tests/test_utils/test_model_loaders.py (file management and loading tests)

This file now re-exports the test classes for backward compatibility.
"""

# Re-export all test classes from the new modules
from tests.test_utils.test_model_validator import (
    TestModelValidator,
    TestFrameworkAvailability,
    TestModelValidationIntegration,
    TestValidatorErrorHandling,
)

from tests.test_utils.test_model_loaders import (
    TestModelFileManager,
    TestFileManagerErrorHandling,
)

# Re-export fixtures for backward compatibility
from tests.test_utils.test_model_validator import (
    temp_model_file,
    temp_json_file,
    temp_zip_file,
)
