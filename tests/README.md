# EasyMLOps Test Suite

This directory contains comprehensive unit and integration tests for the EasyMLOps platform.

## 📊 Test Statistics

- **Total Tests**: 1,366+ tests
- **Overall Coverage**: 56% (up from 37%)
- **Key Module Coverage**:
  - Deployment Service: 73%
  - Drift Routes: 95%
  - Alert Rules: 88%
  - A/B Testing: 45% (73% when tests pass individually)
- **Test Isolation**: UUID-based fixtures prevent test interference
- **Test File Organization**: Modular structure with files under 500 lines

## 📁 Test Structure

```
tests/
├── conftest.py              # Pytest configuration and shared fixtures
├── fixtures/                # Reusable test fixtures
│   ├── database.py         # Database fixtures with UUID-based isolation
│   └── services.py         # Service fixtures
├── test_config.py          # Configuration module tests
├── test_database.py         # Database operations tests
├── test_models/            # Database models tests
│   └── test_models.py
├── test_routes/            # API route tests (organized by domain)
│   ├── monitoring/         # Monitoring route tests
│   │   ├── test_drift_comprehensive.py      # 95% coverage
│   │   ├── test_alerts.py
│   │   ├── test_ab_testing.py
│   │   ├── test_canary.py
│   │   ├── test_analytics.py
│   │   ├── test_governance.py
│   │   └── ...             # Domain-specific route tests
│   ├── dynamic/            # Dynamic route tests
│   ├── schemas/            # Schema route tests
│   └── ...                 # Additional route tests
├── test_services/          # Service layer tests (organized by domain)
│   ├── monitoring/         # Monitoring service tests
│   │   ├── test_ab_testing_comprehensive.py
│   │   ├── test_ab_testing_private_methods.py
│   │   ├── test_alert_rules.py              # 88% coverage
│   │   ├── test_alert_management_comprehensive.py
│   │   └── ...             # Domain-specific service tests
│   ├── test_deployment_service_comprehensive.py  # 73% coverage
│   └── ...                 # Additional service tests
├── test_core/              # Core application tests
│   ├── test_app_factory_comprehensive.py
│   └── test_routes_comprehensive.py
├── test_utils/             # Utility tests
└── README.md               # This file
```

## 🧪 Test Categories

### Unit Tests
- **Configuration**: Settings validation, environment variables
- **Database**: Connection, session management, migrations
- **Models**: SQLAlchemy models, validation, relationships
- **Services**: Business logic, monitoring, deployment
- **Schemas**: Pydantic models, validation

### Integration Tests
- **API Endpoints**: FastAPI route testing
- **Service Integration**: Cross-service interactions
- **Database Integration**: Full CRUD operations
- **Model Lifecycle**: Upload → Deploy → Monitor → Delete

## 🚀 Quick Start

### Prerequisites

Install test dependencies:
```bash
pip install pytest pytest-asyncio pytest-cov pytest-mock
pip install httpx  # For FastAPI testing
pip install aiofiles  # For async file operations
```

### Running Tests

#### Basic Usage
```bash
# Run all tests
python -m pytest

# Run with coverage
python -m pytest --cov=app

# Run specific test file
python -m pytest tests/test_config.py

# Run specific test function
python -m pytest tests/test_config.py::TestSettings::test_settings_defaults
```

#### Using the Test Runner
```bash
# Run all tests
python run_tests.py

# Run only unit tests
python run_tests.py --unit

# Run API tests only
python run_tests.py --api

# Run with detailed coverage
python run_tests.py --coverage

# Run specific test file
python run_tests.py --file config

# Quick test suite (no slow tests)
python run_tests.py quick

# CI/CD test suite
python run_tests.py ci
```

## 🏷️ Test Markers

Tests are organized using pytest markers:

- `@pytest.mark.unit` - Unit tests
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.slow` - Slow running tests
- `@pytest.mark.database` - Database-dependent tests
- `@pytest.mark.api` - API endpoint tests
- `@pytest.mark.service` - Service layer tests
- `@pytest.mark.async` - Asynchronous tests

### Running by Marker
```bash
# Unit tests only
python -m pytest -m unit

# Skip slow tests
python -m pytest -m "not slow"

# Database tests only
python -m pytest -m database
```

## 🔧 Test Configuration

### Environment Variables
Tests use the following environment variables:
```bash
TESTING=true
DATABASE_URL=sqlite:///./test.db
DEBUG=true
LOG_LEVEL=ERROR
ENABLE_METRICS=false
```

### Test Database
- Uses SQLite in-memory database for speed
- Fresh database created for each test function
- Automatic cleanup after tests

### Fixtures
Key fixtures available in `conftest.py`:

#### Database Fixtures
- `test_engine` - Test database engine
- `test_session` - Database session for tests (isolated per test)
- `isolated_test_session` - Synchronous isolated database session for test isolation
- `test_settings` - Test configuration

#### Model Fixtures
- `test_model` - Sample model instance (UUID-based ID)
- `sample_deployment` - Sample deployment instance (UUID-based ID)
- `sample_model_data` - Test model data
- `temp_model_file` - Temporary model file
- `mock_prediction_log` - Mock prediction log (UUID-based ID)
- `mock_performance_metric` - Mock performance metric (UUID-based ID)
- `sample_alerts_list` - Sample alerts (UUID-based IDs)

#### API Fixtures
- `client` - FastAPI test client
- `async_mock` - Async mock factory

## 📊 Coverage

The test suite currently achieves **56% overall coverage** (up from 37%), with key modules achieving 70-95% coverage. Coverage reports are generated in:
- Terminal: Summary report
- HTML: `htmlcov/index.html`
- XML: `coverage.xml` (for CI/CD)

### Coverage by Module

| Module | Coverage | Status |
|--------|----------|--------|
| Deployment Service | 73% | ✅ Excellent |
| Drift Routes | 95% | ✅ Excellent |
| Alert Rules | 88% | ✅ Excellent |
| A/B Testing | 45% | 🔄 In Progress |
| Overall | 56% | ✅ Good |

### Running Coverage
```bash
# Basic coverage
python -m pytest --cov=app

# Detailed coverage with missing lines
python -m pytest --cov=app --cov-report=term-missing

# HTML report
python -m pytest --cov=app --cov-report=html

# Coverage for specific module
python -m pytest --cov=app.services.deployment_service --cov-report=term
```

## 🏃‍♂️ Test Development

### Writing New Tests

1. **Create test file**: Follow naming convention `test_*.py`
2. **Import fixtures**: Use fixtures from `conftest.py`
3. **Add markers**: Mark tests with appropriate categories
4. **Follow patterns**: Use existing tests as templates

#### Example Test Structure
```python
import pytest
from unittest.mock import patch, MagicMock

class TestYourFeature:
    """Test your feature functionality"""
    
    def test_basic_functionality(self, test_session):
        """Test basic functionality"""
        # Arrange
        data = {"test": "data"}
        
        # Act
        result = your_function(data)
        
        # Assert
        assert result is not None
        assert result["test"] == "data"
    
    @pytest.mark.async
    async def test_async_functionality(self, test_session):
        """Test async functionality"""
        result = await your_async_function()
        assert result is True
    
    @patch('your.module.dependency')
    def test_with_mock(self, mock_dep, test_session):
        """Test with mocked dependencies"""
        mock_dep.return_value = "mocked"
        result = your_function_with_dependency()
        assert result == "mocked"
```

### Test Data Management

#### Using Fixtures
```python
def test_with_model(self, test_model):
    """Test using model fixture"""
    assert test_model.name == "test_model"
    # test_model is automatically created and cleaned up

def test_with_custom_data(self, test_session, sample_model_data):
    """Test with custom test data"""
    from app.models.model import Model
    
    custom_data = sample_model_data.copy()
    custom_data["name"] = "custom_model"
    
    model = Model(**custom_data)
    test_session.add(model)
    test_session.commit()
```

#### File Testing
```python
def test_file_upload(self, client, temp_model_file):
    """Test file upload functionality"""
    with open(temp_model_file, "rb") as f:
        files = {"file": ("test.joblib", f, "application/octet-stream")}
        response = client.post("/upload", files=files)
    
    assert response.status_code == 200
```

### Mocking Guidelines

#### Database Mocking
```python
@patch('app.database.SessionLocal')
def test_db_error(self, mock_session_local):
    mock_session = MagicMock()
    mock_session.query.side_effect = Exception("DB Error")
    mock_session_local.return_value = mock_session
    
    # Test error handling
```

#### External Service Mocking
```python
@patch('app.services.bentoml_service.bentoml')
def test_bentoml_service(self, mock_bentoml):
    mock_bentoml.build.return_value = MagicMock(tag="test:latest")
    
    # Test BentoML integration
```

#### Async Mocking
```python
@pytest.mark.asyncio
async def test_async_service(self, async_mock):
    with patch('aiohttp.ClientSession.get') as mock_get:
        mock_response = async_mock(return_value={"status": "ok"})
        mock_get.return_value.__aenter__.return_value.json = mock_response
        
        # Test async HTTP calls
```

## 🚨 Debugging Tests

### Failed Test Debugging
```bash
# Drop into debugger on failure
python -m pytest --pdb

# Show full traceback
python -m pytest --tb=long

# Run last failed tests only
python -m pytest --lf

# Stop on first failure
python -m pytest -x
```

### Logging in Tests
```python
import logging

def test_with_logging(self, caplog):
    with caplog.at_level(logging.INFO):
        your_function_that_logs()
    
    assert "Expected log message" in caplog.text
```

## 🔄 Continuous Integration

### GitHub Actions Example
```yaml
name: Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: "3.9"
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov
      - name: Run tests
        run: python run_tests.py ci
      - name: Upload coverage
        uses: codecov/codecov-action@v1
```

## 📈 Performance Testing

### Slow Test Management
Mark slow tests to exclude from quick runs:
```python
@pytest.mark.slow
def test_heavy_computation(self):
    # Long-running test
    pass
```

### Parallel Testing
```bash
# Install pytest-xdist
pip install pytest-xdist

# Run tests in parallel
python -m pytest -n 4  # 4 workers
python run_tests.py --parallel 4
```

## 🛡️ Security Testing

### Environment Isolation
- Tests run in isolated environment
- Temporary files cleaned up automatically
- Test database separate from development/production

### Secrets Management
```python
@patch.dict(os.environ, {"SECRET_KEY": "test-secret"})
def test_with_secrets(self):
    # Test with controlled environment
    pass
```

## 📚 Best Practices

### Test Naming
- Use descriptive test names: `test_upload_model_with_invalid_extension`
- Group related tests in classes: `TestModelUpload`
- Follow AAA pattern: Arrange, Act, Assert

### Test Independence
- Each test should be independent
- Use fixtures for shared setup
- Clean up after tests (automatic with fixtures)
- **Use UUIDs for test data**: All fixtures use `str(uuid.uuid4())` instead of hardcoded IDs to prevent test isolation issues
- **Isolated database sessions**: Each test gets a fresh database session via `test_session` fixture

### Error Testing
```python
def test_error_handling(self):
    with pytest.raises(ValueError, match="Expected error message"):
        function_that_should_raise_error()
```

### Parameterized Tests
```python
@pytest.mark.parametrize("input,expected", [
    ("valid_input", True),
    ("invalid_input", False),
    ("", False),
])
def test_validation(self, input, expected):
    result = validate_input(input)
    assert result == expected
```

## 🔍 Troubleshooting

### Common Issues

1. **Import Errors**: Check PYTHONPATH and relative imports
2. **Database Errors**: Ensure test database is properly isolated
3. **Async Errors**: Use `pytest-asyncio` and `asyncio_mode = auto`
4. **File Cleanup**: Use `temp_model_file` fixture for temporary files

### Getting Help
- Check pytest documentation: https://docs.pytest.org/
- FastAPI testing: https://fastapi.tiangolo.com/tutorial/testing/
- SQLAlchemy testing: https://docs.sqlalchemy.org/en/14/orm/session_transaction.html

## 📝 Contributing

When adding new features:
1. Write tests first (TDD approach)
2. Ensure tests pass: `python run_tests.py`
3. Check coverage: `python run_tests.py --coverage`
4. Update documentation if needed

### Test Checklist
- [ ] Unit tests for new functions/methods
- [ ] Integration tests for API endpoints
- [ ] Error handling tests
- [ ] Edge case tests
- [ ] Performance considerations
- [ ] Documentation updates
- [ ] **Use UUIDs for test data IDs** (not hardcoded strings)
- [ ] **Ensure test isolation** (no shared state between tests)
- [ ] **Keep test files under 500 lines** (split if needed)

## 🔧 Recent Improvements

### Test Isolation Fixes
- Replaced all hardcoded IDs with UUIDs in fixtures
- Fixed test interference issues in:
  - `tests/fixtures/database.py`
  - `tests/test_services/monitoring/test_alert_rules.py`
  - `tests/test_services/monitoring/test_alert_management_comprehensive.py`
  - `tests/test_routes/monitoring/test_alerts.py`
  - `tests/test_services/test_deployment_service_comprehensive.py`

### Coverage Improvements
- Added comprehensive tests for `deployment_service.py` (54% → 73%)
- Added error handling tests for drift routes (90% → 95%)
- Added tests for alert rules service (18% → 88%)
- Added tests for A/B testing private methods

### Test Organization
- Reorganized test files to mirror application structure
- Split large test files to stay under 500 lines
- Created domain-specific test directories (`monitoring/`, `dynamic/`, `schemas/`) 