# EasyMLOps - Production-Ready ML Deployment Platform

A comprehensive ML Operations platform that empowers data scientists to deploy machine learning models with zero-code production-ready API endpoints, advanced monitoring, schema validation, and enterprise-grade management capabilities.

## ✨ Key Features

### 🚀 **No-Code Model Deployment**
- **Universal Model Support**: Deploy models from any framework (scikit-learn, TensorFlow, PyTorch, XGBoost, LightGBM, H2O, ONNX)
- **Instant API Generation**: Upload a model file and get REST API endpoints immediately
- **BentoML Integration**: Robust model serving with automatic containerization
- **Deployment Management**: Start, stop, scale, and monitor deployments with full lifecycle control

### 🛡️ **Advanced Schema Management**
- **Dynamic Schema Definition**: Define input/output schemas with comprehensive validation
- **Schema Generation**: Auto-generate schemas from sample data
- **Schema Versioning**: Track schema changes with version control and migration support
- **Format Conversion**: Convert between JSON Schema, OpenAPI, and other formats
- **Template Library**: Pre-built schema templates for common ML use cases

### 📊 **Enterprise-Grade Monitoring & MLOps**
- **Real-time Performance Metrics**: Latency, throughput, success rates, and error tracking with percentile analysis
- **System Health Monitoring**: API server, database, storage, and service health checks with resource usage tracking
- **Prediction Logging**: Comprehensive audit trail of all predictions with metadata and ground truth tracking
- **Alert Management**: Configurable alert rules with severity levels, escalation policies, and automated notifications
- **Model Drift Detection**: Feature drift, data drift, and prediction drift detection using PSI and KS tests
- **Performance Degradation**: Automatic detection of model performance degradation with statistical significance testing
- **A/B Testing**: Built-in A/B testing framework with variant assignment, metrics tracking, and statistical analysis
- **Canary Deployments**: Gradual rollout with health checks, automatic rollback, and traffic splitting
- **Model Versioning**: Compare model versions with performance regression detection
- **Bias & Fairness**: Monitor protected attributes, calculate fairness metrics, and track demographic distributions
- **Model Explainability**: SHAP and LIME explanations with feature importance analysis
- **Data Quality**: Outlier detection, anomaly detection, and data quality metrics
- **Model Lifecycle**: Retraining triggers, job management, and model card generation
- **Governance**: Data lineage tracking, compliance records, and retention policies
- **Analytics Dashboard**: Usage patterns, performance trends, and comprehensive model analytics

### 💻 **Comprehensive Web Interface**
- **Intuitive Dashboard**: Beautiful, responsive web UI with real-time metrics and system health
- **Model Management**: Full CRUD operations with drag-and-drop upload, model details view, and deletion
- **Deployment Console**: Visual deployment management with start/stop controls and status monitoring
- **Live Testing**: Interactive prediction testing with real-time results and schema validation
- **Schema Management**: Complete schema operations including validation, generation, comparison, and format conversion
- **Advanced Monitoring**: Comprehensive monitoring dashboard with performance metrics, drift detection, explainability, data quality, and fairness analysis
- **A/B Testing**: Full A/B testing interface with test creation, management, and metrics tracking
- **Canary Deployments**: Canary deployment management with gradual rollout controls
- **Governance & Compliance**: Data lineage, workflows, compliance records, and retention policies
- **Analytics & Reporting**: Time series analysis, comparative analytics, custom dashboards, and automated reports
- **Model Lifecycle**: Model cards, retraining jobs, and lifecycle management
- **Integrations**: External integrations, webhooks, and sampling configurations
- **Audit Logs**: Complete audit trail viewer with filtering capabilities

### 🏗️ **Production Architecture**
- **FastAPI Backend**: High-performance async API with automatic documentation
- **Database Flexibility**: PostgreSQL for production, SQLite for development/demo
- **Async Operations**: Full async support for database and model operations
- **RESTful APIs**: Comprehensive REST API with OpenAPI documentation
- **Error Handling**: Global exception handling with structured error responses

### 🧪 **Comprehensive Testing**
- **Extensive Test Suite**: 1,366+ tests covering all functionality
- **High Test Coverage**: 56% overall coverage (up from 37%), with key modules at 70-95% coverage
- **Test Isolation**: Proper test isolation using UUIDs to prevent test interference
- **Multiple Test Types**: Unit, integration, API, service, and monitoring tests
- **Test Categories**: Organized by functionality (models, deployments, schemas, monitoring, A/B testing, canary, drift, etc.)
- **Modular Architecture**: Refactored codebase with all files under 500 lines for maintainability
- **Comprehensive Service Tests**: Deployment service (73% coverage), drift routes (95% coverage), alert rules (88% coverage)
- **CI/CD Ready**: Automated testing with coverage reporting
- **Cross-Platform**: Windows, Linux, macOS compatibility

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────┐    ┌─────────────────┐
│   Web UI        │    │  FastAPI     │    │   BentoML       │
│   (HTML/JS)     │◄──►│   Service    │◄──►│   Services      │
└─────────────────┘    └──────────────┘    └─────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │  PostgreSQL      │
                    │  or SQLite       │
                    └──────────────────┘
```

## 📋 Prerequisites

- **Python 3.12+**
- **Poetry** (for dependency management)
- **PostgreSQL** (for production) or **SQLite** (for demo/development)
- **Docker** (optional, for containerized deployment)

## 🛠️ Quick Start

### 🎯 **One-Click Demo** (Recommended for First-Time Users)

**Zero setup required!** Perfect for testing and development.

```bash
# 1. Clone and install
git clone <repository-url>
cd EasyMLOps
poetry install

# 2. One-click demo start
python demo.py
```

**What the demo provides:**
- ✅ SQLite database (no PostgreSQL setup needed)
- ✅ Auto-creates all required directories
- ✅ Opens browser at http://localhost:8000
- ✅ Includes sample models and data
- ✅ Full feature access in minutes

### 🏭 **Production Setup** (PostgreSQL)

For production deployments with enterprise features and scalability.

#### 1. **Install Dependencies**
```bash
git clone <repository-url>
cd EasyMLOps
poetry install
```

#### 2. **Setup PostgreSQL Database**
```sql
-- Create database and user
CREATE DATABASE easymlops;
CREATE USER easymlops_user WITH ENCRYPTED PASSWORD 'your_secure_password';
GRANT ALL PRIVILEGES ON DATABASE easymlops TO easymlops_user;
```

#### 3. **Configure Environment**
```bash
# Copy the example environment file
cp env.example .env

# Edit .env with your configuration:
DB_HOST=localhost
DB_PORT=5432
DB_USER=easymlops_user
DB_PASSWORD=your_secure_password
DB_NAME=easymlops
USE_SQLITE=false
```

#### 4. **Start Production Server**
```bash
# Standard production mode
poetry run python -m app.main

# Custom configuration
poetry run python -m app.main --host 0.0.0.0 --port 8000

# Debug mode with auto-reload
poetry run python -m app.main --debug
```

## 🎮 Usage & Configuration

### **Command-Line Options**

```bash
# Demo mode (SQLite, no database setup)
python -m app.main --demo

# Production mode (PostgreSQL)
python -m app.main

# Custom SQLite for development
python -m app.main --sqlite --db-path my_dev.db

# Custom host and port
python -m app.main --host 0.0.0.0 --port 8080

# Debug mode with auto-reload
python -m app.main --debug

# Disable browser auto-open
python -m app.main --demo --no-browser
```

### **Access Points**

Once running, access:
- **🎛️ Web Interface**: http://localhost:8000
- **📖 API Documentation**: http://localhost:8000/docs
- **📋 Alternative Docs**: http://localhost:8000/redoc
- **💓 Health Check**: http://localhost:8000/health

## 💡 Platform Usage

### **🌐 Web Interface Features**

The web interface provides comprehensive model management:

- **📊 Dashboard**: Real-time statistics and system health overview
- **📤 Model Upload**: Drag-and-drop interface with validation and metadata entry
- **🔧 Schema Management**: Visual schema builder with field validation
- **🚀 Deployment Console**: Deploy, monitor, and manage model services
- **🧪 Live Testing**: Interactive prediction testing with real-time results
- **📈 Monitoring**: Performance metrics, logs, and alert management

### **🔌 REST API Usage**

#### **Upload a Model**
```bash
curl -X POST "http://localhost:8000/api/v1/models/upload" \
  -F "file=@my_model.joblib" \
  -F "name=house_price_predictor" \
  -F "description=Predicts house prices based on features" \
  -F "model_type=regression" \
  -F "framework=sklearn"
```

#### **Define Input Schema**
```bash
curl -X POST "http://localhost:8000/api/v1/schemas/{model_id}/schemas" \
  -H "Content-Type: application/json" \
  -d '{
    "input_schema": {
      "type": "object",
      "properties": {
        "square_feet": {"type": "number", "minimum": 500, "maximum": 10000},
        "bedrooms": {"type": "integer", "minimum": 1, "maximum": 10},
        "bathrooms": {"type": "number", "minimum": 1, "maximum": 10},
        "age": {"type": "integer", "minimum": 0, "maximum": 100}
      },
      "required": ["square_feet", "bedrooms", "bathrooms"]
    }
  }'
```

#### **Deploy Model**
```bash
curl -X POST "http://localhost:8000/api/v1/deployments/" \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "your_model_id",
    "deployment_name": "house_price_api",
    "description": "Production house price prediction API"
  }'
```

#### **Make Predictions**
```bash
# Schema-validated prediction
curl -X POST "http://localhost:8000/api/v1/predict/{deployment_id}" \
  -H "Content-Type: application/json" \
  -d '{
    "square_feet": 2000,
    "bedrooms": 3,
    "bathrooms": 2.5,
    "age": 10
  }'

# Batch predictions
curl -X POST "http://localhost:8000/api/v1/predict/{deployment_id}/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "data": [
      {"square_feet": 2000, "bedrooms": 3, "bathrooms": 2.5, "age": 10},
      {"square_feet": 1500, "bedrooms": 2, "bathrooms": 2, "age": 5}
    ]
  }'

# Probability predictions (for classification)
curl -X POST "http://localhost:8000/api/v1/predict/{deployment_id}/proba" \
  -H "Content-Type: application/json" \
  -d '{"data": [1.0, 2.0, 3.0, 4.0]}'
```

#### **Monitor Performance**
```bash
# Get model performance metrics
curl "http://localhost:8000/api/v1/monitoring/models/{model_id}/performance?start_time=2024-01-01T00:00:00Z&end_time=2024-01-01T23:59:59Z"

# System health status
curl "http://localhost:8000/api/v1/monitoring/health"

# Active alerts
curl "http://localhost:8000/api/v1/monitoring/alerts"
```

## ⚙️ Configuration

### **Database Modes**

| Mode | Database | Use Case | Configuration |
|------|----------|----------|---------------|
| **Demo** | SQLite | Testing, development, demos | `--demo` or `python demo.py` |
| **Development** | SQLite | Custom dev environments | `--sqlite --db-path custom.db` |
| **Production** | PostgreSQL | Production deployments | Default with `.env` configuration |

### **Environment Variables**

#### **PostgreSQL Configuration (Production)**
```bash
# Database
DB_HOST=localhost
DB_PORT=5432
DB_USER=easymlops_user
DB_PASSWORD=your_password
DB_NAME=easymlops
USE_SQLITE=false

# Application
APP_NAME=EasyMLOps
DEBUG=false
HOST=0.0.0.0
PORT=8000

# File Storage
MAX_FILE_SIZE=524288000  # 500MB
MODELS_DIR=models
BENTOS_DIR=bentos
STATIC_DIR=static

# Security
SECRET_KEY=your-secret-key-change-in-production
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Monitoring
ENABLE_METRICS=true
METRICS_PORT=9090
LOG_LEVEL=INFO
```

#### **SQLite Configuration (Demo/Development)**
```bash
USE_SQLITE=true
SQLITE_PATH=demo.db
DEBUG=true
```

## 📁 Project Structure

```
easymlops/
├── app/                     # Core application
│   ├── main.py             # FastAPI application entry point
│   ├── config.py           # Configuration management
│   ├── database.py         # Database connection and session management
│   ├── core/               # Core application factory and routing
│   │   ├── app_factory.py  # Application factory pattern
│   │   └── routes.py       # Route registration
│   ├── models/             # SQLModel database models
│   │   ├── model.py        # Model and deployment models
│   │   └── monitoring/     # Modular monitoring models
│   │       ├── base.py     # Base models and enums
│   │       ├── performance.py
│   │       ├── drift.py
│   │       ├── testing.py
│   │       └── ...         # Domain-specific model modules
│   ├── routes/             # FastAPI route definitions
│   │   ├── models.py       # Model CRUD operations
│   │   ├── deployments.py  # Deployment management
│   │   ├── dynamic/        # Dynamic prediction endpoints (refactored)
│   │   │   ├── prediction_handlers.py
│   │   │   ├── schema_handler.py
│   │   │   └── ...
│   │   ├── schemas.py      # Schema management
│   │   └── monitoring.py   # Monitoring and metrics
│   ├── services/           # Business logic services
│   │   ├── bentoml/        # BentoML integration (refactored)
│   │   │   ├── builders/   # Framework-specific builders
│   │   │   └── ...
│   │   ├── monitoring/     # Monitoring services (refactored)
│   │   │   ├── performance/    # Performance monitoring modules
│   │   │   ├── drift/          # Drift detection modules
│   │   │   ├── degradation/    # Performance degradation modules
│   │   │   ├── ab_testing.py   # A/B testing service
│   │   │   ├── canary.py        # Canary deployment service
│   │   │   ├── fairness.py      # Bias & fairness monitoring
│   │   │   ├── explainability.py
│   │   │   └── ...              # 21 domain-specific services
│   │   ├── schema/         # Schema services (refactored)
│   │   │   ├── service.py
│   │   │   ├── validation.py
│   │   │   └── ...
│   │   ├── deployment_service.py
│   │   └── monitoring_service.py  # Facade pattern
│   ├── schemas/            # Pydantic schemas
│   │   ├── model.py        # Model and deployment schemas
│   │   └── monitoring/     # Modular monitoring schemas
│   │       ├── base.py
│   │       ├── alerts.py
│   │       └── ...         # Domain-specific schema modules
│   └── utils/              # Utility functions
│       └── model_utils/    # Model utilities (refactored)
│           └── frameworks/ # Framework-specific detectors
│               ├── detector.py
│               ├── sklearn_detector.py
│               └── ...
├── tests/                  # Comprehensive test suite (1,366+ tests, 56% coverage)
│   ├── conftest.py         # Pytest configuration and shared fixtures
│   ├── fixtures/            # Reusable test fixtures
│   │   ├── database.py     # Database fixtures with UUID-based isolation
│   │   └── services.py     # Service fixtures
│   ├── test_services/      # Service layer tests (refactored)
│   │   ├── monitoring/     # Monitoring service tests
│   │   │   ├── test_ab_testing_comprehensive.py
│   │   │   ├── test_ab_testing_private_methods.py
│   │   │   ├── test_alert_rules.py
│   │   │   ├── test_alert_management_comprehensive.py
│   │   │   └── ...         # Domain-specific test modules
│   │   ├── test_deployment_service_comprehensive.py  # 73% coverage
│   │   └── ...             # Additional service tests
│   ├── test_routes/        # API route tests
│   │   ├── monitoring/    # Monitoring route tests
│   │   │   ├── test_drift_comprehensive.py  # 95% coverage
│   │   │   ├── test_alerts.py
│   │   │   ├── test_ab_testing.py
│   │   │   └── ...         # Domain-specific route tests
│   │   ├── dynamic/        # Dynamic route tests
│   │   ├── schemas/        # Schema route tests
│   │   └── ...             # Additional route tests
│   ├── test_core/          # Core application tests
│   ├── test_models/        # Database model tests
│   ├── test_utils/         # Utility tests
│   └── ...                 # Additional test modules
├── static/                 # Web interface files
│   ├── index.html          # Main web interface
│   ├── css/                # Stylesheets
│   └── js/                 # JavaScript functionality
├── models/                 # Uploaded models storage
├── bentos/                 # BentoML services storage
├── logs/                   # Application logs
├── demo.py                 # One-click demo launcher
├── run_tests.py           # Advanced test runner
├── pyproject.toml         # Dependencies and project config
└── README.md              # This file
```

## 🧪 Development & Testing

### **Advanced Test Runner**
```bash
# Run all tests (1,366+ tests, 56% coverage)
python run_tests.py

# Run specific test categories
python run_tests.py --unit          # Unit tests only
python run_tests.py --api           # API tests only
python run_tests.py --database      # Database tests only
python run_tests.py --monitoring    # Monitoring tests only
python run_tests.py --deployment    # Deployment tests only
python run_tests.py --service       # Service layer tests only
python run_tests.py --integration   # Integration tests only
python run_tests.py --config        # Configuration tests only

# Run with coverage
python run_tests.py --coverage

# Run specific test file
python run_tests.py --file models   # runs test_models.py
python run_tests.py --file monitoring_performance  # runs test_monitoring_performance.py

# Fast test suite (skip slow tests)
python run_tests.py --fast

# Run tests in parallel
python run_tests.py --parallel 4

# Stop on first failure
python run_tests.py --failfast

# Special commands
python run_tests.py quick           # Quick test suite
python run_tests.py ci              # CI/CD test suite
python run_tests.py check           # Check test environment setup
```

### **Development Commands**
```bash
# Code formatting
poetry run black .
poetry run isort .

# Linting
poetry run flake8
poetry run mypy .

# Start in debug mode
poetry run python -m app.main --debug

# Watch for changes (with auto-reload)
poetry run python -m app.main --debug --reload
```

## 📦 Supported Model Formats

| Framework | File Extensions | Features |
|-----------|----------------|----------|
| **Scikit-learn** | `.joblib`, `.pkl` | Classification, regression, clustering |
| **TensorFlow** | `.h5`, `.pb`, `.keras` | Deep learning models |
| **PyTorch** | `.pt`, `.pth` | Neural networks |
| **XGBoost** | `.ubj`, `.json` | Gradient boosting |
| **LightGBM** | `.txt`, `.model` | Gradient boosting |
| **H2O** | `.mojo`, `.pojo` | AutoML models |
| **ONNX** | `.onnx` | Cross-platform models |

## 🐳 Container Deployment

### **Docker**
```bash
# Build image
docker build -t easymlops .

# Run with PostgreSQL
docker run -d \
  -p 8000:8000 \
  -e DB_HOST=your_db_host \
  -e DB_USER=your_db_user \
  -e DB_PASSWORD=your_db_password \
  easymlops

# Run in demo mode (SQLite)
docker run -d -p 8000:8000 easymlops --demo
```

### **Docker Compose**
```bash
# Start with PostgreSQL
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### **Kubernetes**
```bash
# Deploy to cluster
kubectl apply -f k8s/

# Check deployment
kubectl get pods -l app=easymlops

# Scale deployment
kubectl scale deployment easymlops --replicas=3
```

## 📊 Monitoring & Observability

### **Comprehensive Monitoring Features**

The platform includes 21 specialized monitoring services organized in a modular architecture, all accessible through the web interface:

- **📈 Performance Monitoring**: Request latency, throughput, error rates with percentile tracking (p50, p95, p99)
- **🏥 System Health**: Real-time health checks for all system components with resource usage tracking
- **📝 Prediction Logging**: Complete audit trail with request/response logging and ground truth tracking
- **🚨 Alert Management**: Configurable alert rules with severity levels, escalation policies, and notifications
- **📊 Analytics Dashboard**: Usage patterns, performance trends, and comprehensive model insights
- **🔍 Error Tracking**: Structured error logging with contextual information
- **🌊 Drift Detection**: Feature drift, data drift, and prediction drift using PSI and KS tests
- **📉 Performance Degradation**: Automatic detection with statistical significance testing
- **🧪 A/B Testing**: Built-in framework with variant assignment and statistical analysis
- **🦅 Canary Deployments**: Gradual rollout with automatic rollback capabilities
- **⚖️ Bias & Fairness**: Protected attribute monitoring and fairness metrics
- **🔬 Model Explainability**: SHAP and LIME explanations with feature importance
- **✅ Data Quality**: Outlier detection, anomaly detection, and quality metrics
- **🔄 Model Lifecycle**: Retraining triggers, job management, and model cards
- **📋 Governance**: Data lineage, compliance records, and retention policies
- **🔗 Integration**: Webhooks, external integrations, and sampling configurations
- **📜 Audit Logging**: Comprehensive audit trail for compliance

### **Monitoring Endpoints** (All accessible via Web UI)

**Core Monitoring:**
```bash
# System health overview
GET /api/v1/monitoring/health

# Dashboard metrics
GET /api/v1/monitoring/dashboard

# Model performance metrics
GET /api/v1/monitoring/models/{model_id}/performance

# Prediction logs
GET /api/v1/monitoring/models/{model_id}/predictions/logs

# Resource usage
GET /api/v1/monitoring/models/{model_id}/resources

# Confidence metrics
GET /api/v1/monitoring/models/{model_id}/confidence

# Deployment summary
GET /api/v1/monitoring/deployments/{deployment_id}/summary
```

**Drift Detection:**
```bash
# Feature drift
POST /api/v1/monitoring/models/{model_id}/drift/feature

# Data drift
POST /api/v1/monitoring/models/{model_id}/drift/data

# Prediction drift
POST /api/v1/monitoring/models/{model_id}/drift/prediction

# Drift history
GET /api/v1/monitoring/models/{model_id}/drift
```

**A/B Testing:**
```bash
# Create A/B test
POST /api/v1/monitoring/ab-tests

# Start/stop test
POST /api/v1/monitoring/ab-tests/{test_id}/start
POST /api/v1/monitoring/ab-tests/{test_id}/stop

# Get metrics
GET /api/v1/monitoring/ab-tests/{test_id}/metrics

# Assign variant
POST /api/v1/monitoring/ab-tests/{test_id}/assign
```

**Canary Deployments:**
```bash
# Create canary
POST /api/v1/monitoring/canary

# Manage rollout
POST /api/v1/monitoring/canary/{canary_id}/start
POST /api/v1/monitoring/canary/{canary_id}/advance
POST /api/v1/monitoring/canary/{canary_id}/rollback

# Get status
GET /api/v1/monitoring/canary/{canary_id}/metrics
GET /api/v1/monitoring/canary/{canary_id}/health
```

**Alerts:**
```bash
# Get alerts
GET /api/v1/monitoring/alerts

# Manage alerts
POST /api/v1/monitoring/alerts/{alert_id}/resolve
POST /api/v1/monitoring/alerts/{alert_id}/acknowledge

# Alert rules
POST /api/v1/monitoring/alert-rules
POST /api/v1/monitoring/alerts/check
```

**Explainability:**
```bash
# SHAP explanations
POST /api/v1/monitoring/models/{model_id}/explain/shap

# LIME explanations
POST /api/v1/monitoring/models/{model_id}/explain/lime

# Feature importance
GET /api/v1/monitoring/models/{model_id}/explain/importance
```

**Data Quality & Fairness:**
```bash
# Data quality
POST /api/v1/monitoring/models/{model_id}/data-quality/metrics
POST /api/v1/monitoring/models/{model_id}/data-quality/outliers
POST /api/v1/monitoring/models/{model_id}/data-quality/anomaly

# Fairness
POST /api/v1/monitoring/models/{model_id}/fairness/metrics
GET /api/v1/monitoring/models/{model_id}/fairness/demographics
```

**Additional Features:**
```bash
# Performance degradation
POST /api/v1/monitoring/models/{model_id}/degradation/detect

# Baseline management
POST /api/v1/monitoring/models/{model_id}/baseline
GET /api/v1/monitoring/models/{model_id}/baseline

# Model lifecycle
GET /api/v1/monitoring/models/{model_id}/card
POST /api/v1/monitoring/models/{model_id}/card/generate

# Analytics
POST /api/v1/monitoring/analytics/time-series

# Audit logs
GET /api/v1/monitoring/audit
```

### **Health Check Endpoints**
- `GET /health` - Basic health status
- `GET /api/v1/monitoring/health` - Comprehensive system health with component details

## 🔒 Security Features

- **🔐 Environment-based Configuration**: Secure credential management
- **✅ Input Validation**: Comprehensive request validation and sanitization
- **🛡️ CORS Protection**: Configurable cross-origin policies
- **📁 Secure File Upload**: File type, size, and content validation
- **💾 Database Security**: Connection pooling and SQL injection protection
- **🚫 Error Handling**: Secure error responses without sensitive data exposure
- **🔑 Authentication Ready**: JWT token support for future auth implementation

## 🤝 Contributing

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/amazing-feature`)
3. **Make your changes** with comprehensive tests
4. **Run the test suite** (`python run_tests.py`)
5. **Submit a pull request**

### **Development Guidelines**
- Follow the existing code style and patterns
- Add tests for all new features
- Update documentation for new functionality
- Ensure all tests pass before submitting PR

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support & Troubleshooting

### **Quick Help**
- **🌐 API Documentation**: Visit `/docs` when running the application
- **💓 Health Status**: Check `/health` endpoint for system status
- **📋 Application Logs**: Review `logs/` directory for detailed information

### **Common Issues & Solutions**

| Problem | Solution |
|---------|----------|
| **Demo won't start** | Run `poetry install` then `python demo.py` |
| **PostgreSQL connection errors** | Check `.env` configuration and database access |
| **Port already in use** | Use `--port 8080` or kill existing process |
| **File upload fails** | Check `MODELS_DIR` permissions and `MAX_FILE_SIZE` setting |
| **Tests failing** | Run `python run_tests.py --database` to check database setup |
| **BentoML service errors** | Check `bentos/` directory permissions and disk space |
| **Schema validation errors** | Verify schema format and required fields |

### **Getting Started Checklist**

#### **✅ For Demo/Testing:**
1. `git clone <repo> && cd EasyMLOps`
2. `poetry install`
3. `python demo.py`
4. Open http://localhost:8000
5. Upload a model and test predictions

#### **✅ For Production:**
1. Set up PostgreSQL database
2. Configure `.env` file with database credentials
3. `poetry run python -m app.main`
4. Monitor via `/health` endpoint
5. Configure monitoring and alerts

## 🗺️ Roadmap

### **Completed Features** ✅
- **🔄 Model Versioning**: Complete model lifecycle management with version comparison
- **🧪 A/B Testing**: Built-in A/B testing framework with statistical analysis and full UI
- **🦅 Canary Deployments**: Gradual rollout with automatic rollback and full UI
- **🌊 Drift Detection**: Feature, data, and prediction drift detection with UI controls
- **📉 Performance Degradation**: Automatic detection with statistical testing and UI
- **⚖️ Bias & Fairness**: Protected attribute monitoring and fairness metrics with UI
- **🔬 Model Explainability**: SHAP and LIME explanations with UI
- **✅ Data Quality**: Outlier and anomaly detection with UI
- **📋 Governance**: Data lineage and compliance tracking with UI structure
- **📊 Advanced Dashboards**: Comprehensive monitoring dashboards with full UI
- **💻 Frontend Implementation**: 85%+ backend endpoint coverage with comprehensive web interface
- **📝 Schema Management**: Full schema operations (validate, generate, compare, convert) with dedicated UI
- **📊 Analytics**: Time series analysis, comparative analytics, dashboards, and reports with UI
- **🔄 Lifecycle Management**: Model cards, retraining jobs with UI
- **🔗 Integrations**: External integrations and webhooks with UI structure
- **📜 Audit Logging**: Complete audit trail viewer with filtering
- **🧪 Test Coverage**: Comprehensive test suite with 56% overall coverage (up from 37%)
- **🔧 Test Quality**: Fixed test isolation issues using UUID-based fixtures
- **📈 Service Testing**: Deployment service (73% coverage), drift routes (95% coverage), alert rules (88% coverage)

### **Planned Features**
- **☁️ Multi-Cloud**: Support for AWS, GCP, Azure deployments
- **👥 Multi-User**: Authentication, authorization, and role-based access control
- **🔄 Auto-Retraining**: Enhanced automated model retraining workflows with complete UI
- **🔗 Integration Hub**: Additional connectors for popular ML platforms and tools
- **📱 Mobile App**: Native mobile application for monitoring
- **🌐 Multi-Region**: Support for multi-region deployments
- **📊 Enhanced Analytics**: Complete forms for comparative analytics, dashboards, and reports
- **📋 Governance UI**: Detailed forms for lineage, workflows, compliance, and retention policies
- **🔔 Alert Management**: Complete alert rule management, notification channels, and escalation UI
- **📈 Deployment Management**: Full deployment update, delete, start, scale, and metrics UI
- **🔄 Schema Versioning**: Complete schema versioning and management UI

### **Performance Goals**
- **⚡ Sub-100ms**: Prediction latency optimization
- **📈 1000+ RPS**: High-throughput model serving
- **🏗️ Horizontal Scaling**: Kubernetes-native auto-scaling
- **📊 Advanced Monitoring**: Real-time performance dashboards

---

**🚀 Ready to deploy your ML models in production? Start with `python demo.py` and experience EasyMLOps in action!** 