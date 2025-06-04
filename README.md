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

### 📊 **Enterprise-Grade Monitoring**
- **Real-time Performance Metrics**: Latency, throughput, success rates, and error tracking
- **System Health Monitoring**: API server, database, storage, and service health checks
- **Prediction Logging**: Comprehensive audit trail of all predictions with metadata
- **Alert Management**: Configurable alerts with severity levels and automated notifications
- **Analytics Dashboard**: Usage patterns, performance trends, and model analytics

### 💻 **Modern Web Interface**
- **Intuitive Dashboard**: Beautiful, responsive web UI for model management
- **Model Upload**: Drag-and-drop model upload with validation
- **Live Testing**: Interactive prediction testing with real-time results
- **Deployment Console**: Visual deployment management and monitoring
- **Schema Builder**: Visual schema editor with live validation

### 🏗️ **Production Architecture**
- **FastAPI Backend**: High-performance async API with automatic documentation
- **Database Flexibility**: PostgreSQL for production, SQLite for development/demo
- **Async Operations**: Full async support for database and model operations
- **RESTful APIs**: Comprehensive REST API with OpenAPI documentation
- **Error Handling**: Global exception handling with structured error responses

### 🧪 **Comprehensive Testing**
- **100% Test Coverage**: Extensive test suite with 363+ passing tests
- **Multiple Test Types**: Unit, integration, API, service, and monitoring tests
- **Test Categories**: Organized by functionality (models, deployments, schemas, monitoring)
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
│   ├── models/             # SQLModel database models
│   │   ├── model.py        # Model and deployment models
│   │   └── monitoring.py   # Monitoring and logging models
│   ├── routes/             # FastAPI route definitions
│   │   ├── models.py       # Model CRUD operations
│   │   ├── deployments.py  # Deployment management
│   │   ├── dynamic.py      # Dynamic prediction endpoints
│   │   ├── schemas.py      # Schema management
│   │   └── monitoring.py   # Monitoring and metrics
│   ├── services/           # Business logic services
│   │   ├── bentoml_service.py     # BentoML integration
│   │   ├── deployment_service.py  # Deployment management
│   │   ├── schema_service.py      # Schema operations
│   │   └── monitoring_service.py  # Monitoring and analytics
│   ├── schemas/            # Pydantic schemas
│   │   ├── model.py        # Model and deployment schemas
│   │   └── monitoring.py   # Monitoring schemas
│   └── utils/              # Utility functions
├── tests/                  # Comprehensive test suite (363+ tests)
│   ├── test_models.py      # Model management tests
│   ├── test_deployments.py # Deployment tests
│   ├── test_schemas.py     # Schema management tests
│   ├── test_monitoring.py  # Monitoring tests
│   └── test_services_comprehensive.py  # Service layer tests
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
# Run all tests (363+ tests)
python run_tests.py

# Run specific test categories
python run_tests.py --unit          # Unit tests only
python run_tests.py --api           # API tests only
python run_tests.py --database      # Database tests only
python run_tests.py --monitoring    # Monitoring tests only
python run_tests.py --deployment    # Deployment tests only
python run_tests.py --service       # Service layer tests only

# Run with coverage
python run_tests.py --coverage

# Run specific test file
python run_tests.py --file models   # runs test_models.py

# Fast test suite (skip slow tests)
python run_tests.py --fast

# Run tests in parallel
python run_tests.py --parallel 4

# Stop on first failure
python run_tests.py --failfast
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

- **📈 Performance Metrics**: Request latency, throughput, error rates with percentile tracking
- **🏥 System Health**: Real-time health checks for all system components
- **📝 Prediction Logging**: Complete audit trail with request/response logging
- **🚨 Alert Management**: Configurable alerts with severity levels and notifications
- **📊 Analytics Dashboard**: Usage patterns, performance trends, and model insights
- **🔍 Error Tracking**: Structured error logging with contextual information

### **Monitoring Endpoints**
```bash
# System health overview
GET /api/v1/monitoring/health

# Model performance metrics
GET /api/v1/monitoring/models/{model_id}/performance

# Active alerts
GET /api/v1/monitoring/alerts

# Deployment metrics
GET /api/v1/deployments/{deployment_id}/metrics
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
5. **Ensure 100% test coverage** (`python run_tests.py --coverage`)
6. **Submit a pull request**

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

### **Planned Features**
- **🔄 Model Versioning**: Complete model lifecycle management with rollback capabilities
- **🧪 A/B Testing**: Canary deployments and traffic splitting
- **📊 Advanced Dashboards**: Custom monitoring dashboards with visualization
- **☁️ Multi-Cloud**: Support for AWS, GCP, Azure deployments
- **👥 Multi-User**: Authentication, authorization, and role-based access control
- **🔄 Auto-Retraining**: Automated model retraining workflows
- **📈 Advanced Analytics**: Real-time model drift detection and performance analytics
- **🔗 Integration Hub**: Connectors for popular ML platforms and tools

### **Performance Goals**
- **⚡ Sub-100ms**: Prediction latency optimization
- **📈 1000+ RPS**: High-throughput model serving
- **🏗️ Horizontal Scaling**: Kubernetes-native auto-scaling
- **📊 Advanced Monitoring**: Real-time performance dashboards

---

**🚀 Ready to deploy your ML models in production? Start with `python demo.py` and experience EasyMLOps in action!** 