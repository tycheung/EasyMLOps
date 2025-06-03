# EasyMLOps - Data Scientist Empowerment Platform

A powerful ML Operations platform that allows data scientists to deploy machine learning models with no-code, production-ready API endpoints, comprehensive monitoring, and controlled inputs/outputs.

## 🚀 Features

- **No-Code Model Deployment**: Upload any ML model and get production-ready APIs instantly
- **Multi-Framework Support**: Compatible with scikit-learn, TensorFlow, PyTorch, H2O, XGBoost, LightGBM, and more
- **Dynamic Schema Definition**: Define input/output schemas with validation
- **Comprehensive Monitoring**: Track performance, latency, and model health
- **Production Ready**: Built with FastAPI, PostgreSQL, and BentoML
- **Demo Mode**: SQLite-based demo requiring no external database setup
- **Container Support**: Docker and Kubernetes ready
- **Web Interface**: User-friendly HTML interface for model management

## 🏗️ Architecture

```
Frontend (HTML) → FastAPI Service → BentoML → PostgreSQL/SQLite
                      ↓
               Docker/Kubernetes
```

## 📋 Prerequisites

- Python 3.12+
- Poetry (for dependency management)
- **For Production**: PostgreSQL database
- **For Demo**: No additional dependencies

## 🛠️ Quick Start

### 🎯 Demo Mode (Recommended for First-Time Users)

**No database setup required!** Perfect for testing and development.

```bash
# 1. Clone and install
git clone <repository-url>
cd EasyMLOps
poetry install

# 2. One-click demo start
python demo.py
```

That's it! The demo will:
- ✅ Use SQLite (no PostgreSQL required)
- ✅ Auto-create demo database and directories
- ✅ Open your browser to http://localhost:8000
- ✅ Provide sample data and examples

### 🏭 Production Setup (PostgreSQL)

For production deployments with full scalability and features.

#### 1. Install Dependencies
```bash
git clone <repository-url>
cd EasyMLOps
poetry install
```

#### 2. Setup PostgreSQL Database
```sql
-- Create database and user
CREATE DATABASE easymlops;
CREATE USER easymlops_user WITH ENCRYPTED PASSWORD 'your_secure_password';
GRANT ALL PRIVILEGES ON DATABASE easymlops TO easymlops_user;
```

#### 3. Configure Environment
```bash
# Copy the example environment file
cp env.example .env

# Edit .env with your PostgreSQL configuration:
# DB_HOST=localhost
# DB_PORT=5432
# DB_USER=easymlops_user
# DB_PASSWORD=your_secure_password
# DB_NAME=easymlops
```

#### 4. Start Production Server
```bash
# Standard production mode
poetry run python -m app.main

# With custom configuration
poetry run python -m app.main --host 0.0.0.0 --port 8000

# Debug mode with auto-reload
poetry run python -m app.main --debug
```

## 🎮 Usage Options

### Command-Line Interface

```bash
# Demo mode (SQLite, no database setup required)
python -m app.main --demo

# Production mode (requires PostgreSQL setup)
python -m app.main

# Custom SQLite database for development
python -m app.main --sqlite --db-path my_test.db

# Custom host and port
python -m app.main --host 127.0.0.1 --port 8080

# Debug mode with auto-reload
python -m app.main --debug

# Disable browser auto-open
python -m app.main --demo --no-browser
```

### Available Endpoints

Once running, access:
- **Web Interface**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

## 📖 Using the Platform

### Web Interface
Visit the web interface to:
- 📤 Upload ML models (`.joblib`, `.pkl`, `.h5`, `.onnx`, etc.)
- 🔧 Define input/output schemas with validation
- ⚙️ Configure model parameters and settings
- 📊 Monitor deployments and performance
- 🧪 Test model predictions with live data

### API Usage

#### Upload a Model
```bash
curl -X POST "http://localhost:8000/api/v1/models/upload" \
  -F "file=@my_model.joblib" \
  -F "name=my_classifier" \
  -F "description=My ML classifier" \
  -F "model_type=classification" \
  -F "framework=sklearn"
```

#### Make Predictions
```bash
curl -X POST "http://localhost:8000/api/v1/models/{model_id}/predict" \
  -H "Content-Type: application/json" \
  -d '{"feature1": 0.5, "feature2": "value"}'
```

#### List Models
```bash
curl "http://localhost:8000/api/v1/models"
```

## 🔧 Configuration

### Database Modes

| Mode | Database | Use Case | Configuration |
|------|----------|----------|---------------|
| **Demo** | SQLite | Testing, development, demos | `--demo` or `python demo.py` |
| **Development** | SQLite | Custom dev environments | `--sqlite --db-path custom.db` |
| **Production** | PostgreSQL | Production deployments | Default with `.env` configuration |

### Environment Variables

#### PostgreSQL Configuration (Production)
```bash
DB_HOST=localhost
DB_PORT=5432
DB_USER=easymlops_user
DB_PASSWORD=your_password
DB_NAME=easymlops
```

#### SQLite Configuration (Demo/Development)
```bash
USE_SQLITE=true
SQLITE_PATH=demo.db
```

#### Application Settings
```bash
APP_NAME=EasyMLOps
DEBUG=false
HOST=0.0.0.0
PORT=8000
MAX_FILE_SIZE=524288000  # 500MB
```

#### File Storage
```bash
MODELS_DIR=models
BENTOS_DIR=bentos
STATIC_DIR=static
```

## 📁 Project Structure

```
easymlops/
├── app/
│   ├── main.py              # FastAPI application entry point
│   ├── config.py            # Configuration management
│   ├── database.py          # Database connection setup
│   ├── models/              # SQLAlchemy/SQLModel models
│   ├── routes/              # FastAPI routes
│   ├── services/            # Business logic services
│   ├── schemas/             # Pydantic schemas
│   └── utils/               # Utility functions
├── tests/                   # Test suite
├── static/                  # HTML UI files
├── models/                  # Uploaded models storage
├── bentos/                  # BentoML services storage
├── logs/                    # Application logs
├── demo.py                  # One-click demo launcher
├── run_tests.py            # Test runner script
├── pyproject.toml          # Dependencies and project config
└── README.md
```

## 🧪 Development & Testing

### Running Tests
```bash
# Run all tests
python run_tests.py

# Run specific test categories
python run_tests.py --unit          # Unit tests only
python run_tests.py --api           # API tests only
python run_tests.py --database      # Database tests only

# Run with coverage
python run_tests.py --coverage

# Quick test suite
python run_tests.py --fast
```

### Development Commands
```bash
# Code formatting
poetry run black .
poetry run isort .

# Linting
poetry run flake8
poetry run mypy .

# Start in debug mode
poetry run python -m app.main --debug
```

## 📦 Supported Model Formats

- **Scikit-learn**: `.joblib`, `.pkl`
- **TensorFlow**: `.h5`, `.pb`, `.keras`
- **PyTorch**: `.pt`, `.pth`
- **ONNX**: `.onnx`
- **XGBoost**: `.ubj`, `.json`
- **LightGBM**: `.txt`, `.model`
- **H2O**: `.mojo`, `.pojo`

## 🐳 Docker Deployment

### Build and Run
```bash
# Build image
docker build -t easymlops .

# Run with Docker Compose
docker-compose up -d

# Manual run with PostgreSQL
docker run -d \
  -p 8000:8000 \
  -e DB_HOST=your_db_host \
  -e DB_USER=your_db_user \
  -e DB_PASSWORD=your_db_password \
  easymlops

# Run in demo mode (SQLite)
docker run -d -p 8000:8000 easymlops --demo
```

## ☸️ Kubernetes Deployment

```bash
# Apply Kubernetes manifests
kubectl apply -f k8s/

# Check deployment status
kubectl get pods -l app=easymlops
```

## 📊 Monitoring & Observability

The platform includes comprehensive monitoring:

- **Request/Response Logging**: All API interactions tracked
- **Performance Metrics**: Latency, throughput, error rates
- **Model Health Checks**: Automated model availability monitoring  
- **Database Connection Monitoring**: Connection pool and query performance
- **System Resource Tracking**: CPU, memory, disk usage
- **Structured Logging**: JSON-formatted logs in `logs/` directory

### Health Endpoints
- `GET /health` - Basic health status
- `GET /health/detailed` - Comprehensive system health

## 🔒 Security Features

- **Environment-based Configuration**: Secure credential management
- **Request Validation**: Input sanitization and validation
- **CORS Protection**: Configurable cross-origin policies
- **Secure File Upload**: File type and size validation
- **Database Connection Pooling**: Secure and efficient DB access
- **Comprehensive Error Handling**: No sensitive data exposure

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes with tests
4. Run the test suite (`python run_tests.py`)
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

### Documentation & Help
- **API Documentation**: Visit `/docs` when running
- **Health Status**: Check `/health` endpoint
- **Application Logs**: Review `logs/` directory

### Quick Troubleshooting

| Problem | Solution |
|---------|----------|
| **Demo won't start** | Run `poetry install` then `python demo.py` |
| **PostgreSQL errors** | Check `.env` configuration and database access |
| **Port already in use** | Use `--port 8080` or different port |
| **File upload fails** | Check `MODELS_DIR` permissions and `MAX_FILE_SIZE` |
| **Tests failing** | Run `python run_tests.py --database` to check test database |

### Getting Started Checklist

✅ **For Demo/Testing:**
1. `git clone <repo> && cd EasyMLOps`
2. `poetry install`
3. `python demo.py`
4. Open http://localhost:8000

✅ **For Production:**
1. Set up PostgreSQL database
2. Configure `.env` file
3. `poetry run python -m app.main`
4. Monitor via `/health` endpoint

## 🗺️ Roadmap

- [ ] Model versioning and rollback capabilities
- [ ] A/B testing and canary deployments
- [ ] Advanced monitoring dashboards
- [ ] Multi-cloud provider integration
- [ ] Multi-user authentication and RBAC
- [ ] Real-time model performance analytics
- [ ] Automated model retraining workflows 