# EasyMLOps - Data Scientist Empowerment Platform

A powerful ML Operations platform that allows data scientists to deploy machine learning models with no-code, production-ready API endpoints, comprehensive monitoring, and controlled inputs/outputs.

## 🚀 Features

- **No-Code Model Deployment**: Upload any ML model and get production-ready APIs instantly
- **Multi-Framework Support**: Compatible with scikit-learn, TensorFlow, PyTorch, H2O, XGBoost, LightGBM, and more
- **Dynamic Schema Definition**: Define input/output schemas with validation
- **Comprehensive Monitoring**: Track performance, latency, and model health
- **Production Ready**: Built with FastAPI, PostgreSQL, and BentoML
- **Container Support**: Docker and Kubernetes ready
- **Web Interface**: User-friendly HTML interface for model management

## 🏗️ Architecture

```
Frontend (HTML) → FastAPI Service → BentoML → PostgreSQL
                      ↓
               Docker/Kubernetes
```

## 📋 Prerequisites

- Python 3.12+
- PostgreSQL database
- Poetry (for dependency management)

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd EasyMLOps
```

### 2. Install Dependencies
```bash
poetry install
```

### 3. Configure Environment
```bash
# Copy the example environment file
cp env.example .env

# Edit the .env file with your configuration
# Update database credentials, secret keys, etc.
```

### 4. Setup Database
Ensure PostgreSQL is running and create the database:
```sql
CREATE DATABASE easymlops;
```

### 5. Run the Application
```bash
# Development mode
poetry run python -m app.main

# Or using uvicorn directly
poetry run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

The application will automatically:
- Create necessary directories
- Set up database tables
- Open your browser to the web interface
- Start the API server

## 📖 Usage

### Web Interface
Visit `http://localhost:8000` to access the web interface where you can:
- Upload ML models
- Define input/output schemas
- Configure model parameters
- Monitor deployments
- Test model predictions

### API Documentation
- **Interactive Docs**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

### Health Checks
- **Basic Health**: `GET /health`
- **Detailed Health**: `GET /health/detailed`

## 🔧 Configuration

All configuration is managed through environment variables. Key settings include:

### Database Configuration
```bash
POSTGRES_SERVER=localhost
POSTGRES_PORT=5432
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_password
POSTGRES_DB=easymlops
```

### Application Settings
```bash
APP_NAME=EasyMLOps
DEBUG=false
HOST=0.0.0.0
PORT=8000
```

### File Storage
```bash
MODELS_DIR=models
BENTOS_DIR=bentos
STATIC_DIR=static
MAX_FILE_SIZE=524288000  # 500MB
```

## 📁 Project Structure

```
easymlops/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application entry point
│   ├── config.py            # Configuration management
│   ├── database.py          # Database connection setup
│   ├── schemas/             # Pydantic schemas
│   ├── models/              # SQLAlchemy models
│   ├── routes/              # FastAPI routes
│   ├── services/            # Business logic
│   └── utils/               # Utility functions
├── static/                  # HTML UI files
├── models/                  # Uploaded models storage
├── bentos/                  # BentoML services storage
├── logs/                    # Application logs
├── pyproject.toml          # Project dependencies
├── env.example             # Environment configuration template
└── README.md
```

## 🧪 Development

### Running Tests
```bash
poetry run pytest
```

### Code Formatting
```bash
poetry run black .
poetry run isort .
```

### Linting
```bash
poetry run flake8
poetry run mypy .
```

## 📦 Supported Model Formats

- **Pickle**: `.pkl`, `.joblib`
- **TensorFlow**: `.h5`, `.pb`
- **ONNX**: `.onnx`
- **JSON**: Model metadata and configurations

## 🐳 Docker Deployment

### Build Docker Image
```bash
docker build -t easymlops .
```

### Run with Docker Compose
```bash
docker-compose up -d
```

## ☸️ Kubernetes Deployment

```bash
kubectl apply -f k8s/
```

## 📊 Monitoring

The platform includes comprehensive monitoring:
- Request/response logging
- Performance metrics
- Model health checks
- Database connection monitoring
- System resource tracking

Logs are stored in the `logs/` directory with JSON formatting for structured analysis.

## 🔒 Security

- Environment-based configuration
- Request validation and sanitization
- CORS protection
- Secure file upload handling
- Database connection pooling
- Error handling and logging

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🆘 Support

For support and questions:
- Check the documentation at `/docs`
- Review the health endpoints at `/health`
- Check application logs in the `logs/` directory

## 🗺️ Roadmap

- [ ] Model versioning and rollback
- [ ] A/B testing capabilities
- [ ] Advanced monitoring dashboards
- [ ] Integration with cloud providers
- [ ] Multi-user authentication
- [ ] Model performance analytics 