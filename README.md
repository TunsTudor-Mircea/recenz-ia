# RecenzIA - AI-Powered Review Analysis Platform

[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15+-blue.svg)](https://www.postgresql.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

AI-powered sentiment analysis platform for Romanian e-commerce reviews with automated web scraping and comprehensive analytics.

## 🌟 Features

### Core Functionality
- 🤖 **Dual ML Models**: RoBERT (Romanian BERT) and XGBoost for sentiment analysis
- 🕷️ **Automated Web Scraping**: Extract reviews from eMAG using Selenium
- 📊 **Advanced Analytics**: Sentiment trends, rating distributions, and product insights
- 🔐 **Secure Authentication**: JWT-based user authentication
- 📈 **Real-time Monitoring**: Health checks and application metrics
- 🚀 **Production-Ready**: Docker-based deployment with Celery background processing

### Security & Monitoring
- Input validation and sanitization
- Security headers (CSP, HSTS, X-Frame-Options)
- Health check endpoints for monitoring
- WebSocket support for real-time job updates

## 🚀 Quick Start

### Prerequisites

- Docker & Docker Compose
- ~ 8GB RAM

### 1. Clone and Setup

```bash
git clone https://github.com/TunsTudor-Mircea/recenz-ia.git
cd recenz-ia
```

### 2. Prepare ML Models

Copy the trained ML models to the `ml-models` directory:
- RoBERT model files → `ml-models/robert/v1/`
- XGBoost model files → `ml-models/xgboost/v1/`

Required XGBoost files:
- `xgboost_model.joblib`
- `preprocessor.joblib`
- `feature_selector.joblib`

### 3. Start Services

```bash
# Start all services (PostgreSQL, Redis, Backend, Celery)
docker-compose up -d

# Check logs
docker-compose logs -f backend

# Verify services are running
docker-compose ps
```

### 4. Access the Application

- **Frontend**: http://localhost:3000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/api/v1/monitoring/health

## 🏗️ Architecture

The application consists of several components:

- **FastAPI Backend**: RESTful API server
- **PostgreSQL**: Database for users, reviews, and jobs
- **Redis**: Message broker and cache
- **Celery Worker**: Background task processing
- **Selenium**: Web scraping with headless Chrome
- **ML Models**: RoBERT and XGBoost for sentiment analysis

## 📚 API Endpoints

### Authentication
- `POST /api/v1/auth/register` - Register new user
- `POST /api/v1/auth/login` - Login and receive JWT token

### Reviews
- `GET /api/v1/reviews/` - List reviews (with filtering)
- `POST /api/v1/reviews/` - Create manual review
- `GET /api/v1/reviews/{id}` - Get specific review
- `PUT /api/v1/reviews/{id}` - Update review
- `DELETE /api/v1/reviews/{id}` - Delete review

### Products
- `GET /api/v1/products/` - List products with review summaries
- `DELETE /api/v1/products/{name}` - Delete product and all reviews

### Scraping
- `POST /api/v1/scraping/` - Create scraping job
- `GET /api/v1/scraping/jobs` - List scraping jobs
- `GET /api/v1/scraping/jobs/{id}` - Get job details
- `WS /api/v1/ws/scraping/{job_id}` - WebSocket for real-time updates

### Analytics
- `GET /api/v1/analytics/summary` - Overall analytics summary
- `GET /api/v1/analytics/products/{product_name}` - Product-specific analytics
- `GET /api/v1/analytics/products/{product_name}/trend` - Product trends

### Monitoring
- `GET /api/v1/monitoring/health` - Basic health check
- `GET /api/v1/monitoring/health/detailed` - Detailed component health
- `GET /api/v1/monitoring/metrics` - Application metrics
- `GET /api/v1/monitoring/readiness` - Kubernetes readiness probe
- `GET /api/v1/monitoring/liveness` - Kubernetes liveness probe


## 🤖 ML Models

### RoBERT Model

**Architecture**: Fine-tuned `dumitrescustefan/bert-base-romanian-cased-v1`

**Features**:
- Transformer-based model for Romanian text
- High accuracy for complex sentiment patterns
- Supports 3-class classification (positive, neutral, negative)

### XGBoost Model

**Architecture**: Gradient boosting with custom feature extraction

**Features**:
- LF-MICF (Local Frequency - Mutual Information Chi-squared Feature) extraction
- IGWO (Improved Grey Wolf Optimizer) feature selection
- Binary classification (positive, negative)
- Fast inference (<1ms per review)

**Module Dependencies**:
- Custom `preprocessing` module for text cleaning, tokenization, lemmatization
- Custom `features` module for LF-MICF and IGWO feature selection
- Uses PYTHONPATH configuration to resolve module imports

## 🔒 Security

### Implemented Security Features

1. **Authentication**: JWT-based with password hashing
2. **Input Validation**: XSS and SQL injection prevention
4. **Security Headers**: CSP, HSTS, X-Frame-Options, X-Content-Type-Options
5. **Request Protection**: Size limits and trusted host validation

## 📊 Monitoring

### Health Checks

The application provides multiple health check endpoints:

- **Basic Health**: Quick response for load balancers
- **Detailed Health**: Component-level status (database, Celery, Redis)
- **Liveness**: Container is running
- **Readiness**: Container is ready to accept traffic

### Metrics

Access application metrics including:
- User statistics
- Review counts and sentiment distribution
- Scraping job statistics
- Celery worker information

## 📁 Project Structure

```
recenz-ia/
├── backend/
│   ├── app/
│   │   ├── api/v1/          # API endpoints
│   │   ├── core/            # Core functionality (database, security, middleware)
│   │   ├── models/          # SQLAlchemy models
│   │   ├── schemas/         # Pydantic schemas
│   │   ├── services/        # Business logic
│   │   ├── tasks/           # Celery tasks
│   │   ├── preprocessing/   # Text preprocessing for XGBoost
│   │   ├── features/        # Feature extraction for XGBoost
│   │   └── main.py          # FastAPI application
│   ├── alembic/             # Database migrations
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/                # Next.js frontend application
├── ml-models/
│   ├── robert/              # RoBERT model files
│   └── xgboost/             # XGBoost model files
└── docker-compose.yml       # Docker orchestration
```

## 🛠️ Technology Stack

- **Backend**: FastAPI 0.104+
- **Database**: PostgreSQL 15+
- **Cache/Queue**: Redis 7+
- **Task Queue**: Celery 5.3+
- **Web Scraping**: Selenium 4.15+
- **ML Frameworks**: PyTorch 2.0+, Transformers 4.35+, XGBoost
- **Frontend**: Next.js, TypeScript, Tailwind CSS
- **Deployment**: Docker, Docker Compose
- **Authentication**: JWT (python-jose)
- **ORM**: SQLAlchemy 2.0+


## 📄 License

This project is licensed under the MIT License.

---

**Built for Romanian e-commerce review analysis**
