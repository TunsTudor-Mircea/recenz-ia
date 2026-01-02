# RecenzIA - Enterprise AI-Powered Review Analysis Platform

[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15+-blue.svg)](https://www.postgresql.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Enterprise-grade AI-powered sentiment analysis platform for Romanian e-commerce reviews with advanced scraping, analytics, and monitoring capabilities.

## 🌟 Features

### Core Functionality
- 🤖 **Dual ML Models**: RoBERT (Romanian BERT, 98% accuracy) + XGBoost (92% accuracy, <1ms inference)
- 🕷️ **Automated Web Scraping**: Extract reviews from eMAG with Selenium
- 📊 **Advanced Analytics**: Comprehensive sentiment trends, rating distributions, and product insights
- 🔐 **Enterprise Security**: JWT authentication, XSS/SQL injection protection, rate limiting
- 📈 **Real-time Monitoring**: Health checks, metrics, and Kubernetes-ready probes
- 🚀 **Production-Ready**: Docker-based deployment with Celery background tasks

### Security & Validation
- Input sanitization and validation (XSS, SQL injection protection)
- Rate limiting (60 requests/minute)
- Security headers (CSP, HSTS, X-Frame-Options)
- Request size limits (10MB max)
- Domain whitelist for scraping URLs

### Monitoring & Observability
- `/health` - Basic health check
- `/health/detailed` - Component-level health status
- `/metrics` - Application metrics (users, reviews, jobs, Celery)
- `/readiness` - Kubernetes readiness probe
- `/liveness` - Kubernetes liveness probe

## 📋 Table of Contents

- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [API Documentation](#api-documentation)
- [Development](#development)
- [Deployment](#deployment)
- [ML Models](#ml-models)
- [Security](#security)
- [Monitoring](#monitoring)

## 🚀 Quick Start

### Prerequisites

- Docker & Docker Compose
- 8GB RAM minimum (16GB recommended)
- GPU (optional, for faster training)

### 1. Clone and Setup

```bash
# Clone the repository
git clone https://github.com/TunsTudor-Mircea/recenz-ia.git
cd recenz-ia

# Copy ML models from POC
mkdir -p ml-models/robert ml-models/xgboost
cp -r sentiment-ai-poc/results/experiments/quick_test ml-models/robert/
cp -r sentiment-ai-poc/results/experiments/xgb_optimized2 ml-models/xgboost/
```

### 2. Start Services

```bash
# Start all services (PostgreSQL, Redis, Backend, Celery)
docker-compose up -d

# Check logs
docker-compose logs -f backend

# Verify services are running
docker-compose ps
```

### 3. Access the Application

- **API**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/api/v1/monitoring/health

### 4. Create Your First User

```bash
# Register a new user
curl -X POST "http://localhost:8000/api/v1/auth/register" \
  -H "Content-Type: application/json" \
  -d '{
    "email": "user@example.com",
    "password": "securepassword123",
    "full_name": "John Doe"
  }'

# Login to get JWT token
curl -X POST "http://localhost:8000/api/v1/auth/login" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=user@example.com&password=securepassword123"
```

### 5. Start Scraping Reviews

```bash
# Use the token from login
export TOKEN="your_jwt_token_here"

# Create a scraping job for an eMAG product
curl -X POST "http://localhost:8000/api/v1/scraping/" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://www.emag.ro/telefon-mobil-apple-iphone-15-128gb-5g-black-mu0n3zd-a/pd/DGX9FV3BM/",
    "site_type": "emag"
  }'

# Check job status
curl -X GET "http://localhost:8000/api/v1/scraping/jobs" \
  -H "Authorization: Bearer $TOKEN"
```

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         Frontend (Coming Soon)               │
│                         Next.js + TypeScript                 │
└────────────────────────┬────────────────────────────────────┘
                         │
                         │ REST API
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                      FastAPI Backend                         │
│  ┌──────────────┬──────────────┬──────────────────────────┐ │
│  │ Auth API     │ Reviews API  │ Analytics API  │ Scraping│ │
│  │ (JWT)        │ (CRUD)       │ (Insights)     │ API     │ │
│  └──────────────┴──────────────┴──────────────────────────┘ │
│  ┌──────────────────────────────────────────────────────────┐│
│  │ Middleware: Security, Rate Limiting, Validation          ││
│  └──────────────────────────────────────────────────────────┘│
└────────────────┬────────────────┬───────────────────────────┘
                 │                │
         ┌───────┴────────┐   ┌──┴────────────┐
         │                │   │               │
    ┌────▼─────┐    ┌────▼────▼──┐    ┌──────▼─────┐
    │PostgreSQL│    │   Redis     │    │   Celery   │
    │ Database │    │   Cache     │    │   Worker   │
    │          │    │   Queue     │    │            │
    └──────────┘    └─────────────┘    └──────┬─────┘
                                              │
                                    ┌─────────▼──────────┐
                                    │  Selenium Scraper  │
                                    │  + ML Models       │
                                    │  (RoBERT/XGBoost)  │
                                    └────────────────────┘
```

### Components

1. **FastAPI Backend**: Main application server with RESTful API
2. **PostgreSQL**: Relational database for users, reviews, and jobs
3. **Redis**: Message broker for Celery and caching
4. **Celery Worker**: Background task processing for scraping
5. **Selenium**: Headless Chrome for web scraping
6. **ML Models**: RoBERT and XGBoost for sentiment analysis

## 📚 API Documentation

### Authentication

#### Register User
```http
POST /api/v1/auth/register
Content-Type: application/json

{
  "email": "user@example.com",
  "password": "password123",
  "full_name": "John Doe"
}
```

#### Login
```http
POST /api/v1/auth/login
Content-Type: application/x-www-form-urlencoded

username=user@example.com&password=password123
```

### Reviews

#### Create Review
```http
POST /api/v1/reviews/
Authorization: Bearer {token}
Content-Type: application/json

{
  "product_name": "iPhone 15 Pro",
  "review_text": "Excellent product! Very satisfied with the purchase.",
  "rating": 5
}
```

#### List Reviews (with filtering)
```http
GET /api/v1/reviews/?product_name=iPhone&sentiment_label=positive&min_rating=4&sort_by=rating&order=desc
Authorization: Bearer {token}
```

#### Update Review
```http
PUT /api/v1/reviews/{review_id}
Authorization: Bearer {token}
Content-Type: application/json

{
  "review_text": "Updated review text",
  "rating": 4
}
```

### Scraping

#### Create Scraping Job
```http
POST /api/v1/scraping/
Authorization: Bearer {token}
Content-Type: application/json

{
  "url": "https://www.emag.ro/product-url",
  "site_type": "emag"
}
```

#### List Scraping Jobs
```http
GET /api/v1/scraping/jobs?status=completed&skip=0&limit=10
Authorization: Bearer {token}
```

### Analytics

#### Get Overall Summary
```http
GET /api/v1/analytics/summary
Authorization: Bearer {token}
```

Response:
```json
{
  "total_reviews": 150,
  "total_products": 5,
  "average_rating": 4.2,
  "sentiment_distribution": {
    "positive": 100,
    "neutral": 30,
    "negative": 20,
    "total": 150
  },
  "top_rated_products": ["iPhone 15", "Samsung Galaxy S24"],
  "most_reviewed_products": ["iPhone 15", "AirPods Pro"]
}
```

#### Get Product Analytics
```http
GET /api/v1/analytics/products/{product_name}
Authorization: Bearer {token}
```

#### Get Product Trend
```http
GET /api/v1/analytics/products/{product_name}/trend?period=day&days=30
Authorization: Bearer {token}
```

### Monitoring

#### Health Check
```http
GET /api/v1/monitoring/health
```

#### Detailed Health Check
```http
GET /api/v1/monitoring/health/detailed
```

Response:
```json
{
  "status": "healthy",
  "timestamp": "2025-01-02T12:00:00",
  "version": "1.0.0",
  "components": {
    "database": {"status": "healthy", "message": "Database connection successful"},
    "celery": {"status": "healthy", "message": "1 worker(s) active"},
    "redis": {"status": "healthy", "message": "Redis broker connection successful"}
  }
}
```

#### Metrics
```http
GET /api/v1/monitoring/metrics
```

## 💻 Development

### Local Setup (Without Docker)

#### Backend

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create .env file
cat > .env << EOF
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/recenzia
REDIS_URL=redis://localhost:6379/0
SECRET_KEY=your-secret-key-here
ROBERT_MODEL_PATH=../ml-models/robert/quick_test
XGBOOST_MODEL_PATH=../ml-models/xgboost/xgb_optimized2/xgboost_model.joblib
XGBOOST_PREPROCESSOR_PATH=../ml-models/xgboost/xgb_optimized2/preprocessor.joblib
XGBOOST_SELECTOR_PATH=../ml-models/xgboost/xgb_optimized2/feature_selector.joblib
EOF

# Run migrations
alembic upgrade head

# Start backend
uvicorn app.main:app --reload
```

#### Celery Worker

```bash
# In a separate terminal
cd backend
source venv/bin/activate

# Start Celery worker
celery -A app.tasks.celery_app worker --loglevel=info
```

### Database Migrations

```bash
# Create a new migration
alembic revision --autogenerate -m "Description of changes"

# Apply migrations
alembic upgrade head

# Rollback migration
alembic downgrade -1
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html

# Run specific test file
pytest tests/test_reviews.py
```

## 🚢 Deployment

### Docker Compose (Production)

```bash
# Build and start services
docker-compose -f docker-compose.prod.yml up -d

# View logs
docker-compose -f docker-compose.prod.yml logs -f

# Scale Celery workers
docker-compose -f docker-compose.prod.yml up -d --scale celery=3
```

### Kubernetes

```yaml
# health check configuration
livenessProbe:
  httpGet:
    path: /api/v1/monitoring/liveness
    port: 8000
  initialDelaySeconds: 30
  periodSeconds: 10

readinessProbe:
  httpGet:
    path: /api/v1/monitoring/readiness
    port: 8000
  initialDelaySeconds: 5
  periodSeconds: 5
```

### Environment Variables

Required environment variables:

- `DATABASE_URL`: PostgreSQL connection string
- `REDIS_URL`: Redis connection string
- `SECRET_KEY`: JWT secret key (use strong random value)
- `ROBERT_MODEL_PATH`: Path to RoBERT model
- `XGBOOST_MODEL_PATH`: Path to XGBoost model
- `CORS_ORIGINS`: Allowed CORS origins (comma-separated)
- `DEBUG`: Set to `False` in production

## 🤖 ML Models

### RoBERT Model

**Architecture**: Fine-tuned `dumitrescustefan/bert-base-romanian-cased-v1`

**Performance**:
- Accuracy: 98%
- Inference time: ~20ms per review
- GPU recommended for faster inference

**Usage**:
```python
from app.services.sentiment import sentiment_analyzer

result = sentiment_analyzer.analyze(
    text="Produsul este excelent!",
    model="robert"
)
# Returns: {"sentiment_label": "positive", "sentiment_score": 0.98, "model_used": "robert"}
```

### XGBoost Model

**Features**: LF-MICF extraction + IGWO feature selection

**Performance**:
- Accuracy: 92%
- Inference time: <1ms per review
- No GPU required

**Usage**:
```python
from app.services.sentiment import sentiment_analyzer

result = sentiment_analyzer.analyze(
    text="Produsul este excelent!",
    model="xgboost"
)
# Returns: {"sentiment_label": "positive", "sentiment_score": 0.95, "model_used": "xgboost"}
```

## 🔒 Security

### Implemented Security Features

1. **Authentication & Authorization**
   - JWT-based authentication
   - Password hashing with bcrypt
   - Token expiration (30 minutes)

2. **Input Validation**
   - XSS protection (HTML escaping)
   - SQL injection prevention
   - Input length limits
   - URL domain whitelist

3. **Rate Limiting**
   - 60 requests per minute per IP
   - Custom limits per endpoint
   - Rate limit headers in responses

4. **Security Headers**
   - Content-Security-Policy
   - X-Content-Type-Options: nosniff
   - X-Frame-Options: DENY
   - HSTS (Strict-Transport-Security)

5. **Request Protection**
   - Request size limits (10MB max)
   - Trusted host validation
   - CORS configuration

### Security Best Practices

- Always use HTTPS in production
- Rotate JWT secret keys regularly
- Keep dependencies updated
- Monitor security logs
- Use environment variables for secrets
- Enable database SSL connections

## 📊 Monitoring

### Health Checks

- **Basic**: `/api/v1/monitoring/health`
- **Detailed**: `/api/v1/monitoring/health/detailed`
- **Liveness**: `/api/v1/monitoring/liveness`
- **Readiness**: `/api/v1/monitoring/readiness`

### Metrics

Access `/api/v1/monitoring/metrics` to get:

- Total users (active/inactive)
- Review statistics (count, avg rating, sentiment distribution)
- Scraping job statistics (by status)
- Celery task queue stats
- Worker information

### Logging

All requests are logged with:
- HTTP method and path
- Client IP address
- Response status code
- Request duration

Example log:
```
2025-01-02 12:00:00 | INFO | Request: GET /api/v1/reviews/ from 192.168.1.1
2025-01-02 12:00:00 | INFO | Response: GET /api/v1/reviews/ status=200 duration=0.123s
```

## 📁 Project Structure

```
recenz-ia/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   └── v1/
│   │   │       ├── auth.py           # Authentication endpoints
│   │   │       ├── reviews.py        # Review CRUD + filtering
│   │   │       ├── scraping.py       # Scraping job management
│   │   │       ├── analytics.py      # Analytics endpoints
│   │   │       └── monitoring.py     # Health & metrics
│   │   ├── core/
│   │   │   ├── database.py          # Database connection
│   │   │   ├── security.py          # JWT & password hashing
│   │   │   ├── middleware.py        # Security middleware
│   │   │   └── validation.py        # Input validation
│   │   ├── models/
│   │   │   ├── user.py              # User model
│   │   │   ├── review.py            # Review model
│   │   │   └── scraping_job.py      # Scraping job model
│   │   ├── schemas/
│   │   │   ├── review.py            # Pydantic schemas
│   │   │   ├── analytics.py         # Analytics schemas
│   │   │   └── scraping_job.py      # Job schemas
│   │   ├── services/
│   │   │   ├── sentiment.py         # ML model integration
│   │   │   ├── scraper.py           # Web scraping logic
│   │   │   └── analytics.py         # Analytics calculations
│   │   ├── tasks/
│   │   │   ├── celery_app.py        # Celery configuration
│   │   │   └── scraping.py          # Async scraping tasks
│   │   ├── config.py                # App configuration
│   │   └── main.py                  # FastAPI app entry
│   ├── alembic/                     # Database migrations
│   ├── requirements.txt             # Python dependencies
│   └── Dockerfile                   # Backend container
├── ml-models/
│   ├── robert/                      # RoBERT model files
│   └── xgboost/                     # XGBoost model files
├── sentiment-ai-poc/                # ML training code
├── docker-compose.yml               # Docker orchestration
└── README.md                        # This file
```

## 🛠️ Technology Stack

- **Backend**: FastAPI 0.104+
- **Database**: PostgreSQL 15+
- **Cache/Queue**: Redis 7+
- **Task Queue**: Celery 5.3+
- **Web Scraping**: Selenium 4.15+
- **ML Framework**: PyTorch 2.0+, Transformers 4.35+
- **Deployment**: Docker, Docker Compose
- **Authentication**: JWT (python-jose)
- **ORM**: SQLAlchemy 2.0+
- **Migration**: Alembic 1.12+

## 📈 Performance

- **API Response Time**: <100ms (avg)
- **Scraping Speed**: ~10 reviews/second
- **Sentiment Analysis**:
  - RoBERT: ~20ms/review
  - XGBoost: <1ms/review
- **Database**: Optimized with indexes on common queries
- **Caching**: Redis for frequently accessed data

## 🗺️ Roadmap

### Phase 1 (✅ Complete)
- [x] Backend API with FastAPI
- [x] User authentication (JWT)
- [x] Review CRUD operations
- [x] Web scraping (eMAG)
- [x] Sentiment analysis integration
- [x] Analytics endpoints
- [x] Security & validation
- [x] Health checks & monitoring
- [x] Docker deployment

### Phase 2 (In Progress)
- [ ] Frontend (Next.js + TypeScript)
- [ ] Real-time dashboards
- [ ] Advanced filtering UI
- [ ] Aspect-based sentiment analysis
- [ ] LDA topic modeling

### Phase 3 (Planned)
- [ ] Multi-site scraping (CEL.ro, Altex)
- [ ] Webhook notifications
- [ ] Export functionality (CSV, PDF)
- [ ] API rate limiting tiers
- [ ] User roles & permissions

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **LaRoSeDa Dataset**: Romanian sentiment analysis dataset
- **Romanian BERT**: `dumitrescustefan/bert-base-romanian-cased-v1`
- **Stanza**: Romanian NLP processing

## 📧 Contact

Tudor Tuns - [GitHub](https://github.com/TunsTudor-Mircea)

Project Link: [https://github.com/TunsTudor-Mircea/recenz-ia](https://github.com/TunsTudor-Mircea/recenz-ia)

---

**Built with ❤️ for Romanian e-commerce**
