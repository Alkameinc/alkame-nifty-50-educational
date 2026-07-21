# Alkame Nifty50 - FastAPI Backend

## Project Overview

This project is a FastAPI backend for the Alkame Nifty50 Educational platform. It provides REST APIs that generate stock trading signals, maintain prediction history, and support manual signal overrides.

The backend is designed for educational purposes and does not execute real trades.

---

## Technologies Used

- Python 3.14
- FastAPI
- Uvicorn
- Pydantic
- Git & GitHub
- VS Code

---

## Project Structure

```
backend/
│
├── main.py
├── dependencies.py
├── schemas.py
│
└── routers/
    ├── signals.py
    ├── history.py
    └── overrides.py
```

---

## Features Implemented

### FastAPI Application

- FastAPI server configured
- Swagger UI enabled
- Health check endpoint
- Root endpoint

---

### Signal API

**Endpoint**

```
GET /signal/{symbol}
```

Returns

- BUY / SELL / HOLD signal
- Confidence values
- Risk analysis
- Reasoning
- Market events

---

### History API

**Endpoint**

```
GET /history/{symbol}
```

Returns

- Previous predictions
- Prediction history
- Model outcomes

---

### Override API

**Endpoint**

```
POST /override
```

Allows manual override of generated signals.

---

## API Documentation

Swagger UI

```
http://127.0.0.1:8000/docs
```

Health Check

```
http://127.0.0.1:8000/health
```

---

## Tested Endpoints

- GET /
- GET /health
- GET /signal/{symbol}
- GET /history/{symbol}
- POST /override

All endpoints tested successfully.

---

## Running the Project

Activate virtual environment

```powershell
venv\Scripts\activate
```

Start FastAPI

```powershell
python -m uvicorn backend.main:app --reload --port 8000
```

Open Swagger

```
http://127.0.0.1:8000/docs
```

---

## Current Progress

- FastAPI Backend Setup
- Dependency Injection
- API Routers
- Pydantic Schemas
- Swagger Documentation
- Endpoint Testing

---

## Author

Suraj Shivsunder Jha

Information Technology Student

Full Stack Developer Intern