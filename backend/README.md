## Backend (FastAPI)

Run locally:

```bat
cd /d C:\Users\sj428\Desktop\main-project-btp
.venv\Scripts\activate
uvicorn backend.app.main:app --reload --port 8000
```

Test:

```bat
curl -X POST http://127.0.0.1:8000/api/risk/predict ^
  -H "Content-Type: application/json" ^
  -d "{\"features\": {}}"
```

