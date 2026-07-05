# AI Chat Backend

一个 OpenAI Chat Completions 风格的 FastAPI 后端骨架。

## 技术栈

- Python 3.11+
- FastAPI
- Pydantic v2
- SQLAlchemy 2 async
- Alembic

## 快速开始

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
alembic upgrade head
uvicorn app.main:app --reload
```

访问：

- 健康检查：`GET /health`
- OpenAI 风格接口：`POST /v1/chat/completions`
- 文档：`GET /docs`

## 示例请求

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "demo-chat",
    "messages": [{"role": "user", "content": "你好"}]
  }'
```
