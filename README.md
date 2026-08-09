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

## Docker 运行

后端 + PostgreSQL（含 pgvector 扩展）一键启动，源码挂载并热重载，适合日常开发：

```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d --build
```

常用命令（也可用 `make docker-*`）：

```bash
make docker-dev     # 启动开发模式 (等价于上面的 compose 命令)
make docker-down    # 停止服务
make docker-logs    # 跟踪后端日志
make docker-db      # 进入 postgres shell
```

说明：

- **Ollama 默认连接宿主机**，请先在本机运行 `ollama serve`。容器内通过
  `host.docker.internal:11434` 访问。如需改成其他地址，编辑
  `docker-compose.yml` 里的 `OLLAMA_BASE_URL`。
- 数据库连接与密钥可在 `docker-compose.yml` 中调整；`JWT_SECRET_KEY`、
  `LLM_PROVIDER` 等会优先读取项目 `.env` 里的同名变量。
- PostgreSQL 数据保存在 named volume `db_data` 中，`docker compose down` 不会
  删除数据；如需彻底清理可加 `-v`。

## MCP 服务

后端在 `http://localhost:8000/mcp` 暴露一个 **MCP Server**（Streamable HTTP），
让外部 MCP 客户端（Claude Code、Claude Desktop 等）直接检索知识库。

暴露的工具：

- `search_knowledge_base(query, top_k=5)` — 对全部文档做向量相似度检索，返回
  命中的 chunk 内容、来源文档与分数
- `list_documents()` — 列出知识库中现有的文档

### 配置 token

```bash
# 生成一个随机 token 并写入 .env
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

在 `.env` 中设置 `MCP_AUTH_TOKEN=<生成的token>`。**不设置时 `/mcp` 会拒绝所有
请求（401）**，保证安全默认。

### 在 Claude Code 中使用

在项目 `.mcp.json`（或全局 MCP 配置）中加入：

```json
{
  "mcpServers": {
    "ai-chat-kb": {
      "type": "http",
      "url": "http://localhost:8000/mcp",
      "headers": {
        "Authorization": "Bearer <MCP_AUTH_TOKEN>"
      }
    }
  }
}
```

然后即可在对话中让 Claude 检索知识库（注意：需运行在真实的 PostgreSQL +
pgvector 数据库上，本地 sqlite 模式不支持向量检索）。
