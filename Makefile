# AI Chat Backend — 常用命令
# 用法: make <target>    (例如 `make dev`)

.PHONY: help setup install dev run test lint format format-check \
        db-upgrade db-revision db-downgrade db-history clean

VENV    := .venv
PYTHON  := $(VENV)/bin/python
PIP     := $(VENV)/bin/pip
UVICORN := $(VENV)/bin/uvicorn
PYTEST  := $(VENV)/bin/pytest
RUFF    := $(VENV)/bin/ruff
ALEMBIC := $(VENV)/bin/alembic

help: ## 列出所有可用命令
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

setup: venv install  ## 初始化环境 (创建 venv 并安装依赖)

venv: ## 创建虚拟环境 (若不存在)
	@test -d $(VENV) || python3 -m venv $(VENV)
	@echo "✓ venv ready: $(VENV)"

install: venv ## 安装项目依赖 (含 dev)
	$(PIP) install -e ".[dev]"

dev: ## 启动开发服务器 (热重载)
	$(UVICORN) app.main:app --reload

run: ## 启动服务器
	$(UVICORN) app.main:app

test: ## 运行测试
	$(PYTEST)

lint: ## 代码检查 (ruff check)
	$(RUFF) check .

lint-fix: ## 自动修复 lint 问题
	$(RUFF) check --fix .

format: ## 格式化代码
	$(RUFF) format .

format-check: ## 检查格式 (不修改)
	$(RUFF) format --check .

db-upgrade: ## 应用所有数据库迁移
	$(ALEMBIC) upgrade head

db-revision: ## 基于模型变更生成新迁移 (mig=迁移说明)
	@test -n "$(mig)" || (echo "用法: make db-revision mig='add user table'"; exit 1)
	$(ALEMBIC) revision --autogenerate -m "$(mig)"

db-downgrade: ## 回滚一次迁移 (rev=目标版本, 默认 -1)
	$(ALEMBIC) downgrade $(if $(rev),$(rev),-1)

db-history: ## 查看迁移历史
	$(ALEMBIC) history

clean: ## 清理缓存与临时文件
	rm -rf .pytest_cache .ruff_cache app/**/__pycache__ alembic/**/__pycache__ tests/**/__pycache__
	find . -name '*.pyc' -delete
