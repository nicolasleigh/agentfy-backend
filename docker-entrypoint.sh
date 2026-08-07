#!/bin/sh
set -e

echo "==> Waiting for database..."
python - <<'PY'
import asyncio
import os
import sys

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import create_async_engine

URL = os.environ.get("DATABASE_URL", "")

async def wait_for_db() -> None:
    if not URL:
        print("DATABASE_URL not set -- skipping DB wait.", file=sys.stderr)
        return
    engine = create_async_engine(URL, poolclass=sa.pool.NullPool)
    for attempt in range(30):
        try:
            async with engine.connect() as conn:
                await conn.execute(sa.text("SELECT 1"))
            print("==> Database ready.")
            await engine.dispose()
            return
        except Exception:
            await asyncio.sleep(2)
    print("==> ERROR: database not reachable after 60s", file=sys.stderr)
    raise SystemExit(1)

asyncio.run(wait_for_db())
PY

echo "==> Applying migrations (alembic upgrade head)..."
alembic upgrade head

echo "==> Starting: $*"
exec "$@"
