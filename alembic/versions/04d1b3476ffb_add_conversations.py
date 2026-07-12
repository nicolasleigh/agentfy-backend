"""add conversations

Revision ID: 04d1b3476ffb
Revises: 20260708_0002
Create Date: 2026-07-10 23:51:38.177452
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = '04d1b3476ffb'
down_revision: Union[str, None] = '20260708_0002'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table('conversations',
        sa.Column('id', sa.String(length=64), nullable=False),
        sa.Column('user_id', sa.String(length=64), nullable=False),
        sa.Column('title', sa.String(length=255), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True),
                  server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True),
                  server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    # SQLite does not support ALTER TABLE ADD CONSTRAINT,
    # so we add the column without the FK constraint here.
    # The FK is enforced at the application/ORM level.
    op.add_column('chat_completions',
        sa.Column('conversation_id', sa.String(length=64), nullable=True))
    # Index for efficient lookups
    op.create_index('ix_chat_completions_conversation_id', 'chat_completions',
                    ['conversation_id'])


def downgrade() -> None:
    op.drop_index('ix_chat_completions_conversation_id', table_name='chat_completions')
    op.drop_column('chat_completions', 'conversation_id')
    op.drop_table('conversations')
