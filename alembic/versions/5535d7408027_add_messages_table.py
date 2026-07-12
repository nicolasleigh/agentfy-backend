"""add messages table

Revision ID: 5535d7408027
Revises: 04d1b3476ffb
Create Date: 2026-07-11 20:55:21.931745
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = '5535d7408027'
down_revision: Union[str, None] = '04d1b3476ffb'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # SQLite does not support ALTER TABLE ADD CONSTRAINT, so FK on
    # chat_completions(user_id, conversation_id) are enforced at the ORM level only.
    op.create_table('messages',
        sa.Column('id', sa.String(length=64), nullable=False),
        sa.Column('conversation_id', sa.String(length=64), nullable=False),
        sa.Column('role', sa.String(length=16), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True),
                  server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
        sa.ForeignKeyConstraint(['conversation_id'], ['conversations.id'],
                                ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_messages_conversation_id'), 'messages',
                    ['conversation_id'], unique=False)
    # The old index on chat_completions is no longer needed now that we have
    # a proper messages table with a dedicated index.
    op.drop_index(op.f('ix_chat_completions_conversation_id'),
                  table_name='chat_completions')


def downgrade() -> None:
    op.create_index(op.f('ix_chat_completions_conversation_id'),
                    'chat_completions', ['conversation_id'], unique=False)
    op.drop_index(op.f('ix_messages_conversation_id'), table_name='messages')
    op.drop_table('messages')
