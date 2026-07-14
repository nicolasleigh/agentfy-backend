"""add documents and document_chunks

Revision ID: 4c7ca4a1a5e7
Revises: 9b90ef18b8cf
Create Date: 2026-07-14 13:07:43.801203
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = '4c7ca4a1a5e7'
down_revision: Union[str, None] = '9b90ef18b8cf'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table('documents',
        sa.Column('id', sa.String(length=64), nullable=False),
        sa.Column('user_id', sa.String(length=64), nullable=False),
        sa.Column('filename', sa.String(length=255), nullable=False),
        sa.Column('content_type', sa.String(length=128), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True),
                  server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id'),
    )

    # Dialect-aware embedding column:
    #   PostgreSQL → vector(768) (pgvector)
    #   SQLite     → Text (json-encoded)
    conn = op.get_bind()
    dialect = conn.dialect.name if hasattr(conn, 'dialect') else 'sqlite'
    if dialect == 'postgresql':
        import pgvector.sqlalchemy
        embedding_type = pgvector.sqlalchemy.Vector(768)
    else:
        embedding_type = sa.Text()

    op.create_table('document_chunks',
        sa.Column('id', sa.String(length=64), nullable=False),
        sa.Column('document_id', sa.String(length=64), nullable=False),
        sa.Column('chunk_index', sa.Integer(), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('embedding', embedding_type, nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True),
                  server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
        sa.ForeignKeyConstraint(['document_id'], ['documents.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(op.f('ix_document_chunks_document_id'), 'document_chunks',
                    ['document_id'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_document_chunks_document_id'), table_name='document_chunks')
    op.drop_table('document_chunks')
    op.drop_table('documents')
