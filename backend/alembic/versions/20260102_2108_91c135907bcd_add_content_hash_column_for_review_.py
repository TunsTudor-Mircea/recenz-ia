"""Add content_hash column for review deduplication

Revision ID: 91c135907bcd
Revises: 76deb472c551
Create Date: 2026-01-02 21:08:57.602167

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '91c135907bcd'
down_revision: Union[str, None] = '76deb472c551'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add content_hash column
    op.add_column('reviews', sa.Column('content_hash', sa.String(64), nullable=True))

    # Add unique index on content_hash
    op.create_index('ix_reviews_content_hash', 'reviews', ['content_hash'], unique=True)

    # For existing reviews, generate content_hash (need to handle this carefully)
    # In production, you'd populate this with a data migration script
    # For now, we'll make it nullable and handle it in the application


def downgrade() -> None:
    # Remove index and column
    op.drop_index('ix_reviews_content_hash', 'reviews')
    op.drop_column('reviews', 'content_hash')
