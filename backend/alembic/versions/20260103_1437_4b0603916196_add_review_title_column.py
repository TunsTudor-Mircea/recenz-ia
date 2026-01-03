"""add_review_title_column

Revision ID: 4b0603916196
Revises: 91c135907bcd
Create Date: 2026-01-03 14:37:37.958710

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '4b0603916196'
down_revision: Union[str, None] = '91c135907bcd'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add review_title column to reviews table
    op.add_column('reviews', sa.Column('review_title', sa.String(length=500), nullable=True))


def downgrade() -> None:
    # Remove review_title column from reviews table
    op.drop_column('reviews', 'review_title')
