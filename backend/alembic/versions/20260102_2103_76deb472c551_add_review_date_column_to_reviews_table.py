"""Add review_date column to reviews table

Revision ID: 76deb472c551
Revises: 
Create Date: 2026-01-02 21:03:11.222482

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '76deb472c551'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add review_date column to reviews table
    op.add_column('reviews', sa.Column('review_date', sa.DateTime(timezone=True), nullable=True))


def downgrade() -> None:
    # Remove review_date column from reviews table
    op.drop_column('reviews', 'review_date')
