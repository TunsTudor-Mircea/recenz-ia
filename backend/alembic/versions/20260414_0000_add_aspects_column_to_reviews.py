"""add aspects column to reviews

Revision ID: a1b2c3d4e5f6
Revises: 4b0603916196
Create Date: 2026-04-14 00:00:00.000000

Adds a JSONB `aspects` column to the reviews table to store per-aspect
sentiment results from ABSA models.  Nullable so existing rows (labelled
by the binary sentiment models) are unaffected.

Schema stored in the column:
    {
        "BATERIE": "positive",
        "ECRAN": "none",
        "SUNET": "none",
        "PERFORMANTA": "positive",
        "CONECTIVITATE": "none",
        "DESIGN": "positive",
        "CALITATE_CONSTRUCTIE": "none",
        "PRET": "negative",
        "LIVRARE": "none",
        "GENERAL": "positive"
    }
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

# revision identifiers, used by Alembic.
revision = 'a1b2c3d4e5f6'
down_revision = '4b0603916196'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        'reviews',
        sa.Column('aspects', JSONB, nullable=True),
    )


def downgrade() -> None:
    op.drop_column('reviews', 'aspects')
