"""
User Pydantic schemas for request/response validation.
"""
from pydantic import BaseModel, EmailStr, UUID4
from typing import Optional
from datetime import datetime


class UserBase(BaseModel):
    """Base user schema."""
    email: EmailStr
    full_name: Optional[str] = None


class UserCreate(UserBase):
    """Schema for user registration."""
    password: str


class UserLogin(BaseModel):
    """Schema for user login."""
    email: EmailStr
    password: str


class UserResponse(UserBase):
    """Schema for user response (excludes password)."""
    id: UUID4
    is_active: bool
    created_at: datetime

    model_config = {
        "from_attributes": True
    }


class Token(BaseModel):
    """Schema for authentication token."""
    access_token: str
    token_type: str = "bearer"


class TokenData(BaseModel):
    """Schema for decoded token data."""
    email: Optional[str] = None
