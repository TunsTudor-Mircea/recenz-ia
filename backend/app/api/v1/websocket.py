"""
WebSocket endpoints for real-time updates.

Enterprise-grade WebSocket implementation with authentication, connection limits,
proper error handling, and resource cleanup.
"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, status, Query
from sqlalchemy.orm import Session
from uuid import UUID
from loguru import logger
import asyncio
import redis.asyncio as aioredis
import json
from typing import Optional

from app.core.database import get_db
from app.core.websocket import manager
from app.models.scraping_job import ScrapingJob
from app.models.user import User
from app.core.security import decode_access_token
from app.config import settings


router = APIRouter()


async def get_current_user_ws(token: Optional[str] = Query(None)) -> Optional[User]:
    """
    Get current user from WebSocket query parameter.

    Args:
        token: JWT token from query parameter

    Returns:
        User if authenticated, None otherwise
    """
    if not token:
        return None

    try:
        payload = decode_access_token(token)
        if not payload or "sub" not in payload:
            return None

        from app.core.database import SessionLocal
        db = SessionLocal()
        try:
            user = db.query(User).filter(User.email == payload["sub"]).first()
            return user
        finally:
            db.close()

    except Exception as e:
        logger.warning(f"WebSocket authentication failed: {e}")
        return None


async def verify_job_ownership(job_id: UUID, user: User, db: Session) -> Optional[ScrapingJob]:
    """
    Verify that the user owns the specified job.

    Args:
        job_id: The job UUID
        user: The authenticated user
        db: Database session

    Returns:
        ScrapingJob if found and owned by user, None otherwise
    """
    job = db.query(ScrapingJob).filter(ScrapingJob.id == job_id).first()

    if not job:
        logger.warning(f"Job {job_id} not found")
        return None

    if job.user_id != user.id:
        logger.warning(
            f"User {user.id} attempted to access job {job_id} "
            f"owned by user {job.user_id}"
        )
        return None

    return job


@router.websocket("/scraping/{job_id}")
async def websocket_job_status(
    websocket: WebSocket,
    job_id: UUID,
    token: Optional[str] = Query(None)
):
    """
    WebSocket endpoint for real-time scraping job status updates.

    Clients connect to this endpoint to receive live updates about a scraping job.
    Requires authentication via query parameter: ?token=<jwt_token>

    Connection lifecycle:
    1. Authenticate user
    2. Verify job ownership
    3. Check connection limits
    4. Send initial job status
    5. If job not terminal, subscribe to Redis updates
    6. Handle ping/pong for keepalive
    7. Auto-close on terminal status or disconnect

    Message format:
    {
        "job_id": "uuid",
        "status": "pending|in_progress|completed|failed",
        "reviews_scraped": int,
        "reviews_created": int,
        "error_message": str (optional),
        "timestamp": str
    }

    Query Parameters:
        token: JWT authentication token

    WebSocket Close Codes:
        1000: Normal closure (job completed/failed)
        1008: Policy violation (authentication failed, job not found, not owned)
        1009: Message too large
        1011: Internal error
        1013: Try again later (connection limit reached)
    """
    from app.core.database import SessionLocal

    job_id_str = str(job_id)
    redis_client = None
    pubsub = None
    db = None

    try:
        # Step 1: Authenticate user
        user = await get_current_user_ws(token)
        if not user:
            logger.warning(f"Unauthenticated WebSocket connection attempt for job {job_id}")
            await websocket.close(
                code=status.WS_1008_POLICY_VIOLATION,
                reason="Authentication required"
            )
            return

        # Step 2: Verify job exists and is owned by user
        db = SessionLocal()
        job = await verify_job_ownership(job_id, user, db)

        if not job:
            await websocket.close(
                code=status.WS_1008_POLICY_VIOLATION,
                reason="Job not found or access denied"
            )
            return

        # Get initial job data before closing the session
        initial_status = {
            "job_id": job_id_str,
            "status": job.status.value,
            "reviews_scraped": job.reviews_scraped or 0,
            "reviews_created": job.reviews_created or 0,
            "error_message": job.error_message,
            "timestamp": (
                job.completed_at or job.started_at or job.created_at
            ).isoformat() if (job.completed_at or job.started_at or job.created_at) else None
        }
        is_terminal = job.status.value in ["completed", "failed"]

        logger.info(
            f"User {user.id} connecting to job {job_id} "
            f"(status: {job.status.value})"
        )

    except Exception as e:
        logger.error(f"Error verifying job {job_id}: {e}")
        if not websocket.client_state.disconnected:
            await websocket.close(
                code=status.WS_1011_INTERNAL_ERROR,
                reason="Internal error"
            )
        return
    finally:
        if db:
            db.close()

    # Step 3: Connect WebSocket with connection limit check
    try:
        accepted = await manager.connect(websocket, job_id_str)
        if not accepted:
            await websocket.close(
                code=status.WS_1013_TRY_AGAIN_LATER,
                reason="Connection limit reached"
            )
            return

        logger.info(f"WebSocket connected for job {job_id} (user: {user.id})")

    except Exception as e:
        logger.error(f"Error accepting WebSocket for job {job_id}: {e}")
        return

    # Step 4: Send initial job status
    try:
        await websocket.send_json(initial_status)

        # If job is already in terminal state, close connection after sending status
        if is_terminal:
            logger.info(
                f"Job {job_id} already in terminal state ({initial_status['status']}), "
                "closing connection"
            )
            await asyncio.sleep(0.5)  # Ensure message is delivered
            await websocket.close(
                code=status.WS_1000_NORMAL_CLOSURE,
                reason=f"Job {initial_status['status']}"
            )
            await manager.disconnect(websocket, job_id_str)
            return

    except Exception as e:
        logger.error(f"Error sending initial status for job {job_id}: {e}")
        await manager.disconnect(websocket, job_id_str)
        return

    # Step 5: Subscribe to Redis for updates (only if not terminal)
    try:
        redis_client = await aioredis.from_url(
            settings.REDIS_URL,
            encoding="utf-8",
            decode_responses=True,
            socket_connect_timeout=5,
            socket_keepalive=True
        )
        pubsub = redis_client.pubsub()
        channel = f"job_updates:{job_id_str}"
        await pubsub.subscribe(channel)
        logger.info(f"Subscribed to Redis channel: {channel}")

    except Exception as e:
        logger.error(f"Error setting up Redis for job {job_id}: {e}")
        await websocket.close(code=status.WS_1011_INTERNAL_ERROR, reason="Internal error")
        await manager.disconnect(websocket, job_id_str)
        return

    # Step 6 & 7: Handle messages and updates
    async def listen_redis():
        """Listen for Redis pub/sub messages and forward to WebSocket."""
        try:
            logger.debug(f"Starting Redis listener for channel: {channel}")
            while True:
                try:
                    message = await pubsub.get_message(
                        ignore_subscribe_messages=True,
                        timeout=1.0
                    )

                    if message and message["type"] == "message":
                        try:
                            data = json.loads(message["data"])
                            await websocket.send_json(data)
                            logger.debug(f"Forwarded update to WebSocket for job {job_id}: {data.get('status')}")

                            # Close connection if job reaches terminal state
                            if data.get("status") in ["completed", "failed"]:
                                logger.info(
                                    f"Job {job_id} reached terminal state via Redis, "
                                    "closing connection"
                                )
                                await asyncio.sleep(0.5)  # Ensure message is delivered
                                return

                        except json.JSONDecodeError as e:
                            logger.error(f"Error parsing Redis message for job {job_id}: {e}")
                        except Exception as e:
                            logger.error(f"Error forwarding message for job {job_id}: {e}")
                            return

                    await asyncio.sleep(0.1)  # Small delay to prevent busy loop

                except asyncio.CancelledError:
                    logger.info(f"Redis listener cancelled for job {job_id}")
                    break
                except Exception as e:
                    logger.error(f"Error in Redis listener loop for job {job_id}: {e}")
                    await asyncio.sleep(1)  # Wait before retrying

        except Exception as e:
            logger.error(f"Redis listener error for job {job_id}: {e}")

    async def handle_client_messages():
        """Handle messages from client (heartbeat/ping)."""
        try:
            while True:
                try:
                    data = await asyncio.wait_for(
                        websocket.receive_text(),
                        timeout=60.0  # 60 second timeout
                    )

                    # Echo back for heartbeat/ping-pong
                    if data == "ping":
                        await websocket.send_json({"type": "pong"})

                except asyncio.TimeoutError:
                    logger.warning(f"Client timeout for job {job_id}, closing connection")
                    break
                except WebSocketDisconnect:
                    logger.info(f"Client disconnected from job {job_id}")
                    break
                except Exception as e:
                    logger.error(f"Error receiving message for job {job_id}: {e}")
                    break

        except Exception as e:
            logger.error(f"Client message handler error for job {job_id}: {e}")

    # Run both tasks concurrently
    try:
        done, pending = await asyncio.wait(
            [
                asyncio.create_task(listen_redis()),
                asyncio.create_task(handle_client_messages())
            ],
            return_when=asyncio.FIRST_COMPLETED
        )

        # Cancel pending tasks
        for task in pending:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        # Check if any task raised an exception
        for task in done:
            if task.exception():
                logger.error(
                    f"Task raised exception for job {job_id}: {task.exception()}"
                )

    except Exception as e:
        logger.error(f"Error in concurrent tasks for job {job_id}: {e}")

    finally:
        # Cleanup
        logger.info(f"Cleaning up WebSocket connection for job {job_id}")

        if pubsub:
            try:
                await pubsub.unsubscribe(channel)
                await pubsub.close()
            except Exception as e:
                logger.warning(f"Error closing pubsub for job {job_id}: {e}")

        if redis_client:
            try:
                await redis_client.close()
            except Exception as e:
                logger.warning(f"Error closing Redis client for job {job_id}: {e}")

        await manager.disconnect(websocket, job_id_str)
        logger.info(f"WebSocket cleanup complete for job {job_id}")
