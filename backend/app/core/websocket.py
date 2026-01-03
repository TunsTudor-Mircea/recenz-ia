"""
WebSocket connection manager for real-time job updates.

Enterprise-grade connection manager with proper async synchronization,
connection limits, and cleanup.
"""
from typing import Dict, Set
from fastapi import WebSocket
from loguru import logger
import asyncio


class ConnectionManager:
    """Manages WebSocket connections for scraping job updates with async safety."""

    def __init__(self, max_connections_per_job: int = 10):
        # Dictionary mapping job_id to set of active connections
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        self.max_connections_per_job = max_connections_per_job
        # Async lock for thread-safe operations
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket, job_id: str) -> bool:
        """
        Accept a WebSocket connection and subscribe to job updates.

        Args:
            websocket: The WebSocket connection
            job_id: The job ID to subscribe to

        Returns:
            True if connection was accepted, False if rejected (limit reached)
        """
        async with self._lock:
            # Check connection limit
            current_connections = len(self.active_connections.get(job_id, set()))
            if current_connections >= self.max_connections_per_job:
                logger.warning(
                    f"Connection limit reached for job {job_id}: "
                    f"{current_connections}/{self.max_connections_per_job}"
                )
                return False

            await websocket.accept()

            if job_id not in self.active_connections:
                self.active_connections[job_id] = set()

            self.active_connections[job_id].add(websocket)
            logger.info(
                f"WebSocket connected for job {job_id}. "
                f"Total connections: {len(self.active_connections[job_id])}"
            )

            return True

    async def disconnect(self, websocket: WebSocket, job_id: str):
        """
        Remove a WebSocket connection.

        Args:
            websocket: The WebSocket connection to remove
            job_id: The job ID
        """
        async with self._lock:
            if job_id in self.active_connections:
                self.active_connections[job_id].discard(websocket)

                # Clean up empty job entries
                if not self.active_connections[job_id]:
                    del self.active_connections[job_id]
                    logger.info(f"No more connections for job {job_id}, cleaned up")
                else:
                    logger.info(
                        f"WebSocket disconnected for job {job_id}. "
                        f"Remaining: {len(self.active_connections[job_id])}"
                    )

    async def get_connection_count(self, job_id: str) -> int:
        """
        Get the number of active connections for a job.

        Args:
            job_id: The job ID

        Returns:
            Number of active connections
        """
        async with self._lock:
            return len(self.active_connections.get(job_id, set()))

    async def send_job_update(self, job_id: str, message: dict):
        """
        Send an update to all clients listening to a specific job.

        Args:
            job_id: The job ID
            message: The message dict to send
        """
        async with self._lock:
            if job_id not in self.active_connections:
                logger.debug(f"No active connections for job {job_id}")
                return

            # Create a copy of the set to avoid modification during iteration
            connections = self.active_connections[job_id].copy()

        # Send outside the lock to avoid holding it during I/O
        disconnected = []
        for connection in connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error sending message to WebSocket: {e}")
                disconnected.append(connection)

        # Clean up disconnected clients
        if disconnected:
            async with self._lock:
                if job_id in self.active_connections:
                    for connection in disconnected:
                        self.active_connections[job_id].discard(connection)

                    if not self.active_connections[job_id]:
                        del self.active_connections[job_id]
                        logger.info(
                            f"All connections for job {job_id} disconnected, cleaned up"
                        )


# Global connection manager instance
manager = ConnectionManager(max_connections_per_job=10)
