import { useEffect, useState, useRef, useCallback } from "react"
import type { ScrapingJob } from "@/types/api"

interface UseScrapingJobWSOptions {
  jobId: string | null
  onStatusChange?: (job: Partial<ScrapingJob>) => void
  onComplete?: (job: Partial<ScrapingJob>) => void
  onError?: (job: Partial<ScrapingJob>) => void
}

export function useScrapingJobWS({
  jobId,
  onStatusChange,
  onComplete,
  onError,
}: UseScrapingJobWSOptions) {
  const [job, setJob] = useState<Partial<ScrapingJob> | null>(null)
  const [isConnected, setIsConnected] = useState(false)
  const [connectionError, setConnectionError] = useState<string | null>(null)

  // Refs to avoid recreating connect function
  const wsRef = useRef<WebSocket | null>(null)
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null)
  const reconnectAttemptsRef = useRef(0)
  const isTerminalStatusRef = useRef(false)
  const pingIntervalRef = useRef<NodeJS.Timeout | null>(null)
  const mountedRef = useRef(true)

  // Stable callback refs
  const onStatusChangeRef = useRef(onStatusChange)
  const onCompleteRef = useRef(onComplete)
  const onErrorRef = useRef(onError)

  // Update callback refs when they change
  useEffect(() => {
    onStatusChangeRef.current = onStatusChange
    onCompleteRef.current = onComplete
    onErrorRef.current = onError
  }, [onStatusChange, onComplete, onError])

  const maxReconnectAttempts = 5
  const baseReconnectDelay = 1000 // 1 second

  // Cleanup function
  const cleanup = useCallback(() => {
    // Clear reconnection timeout
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current)
      reconnectTimeoutRef.current = null
    }

    // Clear ping interval
    if (pingIntervalRef.current) {
      clearInterval(pingIntervalRef.current)
      pingIntervalRef.current = null
    }

    // Close WebSocket
    if (wsRef.current) {
      // Remove event listeners to prevent callbacks after cleanup
      wsRef.current.onopen = null
      wsRef.current.onmessage = null
      wsRef.current.onerror = null
      wsRef.current.onclose = null

      if (wsRef.current.readyState === WebSocket.OPEN ||
          wsRef.current.readyState === WebSocket.CONNECTING) {
        wsRef.current.close(1000, "Component cleanup")
      }
      wsRef.current = null
    }
  }, [])

  const connect = useCallback(() => {
    if (!jobId || !mountedRef.current) return

    // Don't reconnect if job already reached terminal status
    if (isTerminalStatusRef.current) {
      console.log("[WebSocket] Job already in terminal status, skipping connection")
      return
    }

    // Cleanup existing connection before creating new one
    cleanup()

    // Get WebSocket URL from environment (convert http to ws)
    const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"
    const wsProtocol = apiUrl.startsWith("https") ? "wss" : "ws"
    const wsHost = apiUrl.replace(/^https?:\/\//, "")

    // Get auth token from localStorage
    const token = typeof window !== "undefined" ? localStorage.getItem("access_token") : null
    const wsUrl = `${wsProtocol}://${wsHost}/api/v1/ws/scraping/${jobId}${token ? `?token=${token}` : ""}`

    try {
      console.log(`[WebSocket] Attempting to connect to: ${wsUrl}`)
      const ws = new WebSocket(wsUrl)

      ws.onopen = () => {
        if (!mountedRef.current) {
          ws.close()
          return
        }

        console.log(`[WebSocket] Connected to job ${jobId}`)
        setIsConnected(true)
        setConnectionError(null)
        reconnectAttemptsRef.current = 0

        // Setup ping interval
        pingIntervalRef.current = setInterval(() => {
          if (ws.readyState === WebSocket.OPEN) {
            ws.send("ping")
          }
        }, 30000) // Ping every 30 seconds
      }

      ws.onmessage = (event) => {
        if (!mountedRef.current) return

        try {
          const data = JSON.parse(event.data)

          // Skip pong messages
          if (data.type === "pong") {
            return
          }

          console.log(`[WebSocket] Received update:`, data)

          // Validate message structure
          if (!data.job_id || !data.status) {
            console.warn("[WebSocket] Invalid message format:", data)
            return
          }

          const jobUpdate: Partial<ScrapingJob> = {
            id: data.job_id,
            status: data.status,
            reviews_scraped: data.reviews_scraped || 0,
            reviews_created: data.reviews_created || 0,
            error_message: data.error_message || null,
          }

          setJob(jobUpdate)

          // Check for terminal status FIRST before any callbacks
          const isTerminal = data.status === "completed" || data.status === "failed"
          if (isTerminal) {
            isTerminalStatusRef.current = true
            console.log(`[WebSocket] Job reached terminal status: ${data.status}`)
          }

          // Call status change callback
          onStatusChangeRef.current?.(jobUpdate)

          // Call specific callbacks based on status
          if (data.status === "completed") {
            onCompleteRef.current?.(jobUpdate)
          } else if (data.status === "failed") {
            onErrorRef.current?.(jobUpdate)
          }

          // Server will close connection for terminal states
        } catch (error) {
          console.error("[WebSocket] Error parsing message:", error)
        }
      }

      ws.onerror = (error) => {
        if (!mountedRef.current) return
        console.error("[WebSocket] Error:", error)
        setConnectionError("WebSocket connection error")
      }

      ws.onclose = (event) => {
        if (!mountedRef.current) return

        console.log(`[WebSocket] Disconnected (code: ${event.code})`)
        setIsConnected(false)

        // Clear ping interval when connection closes
        if (pingIntervalRef.current) {
          clearInterval(pingIntervalRef.current)
          pingIntervalRef.current = null
        }

        // Don't reconnect if job reached terminal status
        if (isTerminalStatusRef.current) {
          console.log("[WebSocket] Job in terminal status, not reconnecting")
          return
        }

        // Don't reconnect if it was a clean close
        // Code 1000 = normal closure, 1005 = no status present
        if (event.code === 1000 || event.code === 1005) {
          console.log("[WebSocket] Clean close, not reconnecting")
          return
        }

        // Don't reconnect if component unmounted
        if (!mountedRef.current) {
          return
        }

        // Attempt reconnection with exponential backoff
        if (reconnectAttemptsRef.current < maxReconnectAttempts) {
          const delay = baseReconnectDelay * Math.pow(2, reconnectAttemptsRef.current)
          console.log(`[WebSocket] Reconnecting in ${delay}ms (attempt ${reconnectAttemptsRef.current + 1}/${maxReconnectAttempts})`)

          reconnectTimeoutRef.current = setTimeout(() => {
            reconnectAttemptsRef.current += 1
            connect()
          }, delay)
        } else {
          setConnectionError("Failed to connect after multiple attempts")
          console.error("[WebSocket] Max reconnection attempts reached")
        }
      }

      wsRef.current = ws
    } catch (error) {
      console.error("[WebSocket] Connection failed:", error)
      setConnectionError("Failed to establish WebSocket connection")
    }
  }, [jobId, cleanup]) // Only jobId and cleanup in dependencies

  useEffect(() => {
    mountedRef.current = true

    if (jobId) {
      // Reset terminal status flag for new job
      isTerminalStatusRef.current = false
      reconnectAttemptsRef.current = 0
      setConnectionError(null)
      connect()
    }

    return () => {
      mountedRef.current = false
      cleanup()
    }
  }, [jobId, connect, cleanup])

  return {
    job,
    isConnected,
    connectionError,
  }
}
