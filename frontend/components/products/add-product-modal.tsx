"use client"

import type React from "react"
import { useState, useEffect } from "react"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Progress } from "@/components/ui/progress"
import { useToast } from "@/hooks/use-toast"
import api from "@/lib/api"
import { Plus, Loader2, CheckCircle2, XCircle } from "lucide-react"
import type { ScrapingJob } from "@/types/api"

interface AddProductModalProps {
  onSuccess?: () => void
}

export function AddProductModal({ onSuccess }: AddProductModalProps) {
  const [open, setOpen] = useState(false)
  const [url, setUrl] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const [job, setJob] = useState<ScrapingJob | null>(null)
  const { toast } = useToast()
  const [pollIntervalId, setPollIntervalId] = useState<NodeJS.Timeout | null>(null)
  const [timeoutId, setTimeoutId] = useState<NodeJS.Timeout | null>(null)

  // Cleanup polling intervals on unmount or dialog close
  useEffect(() => {
    return () => {
      if (pollIntervalId) {
        clearInterval(pollIntervalId)
      }
      if (timeoutId) {
        clearTimeout(timeoutId)
      }
    }
  }, [pollIntervalId, timeoutId])

  // Clear polling when dialog closes
  useEffect(() => {
    if (!open) {
      if (pollIntervalId) {
        clearInterval(pollIntervalId)
        setPollIntervalId(null)
      }
      if (timeoutId) {
        clearTimeout(timeoutId)
        setTimeoutId(null)
      }
    }
  }, [open])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()

    if (!url.includes("emag.ro")) {
      toast({
        title: "Invalid URL",
        description: "Please enter a valid eMAG product URL",
        variant: "destructive",
      })
      return
    }

    setIsLoading(true)

    try {
      const response = await api.post<ScrapingJob>("/api/v1/scraping/", {
        url,
        site_type: "emag",
      })

      setJob(response.data)
      toast({
        title: "Scraping Started",
        description: "Your product is being analyzed. This may take a few minutes.",
      })

      // Poll for job status using the specific job ID
      const pollInterval = setInterval(async () => {
        try {
          const jobResponse = await api.get<ScrapingJob>(`/api/v1/scraping/${response.data.id}`)
          const currentJob = jobResponse.data

          setJob(currentJob)

          if (currentJob.status === "completed") {
            clearInterval(pollInterval)
            setPollIntervalId(null)
            toast({
              title: "Success",
              description: `Successfully analyzed ${currentJob.reviews_created} reviews!`,
            })
            setIsLoading(false)
            setTimeout(() => {
              setOpen(false)
              setUrl("")
              setJob(null)
              onSuccess?.()
            }, 2000)
          } else if (currentJob.status === "failed") {
            clearInterval(pollInterval)
            setPollIntervalId(null)
            toast({
              title: "Error",
              description: currentJob.error_message || "Failed to scrape product",
              variant: "destructive",
            })
            setIsLoading(false)
          }
        } catch (error) {
          console.error("[v0] Failed to poll job status:", error)
        }
      }, 3000)

      setPollIntervalId(pollInterval)

      // Stop polling after 5 minutes
      const timeout = setTimeout(() => {
        clearInterval(pollInterval)
        setPollIntervalId(null)
        setTimeoutId(null)
        if (isLoading) {
          setIsLoading(false)
          toast({
            title: "Taking longer than expected",
            description: "Your scraping job is still running. Check back later.",
          })
        }
      }, 300000)

      setTimeoutId(timeout)
    } catch (error: unknown) {
      const errorMessage =
        error && typeof error === "object" && "response" in error
          ? String((error as { response?: { data?: { detail?: string } } }).response?.data?.detail)
          : "Failed to start scraping"

      toast({
        title: "Error",
        description: errorMessage || "Failed to start scraping. Please try again.",
        variant: "destructive",
      })
      setIsLoading(false)
    }
  }

  const getProgress = () => {
    if (!job) return 0
    if (job.status === "pending") return 10
    if (job.status === "running") {
      return Math.min(90, 10 + (job.reviews_scraped / Math.max(job.reviews_scraped, 50)) * 80)
    }
    if (job.status === "completed") return 100
    return 0
  }

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <Button>
          <Plus className="h-4 w-4 mr-2" />
          Add Product
        </Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>Add New Product</DialogTitle>
          <DialogDescription>Enter an eMAG product URL to start analyzing reviews</DialogDescription>
        </DialogHeader>
        <form onSubmit={handleSubmit}>
          <div className="space-y-4 py-4">
            <div className="space-y-2">
              <Label htmlFor="url">Product URL</Label>
              <Input
                id="url"
                placeholder="https://www.emag.ro/..."
                value={url}
                onChange={(e) => setUrl(e.target.value)}
                required
                disabled={isLoading}
              />
            </div>

            {job && (
              <div className="space-y-2">
                <div className="flex items-center justify-between text-sm">
                  <span className="text-muted-foreground">Status: {job.status}</span>
                  {job.status === "completed" && <CheckCircle2 className="h-4 w-4 text-[var(--sentiment-positive)]" />}
                  {job.status === "failed" && <XCircle className="h-4 w-4 text-[var(--sentiment-negative)]" />}
                </div>
                <Progress value={getProgress()} className="h-2" />
                <p className="text-xs text-muted-foreground">
                  {job.reviews_scraped > 0 && `${job.reviews_scraped} reviews scraped`}
                </p>
              </div>
            )}
          </div>
          <DialogFooter>
            <Button type="submit" disabled={isLoading}>
              {isLoading && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
              {isLoading ? "Analyzing..." : "Start Analysis"}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  )
}
