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
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { useToast } from "@/hooks/use-toast"
import { useScrapingJobWS } from "@/hooks/use-scraping-job-ws"
import api from "@/lib/api"
import { Plus, Loader2, CheckCircle2, XCircle, Wifi, WifiOff } from "lucide-react"
import type { ScrapingJob } from "@/types/api"

interface AddProductModalProps {
  onSuccess?: () => void
}

export function AddProductModal({ onSuccess }: AddProductModalProps) {
  const [open, setOpen] = useState(false)
  const [url, setUrl] = useState("")
  const [modelType, setModelType] = useState("robert")
  const [isLoading, setIsLoading] = useState(false)
  const [jobId, setJobId] = useState<string | null>(null)
  const { toast } = useToast()

  // Use WebSocket hook for real-time job updates
  const { job, isConnected, connectionError } = useScrapingJobWS({
    jobId,
    onStatusChange: (jobUpdate) => {
      console.log("[AddProductModal] Job status changed:", jobUpdate)
    },
    onComplete: (jobUpdate) => {
      toast({
        title: "Success",
        description: `Successfully analyzed ${jobUpdate.reviews_created} reviews!`,
      })
      setIsLoading(false)
      setTimeout(() => {
        setOpen(false)
        setUrl("")
        setJobId(null)
        onSuccess?.()
      }, 2000)
    },
    onError: (jobUpdate) => {
      toast({
        title: "Error",
        description: jobUpdate.error_message || "Failed to scrape product",
        variant: "destructive",
      })
      setIsLoading(false)
    },
  })

  // Reset job when dialog closes
  useEffect(() => {
    if (!open) {
      setJobId(null)
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
        model_type: modelType,
      })

      // Set job ID to trigger WebSocket connection
      setJobId(response.data.id)

      toast({
        title: "Scraping Started",
        description: "Your product is being analyzed. Updates will appear in real-time.",
      })
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
    if (job.status === "in_progress") {
      const scraped = job.reviews_scraped || 0
      return Math.min(90, 10 + (scraped / Math.max(scraped, 50)) * 80)
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

            <div className="space-y-2">
              <Label htmlFor="model">Analysis Model</Label>
              <Select value={modelType} onValueChange={setModelType} disabled={isLoading}>
                <SelectTrigger id="model">
                  <SelectValue placeholder="Select model" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="robert">RoBERT (Recommended - More Accurate)</SelectItem>
                  <SelectItem value="xgboost">XGBoost (Faster)</SelectItem>
                </SelectContent>
              </Select>
              <p className="text-xs text-muted-foreground">
                {modelType === "robert"
                  ? "RoBERT: Deep learning model optimized for Romanian text. More accurate but slower."
                  : "XGBoost: Traditional ML model. Faster processing but slightly less accurate."}
              </p>
            </div>

            {job && (
              <div className="space-y-2">
                <div className="flex items-center justify-between text-sm">
                  <span className="text-muted-foreground">Status: {job.status}</span>
                  <div className="flex items-center gap-2">
                    {isConnected ? (
                      <Wifi className="h-4 w-4 text-green-500" title="Connected" />
                    ) : (
                      <WifiOff className="h-4 w-4 text-gray-400" title="Connecting..." />
                    )}
                    {job.status === "completed" && <CheckCircle2 className="h-4 w-4 text-[var(--sentiment-positive)]" />}
                    {job.status === "failed" && <XCircle className="h-4 w-4 text-[var(--sentiment-negative)]" />}
                  </div>
                </div>
                <Progress value={getProgress()} className="h-2" />
                <p className="text-xs text-muted-foreground">
                  {job.reviews_scraped !== undefined && job.reviews_scraped > 0 && `${job.reviews_scraped} reviews scraped`}
                  {connectionError && <span className="text-red-500"> • {connectionError}</span>}
                </p>
              </div>
            )}
          </div>
          {!job && (
            <DialogFooter>
              <Button type="submit" disabled={isLoading}>
                {isLoading && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
                {isLoading ? "Starting..." : "Start Analysis"}
              </Button>
            </DialogFooter>
          )}
        </form>
      </DialogContent>
    </Dialog>
  )
}
