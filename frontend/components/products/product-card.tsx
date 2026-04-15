"use client"

import Link from "next/link"
import { Card, CardContent } from "@/components/ui/card"
import { StarRating } from "@/components/star-rating"
import { Button } from "@/components/ui/button"
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from "@/components/ui/alert-dialog"
import { MessageSquare, Calendar, Trash2, TrendingUp } from "lucide-react"
import type { SentimentDistribution } from "@/types/api"

interface ProductCardProps {
  name: string
  totalReviews: number
  averageRating: number
  sentimentDistribution: SentimentDistribution
  lastUpdated?: string
  onDelete?: (productName: string) => void
}

export function ProductCard({
  name,
  totalReviews,
  averageRating,
  sentimentDistribution,
  lastUpdated,
  onDelete,
}: ProductCardProps) {
  const positiveRate = sentimentDistribution.total > 0
    ? Math.round((sentimentDistribution.positive / sentimentDistribution.total) * 100)
    : 0
  const negativeRate = 100 - positiveRate

  const handleDelete = (e: React.MouseEvent) => {
    e.preventDefault()
    e.stopPropagation()
    if (onDelete) onDelete(name)
  }

  return (
    <Card className="hover:shadow-md transition-all duration-200 hover:border-primary/40 group h-full flex flex-col">
      {/* Header */}
      <div className="flex items-start justify-between gap-2 px-5 pt-5 pb-3">
        <Link href={`/products/${encodeURIComponent(name)}`} className="flex-1 min-w-0">
          <h3 className="font-semibold text-sm leading-snug line-clamp-2 group-hover:text-primary transition-colors">
            {name}
          </h3>
        </Link>
        {onDelete && (
          <AlertDialog>
            <AlertDialogTrigger asChild>
              <Button
                variant="ghost"
                size="icon"
                className="h-7 w-7 text-muted-foreground hover:text-destructive shrink-0 -mt-0.5"
              >
                <Trash2 className="h-3.5 w-3.5" />
              </Button>
            </AlertDialogTrigger>
            <AlertDialogContent>
              <AlertDialogHeader>
                <AlertDialogTitle>Delete Product</AlertDialogTitle>
                <AlertDialogDescription>
                  Are you sure you want to delete "{name}"? This will permanently delete all {totalReviews} reviews for
                  this product. This action cannot be undone.
                </AlertDialogDescription>
              </AlertDialogHeader>
              <AlertDialogFooter>
                <AlertDialogCancel>Cancel</AlertDialogCancel>
                <AlertDialogAction
                  onClick={handleDelete}
                  className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
                >
                  Delete
                </AlertDialogAction>
              </AlertDialogFooter>
            </AlertDialogContent>
          </AlertDialog>
        )}
      </div>

      <Link href={`/products/${encodeURIComponent(name)}`} className="flex flex-col flex-1">
        <CardContent className="px-5 pb-5 flex flex-col flex-1 gap-3">
          {/* Rating row */}
          <div className="flex items-center gap-2">
            <StarRating rating={averageRating} size={15} />
            <span className="text-sm font-semibold">{averageRating.toFixed(1)}</span>
            <span className="text-xs text-muted-foreground ml-auto flex items-center gap-1">
              <MessageSquare className="h-3.5 w-3.5" />
              {totalReviews} reviews
            </span>
          </div>

          {/* Sentiment bar */}
          <div className="space-y-1.5">
            <div className="flex h-2 w-full rounded-full overflow-hidden bg-muted">
              <div
                style={{ width: `${positiveRate}%`, backgroundColor: "#14b8a6", transition: "width 0.4s ease" }}
              />
              <div
                style={{ width: `${negativeRate}%`, backgroundColor: "#f87171" }}
              />
            </div>
            <div className="flex items-center justify-between text-xs">
              <span className="text-teal-600 font-medium">{sentimentDistribution.positive} positive</span>
              <span className="text-rose-400 font-medium">{sentimentDistribution.negative} negative</span>
            </div>
          </div>

          {/* Spacer */}
          <div className="flex-1" />

          {/* Footer */}
          <div className="flex items-center justify-between pt-3 border-t text-xs text-muted-foreground">
            {lastUpdated ? (
              <div className="flex items-center gap-1">
                <Calendar className="h-3 w-3" />
                <span>Updated {new Date(lastUpdated).toLocaleDateString()}</span>
              </div>
            ) : (
              <span />
            )}
            <div className="flex items-center gap-1 font-semibold text-foreground">
              <TrendingUp className="h-3.5 w-3.5 text-teal-500" />
              {positiveRate}%
            </div>
          </div>
        </CardContent>
      </Link>
    </Card>
  )
}
