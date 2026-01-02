"use client"

import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Label } from "@/components/ui/label"

interface ReviewFiltersProps {
  sentiment: string
  setSentiment: (value: string) => void
  rating: string
  setRating: (value: string) => void
  sortBy: string
  setSortBy: (value: string) => void
}

export function ReviewFilters({ sentiment, setSentiment, rating, setRating, sortBy, setSortBy }: ReviewFiltersProps) {
  return (
    <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
      <div className="space-y-2">
        <Label>Sentiment</Label>
        <Select value={sentiment} onValueChange={setSentiment}>
          <SelectTrigger>
            <SelectValue placeholder="All sentiments" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All sentiments</SelectItem>
            <SelectItem value="positive">Positive</SelectItem>
            <SelectItem value="neutral">Neutral</SelectItem>
            <SelectItem value="negative">Negative</SelectItem>
          </SelectContent>
        </Select>
      </div>

      <div className="space-y-2">
        <Label>Rating</Label>
        <Select value={rating} onValueChange={setRating}>
          <SelectTrigger>
            <SelectValue placeholder="All ratings" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All ratings</SelectItem>
            <SelectItem value="5">5 stars</SelectItem>
            <SelectItem value="4">4+ stars</SelectItem>
            <SelectItem value="3">3+ stars</SelectItem>
            <SelectItem value="2">2+ stars</SelectItem>
            <SelectItem value="1">1+ stars</SelectItem>
          </SelectContent>
        </Select>
      </div>

      <div className="space-y-2">
        <Label>Sort by</Label>
        <Select value={sortBy} onValueChange={setSortBy}>
          <SelectTrigger>
            <SelectValue placeholder="Sort by" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="date_desc">Newest first</SelectItem>
            <SelectItem value="date_asc">Oldest first</SelectItem>
            <SelectItem value="rating_desc">Highest rating</SelectItem>
            <SelectItem value="rating_asc">Lowest rating</SelectItem>
            <SelectItem value="sentiment_desc">Most positive</SelectItem>
            <SelectItem value="sentiment_asc">Most negative</SelectItem>
          </SelectContent>
        </Select>
      </div>
    </div>
  )
}
