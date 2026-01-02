import { Star } from "lucide-react"

interface StarRatingProps {
  rating: number
  maxRating?: number
  size?: number
  showNumber?: boolean
}

export function StarRating({ rating, maxRating = 5, size = 16, showNumber = true }: StarRatingProps) {
  return (
    <div className="flex items-center gap-1">
      {Array.from({ length: maxRating }).map((_, i) => (
        <Star
          key={i}
          size={size}
          className={
            i < Math.floor(rating)
              ? "fill-[var(--sentiment-neutral)] text-[var(--sentiment-neutral)]"
              : "text-muted-foreground/30"
          }
        />
      ))}
      {showNumber && <span className="ml-1 text-sm font-medium">{rating.toFixed(1)}</span>}
    </div>
  )
}
