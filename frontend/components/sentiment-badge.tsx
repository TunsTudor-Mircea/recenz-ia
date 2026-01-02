import { Badge } from "@/components/ui/badge"

interface SentimentBadgeProps {
  sentiment: "positive" | "neutral" | "negative"
  score?: number
  className?: string
}

export function SentimentBadge({ sentiment, score, className }: SentimentBadgeProps) {
  const colors = {
    positive: "bg-[var(--sentiment-positive)] text-white hover:bg-[var(--sentiment-positive)]/90",
    neutral: "bg-[var(--sentiment-neutral)] text-white hover:bg-[var(--sentiment-neutral)]/90",
    negative: "bg-[var(--sentiment-negative)] text-white hover:bg-[var(--sentiment-negative)]/90",
  }

  return (
    <Badge className={`${colors[sentiment]} ${className}`}>
      {sentiment}
      {score !== undefined && ` (${Math.round(score * 100)}%)`}
    </Badge>
  )
}
