"use client"

import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import type { ProductTrend } from "@/types/api"

interface SentimentTrendChartProps {
  data: ProductTrend
}

export function SentimentTrendChart({ data }: SentimentTrendChartProps) {
  const chartData = data.data_points.map((point) => ({
    date: new Date(point.date).toLocaleDateString("en-US", { month: "short", day: "numeric" }),
    sentiment: point.average_sentiment.toFixed(2),
    rating: point.average_rating.toFixed(1),
    count: point.count,
  }))

  return (
    <Card className="lg:col-span-2">
      <CardHeader>
        <CardTitle>Sentiment Trend Over Time</CardTitle>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="date" />
            <YAxis yAxisId="left" domain={[0, 1]} />
            <YAxis yAxisId="right" orientation="right" domain={[0, 5]} />
            <Tooltip />
            <Legend />
            <Line
              yAxisId="left"
              type="monotone"
              dataKey="sentiment"
              stroke="hsl(var(--chart-1))"
              strokeWidth={2}
              name="Sentiment Score"
            />
            <Line
              yAxisId="right"
              type="monotone"
              dataKey="rating"
              stroke="hsl(var(--chart-2))"
              strokeWidth={2}
              name="Average Rating"
            />
          </LineChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  )
}
