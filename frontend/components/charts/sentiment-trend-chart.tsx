"use client"

import { useState, useEffect } from "react"
import { CartesianGrid, Legend, Line, LineChart, Tooltip, XAxis, YAxis } from "recharts"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import type { ProductTrend } from "@/types/api"

interface SentimentTrendChartProps {
  data: ProductTrend
}

export function SentimentTrendChart({ data }: SentimentTrendChartProps) {
  const [mounted, setMounted] = useState(false)
  useEffect(() => setMounted(true), [])

  const chartData = data.data_points.map((point) => ({
    date: new Date(point.date).toLocaleDateString("en-US", { month: "short", day: "numeric" }),
    sentiment: point.average_sentiment ? parseFloat(point.average_sentiment.toFixed(2)) : 0,
    rating: point.average_rating ? parseFloat(point.average_rating.toFixed(1)) : 0,
    count: point.count,
  }))

  return (
    <Card>
      <CardHeader>
        <CardTitle>Sentiment Trend Over Time</CardTitle>
      </CardHeader>
      <CardContent className="flex justify-center overflow-x-auto" style={{ minHeight: 200 }}>
        {!mounted ? (
          <div style={{ height: 280 }} />
        ) : (
          <LineChart width={350} height={280} data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" opacity={0.5} />
            <XAxis dataKey="date" stroke="#6b7280" tick={{ fill: "#374151" }} />
            <YAxis yAxisId="left" domain={[0, 1]} stroke="#22c55e" tick={{ fill: "#6b7280" }} />
            <YAxis yAxisId="right" orientation="right" domain={[0, 5]} stroke="#f59e0b" tick={{ fill: "#6b7280" }} />
            <Tooltip
              contentStyle={{ backgroundColor: "#fff", border: "1px solid #e5e7eb", borderRadius: "8px" }}
              labelStyle={{ color: "#111827" }}
            />
            <Legend wrapperStyle={{ color: "#111827" }} />
            <Line
              yAxisId="left" type="monotone" dataKey="sentiment" stroke="#22c55e" strokeWidth={3}
              name="Sentiment Score" dot={{ fill: "#22c55e", strokeWidth: 2, r: 5 }} activeDot={{ r: 8 }}
            />
            <Line
              yAxisId="right" type="monotone" dataKey="rating" stroke="#f59e0b" strokeWidth={3}
              name="Average Rating" dot={{ fill: "#f59e0b", strokeWidth: 2, r: 5 }} activeDot={{ r: 8 }}
            />
          </LineChart>
        )}
      </CardContent>
    </Card>
  )
}
