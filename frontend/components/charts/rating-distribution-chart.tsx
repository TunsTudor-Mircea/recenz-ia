"use client"

import { useState, useEffect } from "react"
import { Bar, BarChart, CartesianGrid, Tooltip, XAxis, YAxis, Cell } from "recharts"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import type { RatingDistribution } from "@/types/api"

interface RatingDistributionChartProps {
  data: RatingDistribution
}

const CHART_COLOR = "#14b8a6"

export function RatingDistributionChart({ data }: RatingDistributionChartProps) {
  const [mounted, setMounted] = useState(false)
  useEffect(() => setMounted(true), [])

  const chartData = [
    { rating: "5 ★", count: data.rating_5 },
    { rating: "4 ★", count: data.rating_4 },
    { rating: "3 ★", count: data.rating_3 },
    { rating: "2 ★", count: data.rating_2 },
    { rating: "1 ★", count: data.rating_1 },
  ]

  const maxCount = Math.max(data.rating_1, data.rating_2, data.rating_3, data.rating_4, data.rating_5, 1)

  return (
    <Card>
      <CardHeader>
        <CardTitle>Rating Distribution</CardTitle>
      </CardHeader>
      <CardContent className="flex justify-center overflow-x-auto">
        {!mounted ? (
          <div style={{ height: 280 }} />
        ) : (
          <BarChart width={380} height={280} data={chartData} layout="vertical">
            <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" opacity={0.5} />
            <XAxis
              type="number"
              allowDecimals={false}
              domain={[0, maxCount + 0.5]}
              tickCount={Math.min(maxCount + 2, 7)}
              stroke="#6b7280"
              tick={{ fill: "#6b7280" }}
            />
            <YAxis type="category" dataKey="rating" stroke="#6b7280" tick={{ fill: "#374151" }} width={36} />
            <Tooltip
              contentStyle={{ backgroundColor: "#fff", border: "1px solid #e5e7eb", borderRadius: "8px" }}
              labelStyle={{ color: "#111827" }}
            />
            <Bar dataKey="count" radius={[0, 4, 4, 0]}>
              {chartData.map((_, index) => (
                <Cell key={`cell-${index}`} fill={CHART_COLOR} />
              ))}
            </Bar>
          </BarChart>
        )}
      </CardContent>
    </Card>
  )
}
