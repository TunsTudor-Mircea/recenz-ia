"use client"

import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis, Cell } from "recharts"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import type { RatingDistribution } from "@/types/api"

interface RatingDistributionChartProps {
  data: RatingDistribution
}

const CHART_COLOR = "hsl(var(--chart-4))"

export function RatingDistributionChart({ data }: RatingDistributionChartProps) {
  const chartData = [
    { rating: "5 ★", count: data.rating_5 },
    { rating: "4 ★", count: data.rating_4 },
    { rating: "3 ★", count: data.rating_3 },
    { rating: "2 ★", count: data.rating_2 },
    { rating: "1 ★", count: data.rating_1 },
  ]

  return (
    <Card>
      <CardHeader>
        <CardTitle>Rating Distribution</CardTitle>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={chartData} layout="vertical">
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" />
            <YAxis type="category" dataKey="rating" />
            <Tooltip />
            <Bar dataKey="count" radius={[0, 4, 4, 0]}>
              {chartData.map((_, index) => (
                <Cell key={`cell-${index}`} fill={CHART_COLOR} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  )
}
