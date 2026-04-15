"use client"

import { useState, useEffect } from "react"
import { Cell, Pie, PieChart, Legend, Tooltip } from "recharts"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import type { SentimentDistribution } from "@/types/api"

interface SentimentDistributionChartProps {
  data: SentimentDistribution
}

const COLORS = {
  positive: "#14b8a6",
  negative: "#475569",
}

export function SentimentDistributionChart({ data }: SentimentDistributionChartProps) {
  const [mounted, setMounted] = useState(false)
  useEffect(() => setMounted(true), [])

  const allPositive = data.negative === 0 && data.positive > 0
  const allNegative = data.positive === 0 && data.negative > 0

  const chartData = [
    { name: "Positive", value: data.positive, color: COLORS.positive },
    { name: "Negative", value: data.negative, color: COLORS.negative },
  ]

  return (
    <Card>
      <CardHeader>
        <CardTitle>Sentiment Distribution</CardTitle>
      </CardHeader>
      <CardContent className="flex flex-col items-center overflow-hidden">
        {!mounted ? (
          <div style={{ height: 280 }} />
        ) : allPositive || allNegative ? (
          <div className="flex items-center justify-center" style={{ height: 280 }}>
            <div className="relative flex items-center justify-center">
              {/* Outer ghost ring */}
              <div
                className="rounded-full absolute"
                style={{
                  width: 220,
                  height: 220,
                  backgroundColor: allPositive ? COLORS.positive : COLORS.negative,
                  opacity: 0.15,
                }}
              />
              {/* Main circle */}
              <div
                className="rounded-full flex flex-col items-center justify-center relative z-10"
                style={{
                  width: 190,
                  height: 190,
                  backgroundColor: allPositive ? COLORS.positive : COLORS.negative,
                }}
              >
                <span className="text-white text-3xl font-bold">100%</span>
                <span className="text-white text-sm font-medium opacity-90">
                  {allPositive ? "Positive" : "Negative"}
                </span>
              </div>
            </div>
          </div>
        ) : (
          <PieChart width={320} height={280}>
            <Pie data={chartData} cx={160} cy={130} outerRadius={110} dataKey="value" labelLine={false}>
              {chartData.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={entry.color} />
              ))}
            </Pie>
            <Tooltip
              contentStyle={{ backgroundColor: "#fff", border: "1px solid #e5e7eb", borderRadius: "8px" }}
              labelStyle={{ color: "#111827" }}
            />
            <Legend wrapperStyle={{ color: "#111827" }} />
          </PieChart>
        )}
        <div className="mt-2 grid grid-cols-2 gap-4 text-center w-full">
          <div>
            <div className="text-2xl font-bold" style={{ color: COLORS.positive }}>{data.positive}</div>
            <div className="text-xs text-muted-foreground">Positive</div>
          </div>
          <div>
            <div className="text-2xl font-bold" style={{ color: COLORS.negative }}>{data.negative}</div>
            <div className="text-xs text-muted-foreground">Negative</div>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
