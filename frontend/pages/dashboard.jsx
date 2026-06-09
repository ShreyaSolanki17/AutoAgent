/**
 * Evaluation Dashboard Page
 *
 * Shows aggregated quality metrics from both LLM-as-judge scoring and user feedback.
 */

import { useState, useEffect } from "react";
import { useSession } from "next-auth/react";
import { useRouter } from "next/router";
import axios from "axios";

const API_URL = process.env.NEXT_PUBLIC_BACKEND_URL || "http://127.0.0.1:8000";

export default function Dashboard() {
  const { data: session, status } = useSession();
  const router = useRouter();
  const [stats, setStats] = useState(null);
  const [recent, setRecent] = useState([]);
  const [days, setDays] = useState(7);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (status === "unauthenticated") {
      router.replace("/login");
    }
  }, [status, router]);

  useEffect(() => {
    if (!session) return;
    setLoading(true);
    const token = session.user.accessToken;
    const headers = token ? { Authorization: `Bearer ${token}` } : {};

    Promise.all([
      axios.get(`${API_URL}/api/evaluations/stats?days=${days}`, { headers }),
      axios.get(`${API_URL}/api/evaluations/recent?limit=50`, { headers }),
    ])
      .then(([statsRes, recentRes]) => {
        setStats(statsRes.data);
        setRecent(recentRes.data?.evaluations || []);
      })
      .catch((err) => {
        console.error("Failed to load dashboard data:", err);
      })
      .finally(() => setLoading(false));
  }, [session, days]);

  if (status === "loading" || !session) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-900">
        <div className="flex items-center gap-3 text-gray-400">
          <span className="w-5 h-5 border-2 border-gray-500 border-t-blue-500 rounded-full animate-spin" />
          Loading…
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-900 text-white p-6">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold">Evaluation Dashboard</h1>
          <p className="text-sm text-gray-400 mt-1">
            Quality metrics from LLM-as-judge scoring and user feedback
          </p>
        </div>
        <div className="flex gap-2">
          {[7, 14, 30].map((d) => (
            <button
              key={d}
              onClick={() => setDays(d)}
              className={`px-4 py-2 rounded text-sm transition-colors ${
                days === d
                  ? "bg-blue-600 text-white"
                  : "bg-gray-700 text-gray-300 hover:bg-gray-600"
              }`}
            >
              {d} days
            </button>
          ))}
        </div>
      </div>

      {loading && (
        <div className="flex items-center gap-3 text-gray-400 py-12 justify-center">
          <span className="w-5 h-5 border-2 border-gray-500 border-t-blue-500 rounded-full animate-spin" />
          Loading…
        </div>
      )}

      {!loading && stats && (
        <>
          {/* Summary Cards */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
            <StatCard
              label="Avg Overall"
              value={`${(stats.avg_overall * 100).toFixed(1)}%`}
              color="text-white"
            />
            <StatCard
              label="Thumbs Up"
              value={stats.thumbs_up_count}
              color="text-green-400"
            />
            <StatCard
              label="Thumbs Down"
              value={stats.thumbs_down_count}
              color="text-red-400"
            />
            <StatCard
              label="Total Evaluated"
              value={stats.total_evaluations}
              color="text-blue-400"
            />
          </div>

          {/* Score Breakdown + Distribution */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
            <div className="bg-gray-800 rounded-lg p-5">
              <h2 className="text-lg font-semibold mb-4">Score Breakdown</h2>
              <ScoreRow label="Relevance" value={stats.avg_relevance} color="bg-blue-500" />
              <ScoreRow label="Accuracy" value={stats.avg_accuracy} color="bg-green-500" />
              <ScoreRow label="Helpfulness" value={stats.avg_helpfulness} color="bg-yellow-500" />
              <ScoreRow label="Safety" value={stats.avg_safety} color="bg-purple-500" />
              <div className="mt-3 pt-3 border-t border-gray-700">
                <ScoreRow
                  label="Overall"
                  value={stats.avg_overall}
                  color="bg-white"
                  bold
                />
              </div>
            </div>

            <div className="bg-gray-800 rounded-lg p-5">
              <h2 className="text-lg font-semibold mb-4">Score Distribution</h2>
              {stats.score_distribution?.length > 0 ? (
                <div className="space-y-1.5">
                  {stats.score_distribution.map((bucket) => {
                    const maxCount = Math.max(
                      ...stats.score_distribution.map((b) => b.count),
                      1,
                    );
                    const pct = Math.round((bucket.count / maxCount) * 100);
                    return (
                      <div key={bucket.score_range} className="flex items-center gap-2">
                        <span className="text-xs text-gray-400 w-16">
                          {bucket.score_range}
                        </span>
                        <div className="flex-1 bg-gray-700 rounded-full h-3">
                          <div
                            className="bg-blue-500 h-3 rounded-full transition-all"
                            style={{ width: `${pct}%` }}
                          />
                        </div>
                        <span className="text-xs text-gray-300 w-8 text-right">
                          {bucket.count}
                        </span>
                      </div>
                    );
                  })}
                </div>
              ) : (
                <p className="text-sm text-gray-500">No scored responses yet.</p>
              )}
            </div>
          </div>

          {/* Per-Day Trend */}
          {stats.daily_breakdown?.length > 0 && (
            <div className="bg-gray-800 rounded-lg p-5 mb-8">
              <h2 className="text-lg font-semibold mb-4">Daily Trend</h2>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="text-gray-400 border-b border-gray-700">
                      <th className="text-left py-2 pr-4">Date</th>
                      <th className="text-center py-2 px-4">Responses</th>
                      <th className="text-center py-2 px-4">Avg Score</th>
                      <th className="text-center py-2 px-4">Feedback</th>
                    </tr>
                  </thead>
                  <tbody>
                    {stats.daily_breakdown.map((day) => (
                      <tr
                        key={day.day}
                        className="border-b border-gray-700/50 hover:bg-gray-700/30"
                      >
                        <td className="py-2 pr-4">{day.day}</td>
                        <td className="text-center py-2 px-4">{day.total}</td>
                        <td className="text-center py-2 px-4">
                          {day.avg_score
                            ? `${(day.avg_score * 100).toFixed(1)}%`
                            : "-"}
                        </td>
                        <td className="text-center py-2 px-4">
                          {day.feedback_count || "-"}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Recent Evaluations */}
          <div className="bg-gray-800 rounded-lg p-5">
            <h2 className="text-lg font-semibold mb-4">Recent Evaluations</h2>
            {recent.length > 0 ? (
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="text-gray-400 border-b border-gray-700">
                      <th className="text-left py-2 pr-4">Question</th>
                      <th className="text-center py-2 px-4">Feedback</th>
                      <th className="text-center py-2 px-4">Relevance</th>
                      <th className="text-center py-2 px-4">Accuracy</th>
                      <th className="text-center py-2 px-4">Helpfulness</th>
                      <th className="text-center py-2 px-4">Safety</th>
                      <th className="text-center py-2 px-4">Overall</th>
                      <th className="text-left py-2 pl-4">Reasoning</th>
                    </tr>
                  </thead>
                  <tbody>
                    {recent.map((ev) => (
                      <tr
                        key={ev.message_id}
                        className="border-b border-gray-700/50 hover:bg-gray-700/30"
                      >
                        <td className="py-2 pr-4 max-w-xs truncate">
                          {ev.user_goal || "-"}
                        </td>
                        <td className="text-center py-2 px-4">
                          {ev.user_feedback === 1
                            ? "👍"
                            : ev.user_feedback === -1
                              ? "👎"
                              : "-"}
                        </td>
                        <td className="text-center py-2 px-4">
                          {fmtScore(ev.scores?.relevance)}
                        </td>
                        <td className="text-center py-2 px-4">
                          {fmtScore(ev.scores?.accuracy)}
                        </td>
                        <td className="text-center py-2 px-4">
                          {fmtScore(ev.scores?.helpfulness)}
                        </td>
                        <td className="text-center py-2 px-4">
                          {fmtScore(ev.scores?.safety)}
                        </td>
                        <td className="text-center py-2 px-4 font-semibold">
                          {fmtScore(ev.scores?.overall)}
                        </td>
                        <td className="py-2 pl-4 max-w-sm truncate text-gray-400">
                          {ev.judge_reasoning || "-"}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <p className="text-sm text-gray-500">
                No evaluations yet. Start chatting to generate quality scores.
              </p>
            )}
          </div>

          {/* Judge prompt explanation */}
          <div className="mt-8 bg-gray-800/50 rounded-lg p-5 border border-gray-700">
            <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wide mb-2">
              How scoring works
            </h3>
            <p className="text-sm text-gray-400">
              Each response is automatically scored by an LLM-as-judge on four
              dimensions: relevance (30%), accuracy (30%), helpfulness (25%), and
              safety (15%). You can also provide thumbs up/down feedback to
              compare human judgment with automated scores.
            </p>
          </div>
        </>
      )}
    </div>
  );
}

function StatCard({ label, value, color = "text-white" }) {
  return (
    <div className="bg-gray-800 rounded-lg p-4">
      <p className="text-sm text-gray-400">{label}</p>
      <p className={`text-2xl font-bold ${color}`}>{value}</p>
    </div>
  );
}

function ScoreRow({ label, value, color, bold = false }) {
  return (
    <div className="flex items-center gap-3 mb-2">
      <span
        className={`text-sm w-20 ${bold ? "text-gray-300 font-semibold" : "text-gray-400"}`}
      >
        {label}
      </span>
      <div className="flex-1 bg-gray-700 rounded-full h-2.5">
        <div
          className={`${color} h-2.5 rounded-full transition-all`}
          style={{ width: `${Math.round((value || 0) * 100)}%` }}
        />
      </div>
      <span
        className={`text-sm w-12 text-right ${bold ? "text-white font-semibold" : "text-gray-300"}`}
      >
        {((value || 0) * 100).toFixed(1)}%
      </span>
    </div>
  );
}

function fmtScore(val) {
  if (val == null || val === undefined) return "-";
  return `${Math.round(val * 100)}%`;
}
