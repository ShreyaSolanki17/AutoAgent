/**
 * Home / Chat Page
 *
 * Protected by NextAuth — unauthenticated users are redirected to /login.
 * The navbar shows the logged-in user's name and a Sign Out button.
 */

import { useState, useRef, useEffect } from "react";
import { useSession, signOut } from "next-auth/react";
import { useRouter } from "next/router";
import axios from "axios";

const API_URL = process.env.NEXT_PUBLIC_BACKEND_URL || "http://127.0.0.1:8000";

export default function Home() {
  const { data: session, status } = useSession();
  const router = useRouter();

  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [trace, setTrace] = useState([]);
  const [backendStatus, setBackendStatus] = useState("connecting");
  const messagesEndRef = useRef(null);

  // Redirect to /login if unauthenticated
  useEffect(() => {
    if (status === "unauthenticated") {
      router.replace("/login");
    }
  }, [status, router]);

  // Check backend health on mount
  useEffect(() => {
    checkHealth();
  }, []);

  async function checkHealth() {
    try {
      await axios.get(`${API_URL}/api/health`);
      setBackendStatus("connected");
    } catch {
      setBackendStatus("offline");
    }
  }

  async function sendMessage() {
    if (!input.trim() || isLoading) return;

    const userMessage = input.trim();
    setInput("");
    setMessages((prev) => [...prev, { role: "user", content: userMessage }]);
    setIsLoading(true);
    setTrace([]);

    try {
      const res = await axios.post(`${API_URL}/api/chat`, {
        message: userMessage,
      });

      const data = res.data;
      setTrace(data.trace || []);
      setMessages((prev) => [
        ...prev,
        { role: "agent", content: data.final_answer },
      ]);
    } catch {
      setMessages((prev) => [
        ...prev,
        {
          role: "agent",
          content: "Error: Could not reach the backend. Is it running?",
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  }

  function handleKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  }

  // ── Loading state while session resolves ────────────────
  if (status === "loading") {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-900">
        <div className="flex items-center gap-3 text-gray-400">
          <span className="w-5 h-5 border-2 border-gray-500 border-t-blue-500 rounded-full animate-spin" />
          Loading…
        </div>
      </div>
    );
  }

  // Don't render the chat UI if there is no session (redirect in progress)
  if (!session) return null;

  // ── Main UI ──────────────────────────────────────────────
  return (
    <div className="min-h-screen flex flex-col">
      {/* Header / Navbar */}
      <header className="bg-gray-800 border-b border-gray-700 px-6 py-4 flex items-center justify-between">
        <h1 className="text-xl font-bold text-white">AutoAgent</h1>

        <div className="flex items-center gap-4">
          {/* Backend status */}
          <div className="flex items-center gap-2">
            <span className="text-sm text-gray-400">Backend:</span>
            <span
              className={`w-2 h-2 rounded-full ${
                backendStatus === "connected"
                  ? "bg-green-500"
                  : backendStatus === "offline"
                  ? "bg-red-500"
                  : "bg-yellow-500"
              }`}
            />
            <span className="text-sm text-gray-400">{backendStatus}</span>
          </div>

          {/* Divider */}
          <div className="w-px h-6 bg-gray-600" />

          {/* User info */}
          <span className="text-sm text-gray-300">
            Hi, {session.user.name}
          </span>

          {/* Sign out */}
          <button
            onClick={() => signOut({ callbackUrl: "/login" })}
            className="text-sm text-gray-400 hover:text-white transition-colors px-3 py-1.5 rounded border border-gray-600 hover:border-gray-400"
          >
            Sign out
          </button>
        </div>
      </header>

      {/* Main content */}
      <div className="flex-1 flex overflow-hidden">
        {/* Messages area */}
        <main className="flex-1 flex flex-col">
          {/* Messages list */}
          <div className="flex-1 overflow-y-auto px-6 py-4 space-y-4">
            {messages.length === 0 && (
              <div className="text-center text-gray-500 mt-20">
                <p className="text-2xl mb-2">Hi, I&apos;m AutoAgent</p>
                <p>
                  I can search the web, look up Wikipedia, calculate, and
                  remember our conversations.
                </p>
                <p className="mt-4 text-sm">
                  Try: &quot;What is the capital of France?&quot;
                </p>
              </div>
            )}

            {messages.map((msg, i) => (
              <div
                key={i}
                className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
              >
                <div
                  className={`max-w-[70%] px-4 py-3 rounded-lg ${
                    msg.role === "user"
                      ? "bg-blue-600 text-white"
                      : "bg-gray-800 text-gray-100 border border-gray-700"
                  }`}
                >
                  <p className="whitespace-pre-wrap">{msg.content}</p>
                </div>
              </div>
            ))}

            {isLoading && (
              <div className="flex justify-start">
                <div className="bg-gray-800 border border-gray-700 px-4 py-3 rounded-lg">
                  <div className="flex gap-1">
                    <span className="w-2 h-2 bg-gray-500 rounded-full animate-bounce" />
                    <span className="w-2 h-2 bg-gray-500 rounded-full animate-bounce [animation-delay:0.1s]" />
                    <span className="w-2 h-2 bg-gray-500 rounded-full animate-bounce [animation-delay:0.2s]" />
                  </div>
                </div>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>

          {/* Input area */}
          <div className="border-t border-gray-700 px-6 py-4">
            <div className="flex gap-3">
              <textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Ask me anything..."
                rows={1}
                className="flex-1 bg-gray-800 border border-gray-700 rounded-lg px-4 py-3 text-white placeholder-gray-500 focus:outline-none focus:border-blue-500 resize-none"
              />
              <button
                onClick={sendMessage}
                disabled={isLoading || !input.trim()}
                className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                Send
              </button>
            </div>
          </div>
        </main>

        {/* Trace sidebar */}
        <aside className="w-96 border-l border-gray-700 bg-gray-850 overflow-y-auto">
          <div className="p-4 border-b border-gray-700">
            <h2 className="text-sm font-semibold text-gray-400 uppercase tracking-wide">
              Reasoning Trace
            </h2>
          </div>
          <div className="p-4 space-y-3">
            {trace.length === 0 && (
              <p className="text-gray-500 text-sm">
                Send a message to see the agent&apos;s reasoning steps here.
              </p>
            )}
            {trace.map((step, i) => (
              <div
                key={i}
                className={`text-sm px-3 py-2 rounded border-l-2 ${
                  step.startsWith("[Plan]")
                    ? "border-blue-500 bg-blue-500/10"
                    : step.startsWith("[Act]")
                    ? "border-green-500 bg-green-500/10"
                    : step.startsWith("[Observe]")
                    ? "border-yellow-500 bg-yellow-500/10"
                    : "border-gray-600 bg-gray-800"
                }`}
              >
                <p className="text-gray-300 break-words">{step}</p>
              </div>
            ))}
          </div>
        </aside>
      </div>
    </div>
  );
}
