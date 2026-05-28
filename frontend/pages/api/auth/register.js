/**
 * Registration proxy — forwards sign-up requests to the FastAPI backend.
 * We keep this as a separate API route (not in [...nextauth]) so the
 * register page can call it directly without NextAuth involvement.
 */

const BACKEND_URL =
  process.env.NEXT_PUBLIC_BACKEND_URL || "http://127.0.0.1:8000";

export default async function handler(req, res) {
  if (req.method !== "POST") {
    return res.status(405).json({ error: "Method not allowed" });
  }

  const { email, name, password } = req.body;

  if (!email || !name || !password) {
    return res.status(400).json({ error: "All fields are required" });
  }

  try {
    const backendRes = await fetch(`${BACKEND_URL}/api/auth/register`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, name, password }),
    });

    const data = await backendRes.json();

    if (!backendRes.ok) {
      return res.status(backendRes.status).json({ error: data.detail || "Registration failed" });
    }

    return res.status(201).json(data);
  } catch (err) {
    return res.status(500).json({ error: "Could not reach the backend" });
  }
}
