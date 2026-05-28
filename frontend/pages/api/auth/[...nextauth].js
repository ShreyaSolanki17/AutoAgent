/**
 * NextAuth.js Pages Router configuration — v4.
 *
 * Uses the Credentials provider to authenticate against our own
 * FastAPI backend (/api/auth/login).
 */

import NextAuth from "next-auth";
import CredentialsProvider from "next-auth/providers/credentials";

const BACKEND_URL =
  process.env.NEXT_PUBLIC_BACKEND_URL || "http://127.0.0.1:8000";

export default NextAuth({
  // ── Providers ─────────────────────────────────────────
  providers: [
    CredentialsProvider({
      name: "Credentials",
      credentials: {
        email: { label: "Email", type: "email" },
        password: { label: "Password", type: "password" },
      },
      async authorize(credentials) {
        if (!credentials?.email || !credentials?.password) return null;

        try {
          const res = await fetch(`${BACKEND_URL}/api/auth/login`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              email: credentials.email,
              password: credentials.password,
            }),
          });

          if (!res.ok) return null;

          const data = await res.json();

          // Return the shape NextAuth stores in the token
          return {
            id: data.user.id,
            name: data.user.name,
            email: data.user.email,
            accessToken: data.access_token,
          };
        } catch {
          return null;
        }
      },
    }),
  ],

  // ── Session ───────────────────────────────────────────
  session: {
    strategy: "jwt",
    maxAge: 24 * 60 * 60, // 24 hours
  },

  // ── Callbacks ─────────────────────────────────────────
  callbacks: {
    async jwt({ token, user }) {
      // On sign-in, merge the user payload into the JWT
      if (user) {
        token.id = user.id;
        token.accessToken = user.accessToken;
      }
      return token;
    },
    async session({ session, token }) {
      // Expose these fields in useSession()
      if (session.user) {
        session.user.id = token.id;
        session.user.accessToken = token.accessToken;
      }
      return session;
    },
  },

  // ── Pages ────────────────────────────────────────────
  pages: {
    signIn: "/login",
  },

  // ── Security ─────────────────────────────────────────
  secret: process.env.NEXTAUTH_SECRET,
});
