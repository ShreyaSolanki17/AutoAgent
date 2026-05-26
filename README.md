# AutoAgent — Full-Stack AI Agent with Tool Use and Memory

An autonomous AI agent that accepts a user goal, breaks it into sub-tasks,
uses tools to complete them, and remembers context across sessions.

## Tech Stack

| Layer          | Technology                          |
| -------------- | ----------------------------------- |
| LLM            | Groq API (Llama 3.3 70B)            |
| Agent Loop     | LangGraph                           |
| Backend        | FastAPI                             |
| Frontend       | Next.js + Tailwind CSS              |
| Memory / DB    | PostgreSQL + pgvector               |
| Embeddings     | sentence-transformers (local)       |
| Auth           | NextAuth.js                         |
| Deployment     | Docker Compose → Railway / Render   |

## Project Structure

```
AutoAgent/
├── backend/           # FastAPI + LangGraph agent
│   ├── agent/         # Agent loop, state graph, prompts
│   ├── tools/         # Brave Search, Wikipedia, Calculator, File I/O, Memory
│   ├── memory/        # pgvector store, embedding service
│   ├── api/           # FastAPI routes
│   ├── models/        # Pydantic schemas
│   └── main.py        # FastAPI entry point
├── frontend/          # Next.js app
│   ├── pages/         # Next.js pages (chat, login, dashboard)
│   ├── components/    # Reusable UI components
│   ├── styles/        # Tailwind + custom CSS
│   └── public/        # Static assets
├── docker/            # DB init scripts
├── .env.example       # Environment variable template
└── docker-compose.yml # Service orchestration
```

## Getting Started

> Step-by-step setup instructions will be added as each module is built.

## License

MIT
