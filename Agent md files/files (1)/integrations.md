# Seizure-Sentry: Integrations, Plugins and Cloud Setup (for Antigravity)

Read after PRD.md and architecture.md. This file says **which tools to connect, how to connect them, and how the agent should use them** to build and deploy the project.

## 1. Decisions

| Need | Tool | How the agent uses it |
|---|---|---|
| Source control, PRs, CI | **GitHub** (private repo) | GitHub MCP server + Git. One branch and one PR per phase. GitHub Actions for tests. |
| Backend hosting (API, WebSocket, ML) | **Render** (Docker web service) | `render.yaml` blueprint + auto-deploy from GitHub. Render MCP if it connects. |
| Frontend hosting | **Vercel** | Auto-deploy from the same GitHub repo (root directory `frontend`). Vercel CLI for env vars and checks. |
| Extra cloud (optional, later) | **AWS** (S3 for recordings, or EC2 instead of Render) | Only if Render's storage limits become a problem. Not needed for the first deployment. |

Why Render for the backend: it gives **automatic HTTPS and WSS** with no domain or Caddy setup. That removes the main deployment blocker (the browser's Web Serial and Web Push need HTTPS, and an HTTPS page cannot open an insecure `ws://` socket). `architecture.md` section 12 (Caddy on EC2) becomes the **alternative**, not the default.

## 2. Things the owner does by hand (the agent cannot)

1. Create a **private** GitHub repository `seizure-sentry`.
2. Create a GitHub **fine-grained personal access token** limited to that one repository (repository permissions: Contents read/write, Pull requests read/write, Issues read/write, Workflows read/write, Metadata read). Set a short expiry (30 to 90 days).
3. Create a **Render** account and an **API key** (Account settings, API keys).
4. Create a **Vercel** account and connect it to GitHub.
5. Install on the laptop: Git, Node.js 20+, Python 3.11+, Docker Desktop (optional but useful), and the **FTDI VCP driver** if Windows does not detect the dongle.
6. Never paste keys into chat, issues, commits or docs.

## 3. Connect the tools to Antigravity

### 3.1 Where MCP servers are configured
In Antigravity: Agent panel, "..." menu, **MCP Servers**, **Manage MCP Servers**, **View raw config**. This edits `mcp_config.json` (shared location is `%USERPROFILE%\.gemini\config\mcp_config.json` on Windows). **Restart Antigravity completely** after saving. Always merge new servers into the existing file; never overwrite other entries.

> Antigravity may not pass environment variables to MCP servers reliably, so tokens may need to sit in that user-level file. That file lives **outside the repository**. Never copy it into the repo, and rotate the tokens if the file is ever shared.

### 3.2 GitHub MCP (confirmed Antigravity setup)
Remote server with a personal access token (OAuth is not yet supported in Antigravity):
```json
{
  "mcpServers": {
    "github": {
      "serverUrl": "https://api.githubcopilot.com/mcp/",
      "headers": { "Authorization": "Bearer <YOUR_FINE_GRAINED_GITHUB_PAT>" }
    }
  }
}
```
Local alternative (needs Docker running):
```json
{
  "mcpServers": {
    "github": {
      "command": "docker",
      "args": ["run", "-i", "--rm", "-e", "GITHUB_PERSONAL_ACCESS_TOKEN", "ghcr.io/github/github-mcp-server"],
      "env": { "GITHUB_PERSONAL_ACCESS_TOKEN": "<YOUR_FINE_GRAINED_GITHUB_PAT>" }
    }
  }
}
```

### 3.3 Render MCP
Render's official MCP endpoint is `https://mcp.render.com/mcp` and accepts an API key as a bearer token. Render documents setups for Claude Code, Codex and Cursor; Antigravity is not listed, so **treat this as "try it"**:
```json
{
  "mcpServers": {
    "render": {
      "serverUrl": "https://mcp.render.com/mcp",
      "headers": { "Authorization": "Bearer <YOUR_RENDER_API_KEY>" }
    }
  }
}
```
If it does not connect, nothing is lost: deploy through the `render.yaml` Blueprint and the Render dashboard (section 5). The Render API key has account-wide access, so keep human confirmation on for every mutating action.

### 3.4 Vercel
Vercel's official MCP server (`https://mcp.vercel.com`) uses OAuth and only works with clients Vercel has approved. Antigravity was not on the supported list when this was written, so **do not depend on it**. Use:
- **Git integration (primary):** in Vercel, import the GitHub repo, set **Root Directory** to `frontend`, set environment variables, deploy. Every push to `main` deploys production; every PR gets a preview URL.
- **Vercel CLI (helper):** `npm i -g vercel`, `vercel login`, `vercel link`, `vercel env add`, `vercel logs`.
- If a later check shows Antigravity is supported: `npx add-mcp https://mcp.vercel.com` then authorize in the browser.

### 3.5 Optional agent skills
Optional, only if wanted: `npx skills add https://github.com/vercel-labs/agent-skills -y -g --skill vercel-react-best-practices` (React/Vercel guidance). Review any skill before installing it.

### 3.6 Verify the connections
Ask the agent: "List my GitHub repositories", "List my Render services". Both must answer from the tools, not from guesses. If a tool is missing, fix the config before building.

## 4. Repository and GitHub workflow

- Private repo, `main` protected (require PR + passing CI). Branch per phase: `phase-0-foundation`, `phase-1-pipeline`, and so on. One PR per phase with the acceptance checklist from phases.md in the description.
- Conventional commit messages (`feat:`, `fix:`, `test:`, `docs:`, `chore:`).
- Enable **secret scanning and push protection**, and **Dependabot** alerts.
- `.gitignore` must exclude: `.env*`, `data/`, `*.edf`, `backend/samples/` (see licence note), `node_modules/`, `__pycache__/`, `.venv/`, build output.
- Model file: commit only if it is under GitHub's 100 MB file limit; otherwise use **Git LFS** or download it at build time from private storage.
- **Dataset licence note:** TUSZ/EEG recordings come with a data-use agreement. Keep the repo **private**, do not publish raw EDFs, and ask the owner which sample files are allowed in the deployed image.

### CI (`.github/workflows/ci.yml`)
```yaml
name: ci
on:
  pull_request:
  push:
    branches: [main]
jobs:
  backend:
    runs-on: ubuntu-latest
    defaults: { run: { working-directory: backend } }
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - run: pip install -r requirements.txt -r requirements-dev.txt
      - run: pytest -q
  frontend:
    runs-on: ubuntu-latest
    defaults: { run: { working-directory: frontend } }
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with: { node-version: "20", cache: npm, cache-dependency-path: frontend/package-lock.json }
      - run: npm ci
      - run: npm run lint --if-present
      - run: npm run typecheck --if-present
      - run: npm run build
```
Pin action versions the way the repo's policy requires and keep the workflow minimal.

## 5. Backend on Render

### 5.1 Blueprint (`render.yaml` in the repo root)
```yaml
services:
  - type: web
    name: seizure-sentry-api
    runtime: docker
    dockerfilePath: ./backend/Dockerfile
    dockerContext: ./backend
    plan: free            # see 5.2 before the real demo
    healthCheckPath: /health
    autoDeploy: true
    envVars:
      - key: ACCESS_KEY
        sync: false       # entered in the Render dashboard, never in git
      - key: VAPID_PUBLIC_KEY
        sync: false
      - key: VAPID_PRIVATE_KEY
        sync: false
      - key: ALLOWED_ORIGIN
        value: https://<your-vercel-domain>
      - key: MODEL_PATH
        value: /app/models/model.onnx
      - key: DATA_DIR
        value: /data
    # Persistent recordings need a paid instance with a disk. Uncomment after upgrading:
    # plan: starter
    # disk:
    #   name: data
    #   mountPath: /data
    #   sizeGB: 1
```
The container must listen on `0.0.0.0` and Render's `PORT` variable:
`uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-10000}`.

### 5.2 Free-tier behaviour (verify current limits on Render's pricing page)
- Free web services **spin down after a period of inactivity** and take a while to wake. Wake the backend (open `/health`) **5 minutes before any demo**, and show the "Starting analysis server" state in the UI (design.md section 6).
- The free instance filesystem is **not persistent** across restarts and deploys, and has limited RAM. Use CPU-only **ONNX Runtime** and keep the model small. If recordings must persist, either upgrade to a paid instance with a disk or send recordings to S3 (section 7).
- WebSockets are supported on Render web services; keep messages small and send an application-level ping so idle sockets are not dropped.

### 5.3 Frontend to backend
Vercel env vars: `VITE_API_URL=https://seizure-sentry-api.onrender.com` (or the equivalent for the framework in use) and **no secrets**. The access key is typed by the owner in Settings at runtime. Backend `ALLOWED_ORIGIN` must exactly match the Vercel production domain (add preview domains only when needed).

## 6. Frontend on Vercel

- Project root `frontend`, framework preset auto-detected, build command `npm run build`.
- Required headers: none special for Web Serial, but pages must be served over HTTPS (Vercel does this).
- If a Permissions-Policy header is added later, it must **not** block `serial` or `bluetooth` for the app origin.
- Service worker file for Web Push must be served from the site root (`/sw.js`).

## 7. Optional AWS (only if needed later)

- **S3** bucket (private, server-side encryption) for recordings and reports, accessed from the Render backend with a least-privilege IAM user (`s3:PutObject`, `GetObject`, `DeleteObject` on one bucket prefix). Keys only in Render env vars.
- **EC2 + Caddy** deployment as in architecture.md section 12 if Render proves unsuitable.
- Create budgets/alerts before using any paid AWS service. The agent must ask before creating any billable resource.

## 8. Secrets matrix

| Secret | Set in | Notes |
|---|---|---|
| `ACCESS_KEY` | Render env | long random string; typed once into the app Settings |
| `VAPID_PUBLIC_KEY` / `VAPID_PRIVATE_KEY` | Render env (public key also read by frontend via API) | generate with `pywebpush`/`web-push` tooling |
| GitHub PAT | Antigravity user config only | fine-grained, one repo, short expiry |
| Render API key | Antigravity user config only | account-wide, treat as sensitive |
| AWS keys (optional) | Render env | least privilege |

Rotate every secret after the demo period or if exposed.

## 9. Rules for the agent when using these tools

1. Ask before creating anything that **costs money**, deleting any resource, rotating secrets, or changing DNS.
2. Never print, log or commit a secret. Use `.env.example` with empty values.
3. Merge into existing config files; never overwrite them.
4. Prefer PRs over direct pushes to `main`. Wait for CI to pass.
5. After each deployment, report: URL, commit SHA, `/health` result, and any warnings. Report only what was verified.
6. If an integration cannot connect, say so and use the fallback in this file instead of improvising.

## 10. Setup order (checklist)

1. [ ] Private GitHub repo created; scaffold pushed (Phase 0).
2. [ ] GitHub MCP connected and verified.
3. [ ] CI workflow added; first PR passes.
4. [ ] Render blueprint created; backend deployed; `/health` returns OK over HTTPS.
5. [ ] Render MCP tried (optional).
6. [ ] Vercel project linked to the repo; preview deploy works.
7. [ ] CORS and access key verified between Vercel frontend and Render backend (a WSS connection opens and a test frame gets an ack).
8. [ ] Web Push key pair generated and a test notification received.
