# Job Seeker Platform — Complete Architecture Audit

**Scope:** Read-only inspection of the current implementation. No code was changed. Report date 2026-07-24.

**Stack at a glance:** FastAPI (async, Python 3.12) + Celery + PostgreSQL/pgvector + Redis, behind nginx, serving a React 19 / Vite SPA. 14 backend modules, 40 database tables, 7 dedicated Celery worker queues + Beat, a pluggable AI/Embedding/Storage/Email provider layer (env-var driven), and a hand-rolled in-process event bus that turns out to be **dead code** — real cross-module coordination happens via direct Celery task chaining and Redis pub/sub, not the event bus. This distinction matters throughout the report and is central to §17.

---

## Table of Contents
1. [High-Level Architecture](#1-high-level-architecture)
2. [Complete User Workflow](#2-complete-user-workflow)
3. [Module Dependency Graph](#3-module-dependency-graph)
4. [Complete API Flow](#4-complete-api-flow)
5. [Event Flow](#5-event-flow)
6. [Background Workers](#6-background-workers)
7. [Database Flow](#7-database-flow)
8. [AI Flow](#8-ai-flow)
9. [Search Flow](#9-search-flow)
10. [Matching Flow](#10-matching-flow)
11. [Application Flow](#11-application-flow)
12. [Dashboard Data Flow](#12-dashboard-data-flow)
13. [Route Map](#13-route-map)
14. [State Management](#14-state-management)
15. [Complete Project Tree](#15-complete-project-tree)
16. [Automation Flow (Manual Actions)](#16-automation-flow-manual-actions)
17. [Missing Connections](#17-missing-connections)
18. [Recommended Workflow](#18-recommended-workflow)

---

## 1. High-Level Architecture

Everything sits behind a **single public entry point**: `nginx` on port `5000`. Nothing else is published to the host except Postgres/Redis/Ollama's dev ports (5432/6379/11434), which exist for local tooling, not for the app itself.

```
┌───────────────────────────────────────────────────────────────────────┐
│                              BROWSER                                   │
│   React 19 SPA (Vite build) — axios client, React Query, Zustand,      │
│   raw WebSocket (2 sockets: /matching/ws, /workspace/ws)               │
└───────────────────────────────┬───────────────────────────────────────┘
                                 │  http://localhost:5000
                                 ▼
                        ┌─────────────────┐
                        │      nginx      │   reverse proxy, WS upgrade,
                        │   (port 5000)   │   client_max_body_size 20m
                        └───┬─────────┬───┘
                    /api/*  │         │  /*  (everything else)
              ┌─────────────┘         └─────────────┐
              ▼                                     ▼
   ┌────────────────────┐                ┌─────────────────────┐
   │   FastAPI backend   │                │  Vite dev server /   │
   │   uvicorn :8000      │                │  static build :3000  │
   │   app/main.py         │               └─────────────────────┘
   └──┬───────────┬───────┘
      │           │ WebSocket bridge only (matching.ws / workspace.ws
      │           │ subscribe to Redis pub/sub, forward to browser)
      ▼           ▼
┌───────────────────┐        ┌─────────────────────────────┐
│ 14 API modules     │        │        Redis (db 0/1/2)      │
│ (routers→services  │◄──────►│  db0: cache + pub/sub         │
│  →repositories)     │        │  db1: Celery broker           │
└───┬─────────────┬──┘        │  db2: Celery result backend   │
    │             │            └───────────┬──────────────────┘
    ▼             │                        │ task protocol
┌─────────────┐   │                        ▼
│ PostgreSQL   │   │        ┌───────────────────────────────────────┐
│ + pgvector   │   │        │ 7 dedicated Celery workers + Beat       │
│ 40 tables    │   │        │  worker-parsing        (parsing)        │
└─────────────┘   │        │  worker-embedding      (embedding)      │
                   │        │  worker-ingestion      (ingestion,skills)│
                   │        │  worker-matching       (matching)       │
                   │        │  worker-ats-resume     (ats_optimization,│
                   │        │                         ats_files)      │
                   │        │  worker-auto-apply     (auto_apply)     │
                   │        │  worker-email-tracking (email_tracking, │
                   │        │                         notifications)  │
                   │        │  beat                  (scheduler)      │
                   │        └───┬─────────┬──────────┬─────────┬─────┘
                   │             │         │          │         │
                   ▼             ▼         ▼          ▼         ▼
         ┌────────────────┐┌──────────┐┌─────────┐┌─────────┐┌──────────┐
         │ AI Provider      ││ Embedding ││ Storage  ││ Email    ││ (browser  │
         │ (factory-selected)││(factory)  ││(factory) ││(factory) ││ automation)│
         │ ollama(default)  ││ local ST  ││ local FS ││ smtp     ││ Playwright │
         │ anthropic/openai/││ (default) ││ or S3    ││ or SES   ││ (auto_apply│
         │ gemini/groq       ││ voyage/   ││          ││          ││ + job_ing.)│
         │                   ││ openai/   ││          ││          ││            │
         │                   ││ gemini    ││          ││          ││            │
         └────────────────┘└──────────┘└─────────┘└─────────┘└──────────┘
                                                                       │
                                                                       ▼
                                                      ┌────────────────────────┐
                                                      │ Notifications            │
                                                      │ (DB row + email task +   │
                                                      │  push placeholder/no-op) │
                                                      └────────────────────────┘
```

**How modules actually communicate** (three real mechanisms — the formal `EventBus` in `app/events/bus.py` is wired but has zero publishers/subscribers anywhere in the codebase; see §17):

1. **Direct Celery task chaining** — a task's success handler calls `next_task.delay(...)` directly. This is the backbone of every automated pipeline (resume parsing → embedding → scoring → matching, job ingestion → normalization → skills → embedding → matching).
2. **Redis pub/sub bridged to WebSocket** — `workspace/realtime.py` (`workspace:updates:{user_id}`) and `matching/realtime.py` (`matching:updates:{user_id}`) let a Celery worker process (which has no live HTTP connection) notify the API process, which forwards to a connected browser WebSocket.
3. **Synchronous cross-module service calls** — one module directly imports and calls another module's `Service`/`Repository` class within the same request/task (e.g., `user_profile.service` calls `matching.service.upsert_preferences`; `email_tracking.service` calls `tracking.service.change_status`). This is a modular monolith, not microservices — all 14 modules run in the same process and share one database.

**AI/Embedding/Storage/Email are pluggable providers**, selected purely by environment variable (`AI_PROVIDER`, `EMBEDDING_PROVIDER`, `STORAGE_PROVIDER`, `EMAIL_PROVIDER`), resolved through `app/factories/*_provider_factory.py` against `app/interfaces/*.py` `Protocol` contracts, wrapped by two cross-cutting services (`AIService`, `EmbeddingService` in `app/services/`) that add retry (via `tenacity`) and, for embeddings, Redis cache-aside. Default local-dev stack: **Ollama** (LLM) + **local sentence-transformers** (embeddings, 384-dim) + **local filesystem** (storage) + **SMTP** (email) — all swappable to Anthropic/OpenAI/Gemini/Groq, Voyage/OpenAI/Gemini, S3, or SES without code changes.

---

## 2. Complete User Workflow

```
Landing Page (/, public marketing)
   ↓ click "Get Started"
Register  (POST /auth/register)  →  JWT access token + httpOnly refresh cookie issued immediately
   ↓                                  (NO email-verification step exists server-side — see §17)
Login  (POST /auth/login)  — only needed on return visits; register auto-logs-in
   ↓
Onboarding Wizard (/onboarding, gated: user is redirected here until onboarding_completed_at is set)
   ↓ GET /onboarding/status  → { onboarding_completed, has_resume, pipeline_run_id, pipeline_status }
   ↓ collects target_roles, work_authorization, + MatchPreference fields
   ↓ POST /onboarding/complete  →  delegates job-preference fields to MatchingService.upsert_preferences
   ↓  (blocks with error_code=resume_required if no resume uploaded yet)
Resume Upload  (POST /resumes, multipart)
   ↓  ResumeParserService.upload_resume → S3/local upload → Resume + ResumeVersion(status=uploaded) created
   ↓  enqueues Celery: resume_parser.parse_resume
   ┌─────────────────────────── AUTOMATIC PIPELINE (Celery chain, no user action) ───────────────────────────┐
   │ parse_resume (queue parsing)                                                                            │
   │   → pypdf/python-docx text extraction → AIService.extract_structured (Claude/LLM) → ResumeParsedData    │
   │   ↓ triggers                                                                                             │
   │ generate_embedding (queue embedding)                                                                    │
   │   → EmbeddingService.generate_embedding (Voyage/local ST) → ResumeEmbedding (pgvector)                   │
   │   ↓ triggers                                                                                             │
   │ calculate_score (queue embedding, deterministic, no LLM)                                                 │
   │   → ats_resume_generator.scoring.structure_score → ResumeVersion.resume_score                            │
   │   ↓ triggers                                                                                             │
   │ run_ats_baseline_analysis (queue embedding, deterministic)                                                │
   │   → keyword_coverage vs top skills for target_roles → ats_baseline_score, missing_skills                 │
   │   ↓ triggers                                                                                              │
   │ matching.compute_resume_matches (queue matching)                                                          │
   │   → semantic candidate jobs (pgvector cosine) + skill overlap + rule boosts → ResumeJobMatch rows          │
   │   ↓ triggers                                                                                              │
   │ matching.refresh_recommendations (queue matching)                                                         │
   │   → job/company/skill Recommendation rows regenerated                                                     │
   │   ↓ triggers                                                                                              │
   │ workspace.update_dashboard (queue matching)                                                               │
   │   → publishes Redis "workspace_ready" event → WebSocket → dashboard live-updates                          │
   │   ↓ triggers                                                                                              │
   │ notification.send_resume_ready (queue notifications)                                                      │
   │   → Notification row (RESUME_READY) + email send                                                          │
   └──────────────────────────────────────────────────────────────────────────────────────────────────────────┘
   Every step above also calls `_advance_pipeline(...)`, updating `ResumePipelineRun` and publishing a
   `"pipeline_progress"` Redis event — the frontend `PipelineBanner` component polls/subscribes to this to
   show a live progress bar during the ~10–30s the chain takes.
   ↓
Dashboard / Workspace (/workspace)
   ↓  GET /workspace  aggregates: resume health, recommended jobs, applications stats, coach highlights,
   ↓  notifications, "next best actions" — single synchronous call, no AI, composed from 7 other modules
Job Search (/jobs)  — GET /search (hybrid keyword+semantic), independent of the pipeline above
   ↓
Matching (/matching) — view/recompute ResumeJobMatch scores, manage preferences, dismiss recommendations
   ↓  (MANUAL) user must click "Generate ATS Resume" — not auto-triggered by a match
ATS Resume Generation (POST /ats-resumes)
   ↓  optimize_ats_resume_task (LLM rewrite of summary/skills/projects, keyword scoring)
   ↓  triggers generate_ats_resume_files_task (DOCX + PDF via python-docx/reportlab)
   ↓  publishes "ats_resume_ready" Redis event
   ↓  (MANUAL) user must click "Apply" / create an Auto Apply task — not auto-triggered by ATS-resume-ready
Apply (POST /auto-apply/tasks)
   ↓  resume_selection (prefers completed ATS resume) → adapter dispatch (Greenhouse/Lever HTTP,
   ↓  Workday/generic Playwright) → field mapping → validation →
   ↓    ├─ if incomplete/invalid → status=REQUIRES_REVIEW (MANUAL: user edits profile, clicks Retry)
   ↓    └─ if OK → auto-submit → confirmation detection → creates tracking.Application(source=AUTO_APPLY)
   ↓  publishes NotificationEventType.APPLICATION_STATUS_CHANGED via TrackingService
Tracking (/tracker) — Kanban/calendar/timeline of Application status, notes
   ↓  status can also change via Gmail email classification (see below), never automatically from auto-apply
   ↓  beyond the initial "applied" row
Notifications (/notifications) — in-app + email; PUSH channel exists as an interface but is a no-op placeholder
   ↓
Career Coach (/coach) — (MANUAL, fully isolated) resume review, skill-gap, roadmap, interview prep,
   job-readiness score, salary prediction, chat — user must visit and click each tool; nothing upstream
   (resume parsed, ATS score computed, new job matched) ever triggers a coach insight automatically.

Parallel, backend-only loop (no user action):
Gmail connect (/tracker/emails → GET /email-tracking/gmail/connect, OAuth) once connected:
   Celery Beat "gmail-sync-sweep" (every GMAIL_SYNC_INTERVAL_MINUTES, default 10min)
     → poll_gmail_account (Gmail History API) → fetch_and_classify_email (LLM classification)
       → EmailTrackingService.process_email → fuzzy match to an Application → TrackingService.change_status
         → same APPLICATION_STATUS_CHANGED notification + Redis event as a manual status edit

Parallel, backend-only loop (no user action):
Celery Beat "job-ingestion-sweep" (every JOB_INGESTION_SWEEP_INTERVAL_MINUTES, default 360min)
  → fetch_company_jobs (per ATS adapter) → normalize_job → extract_job_skills (LLM) → generate_job_embedding
    → matching.match_new_job_for_active_users ("Smart Trigger": if score ≥ threshold, notifies matching users)
```

---

## 3. Module Dependency Graph

14 backend modules under `backend/app/modules/`. Arrows mean "imports/calls into" (the tail module depends on the head module).

```
auth  ◄── user_profile
auth  ◄── tracking (lazy, for user email lookup during notify)
auth  ◄── matching (repository, notification recipient lookup)

user_profile ──► matching        (delegates job-preference upsert)
user_profile ──► resume_parser   (reads ResumePipelineRunRepository for onboarding gating)

resume_parser ──► ats_resume_generator  (reuses scoring.py for baseline ATS score)
resume_parser ──► user_profile          (reads target_roles for baseline analysis)
resume_parser ──► job_ingestion         (reads Job/Skill for "top skills for role")
resume_parser ──► matching              (hands off to compute_resume_matches at pipeline end)
resume_parser ──► workspace             (publishes pipeline_progress realtime events)

ats_resume_generator ──► resume_parser   (source resume content)
ats_resume_generator ──► job_ingestion   (target job existence check)
ats_resume_generator ──► workspace       (publishes ats_resume_ready event)
ats_resume_generator ◄── auto_apply      (resume_selection prefers completed ATS versions)

job_ingestion ──► matching   (triggers match_new_job_for_active_users after embedding)

search ──► job_ingestion   (read-only queries; search owns no tables of its own)

matching ──► resume_parser, job_ingestion, notification, auth, workspace

auto_apply ──► job_ingestion, ats_resume_generator, resume_parser, tracking

tracking ──► notification, auth (lazy), workspace (lazy)

email_tracking ──► tracking, notification, services.ai

notification (modules/) ──► services/notification/* (push_sender, templates, errors)
notification ──► auth (lazy, email lookup)

analytics ──► ats_resume_generator, job_ingestion, resume_parser, tracking   (read-only aggregation)

ai_career_coach ──► resume_parser, job_ingestion, services.ai

workspace ──► ai_career_coach, matching, notification, resume_parser, tracking, user_profile
              (workspace is the most-connected module — it's the aggregation/composition root)

admin ──► job_ingestion, tracking, auto_apply, notification, email_tracking
          (admin is read/moderate-only; nothing depends on admin)
```

Modules with **zero outbound dependencies** on other modules: `auth` (fully self-contained identity), `search` (pure read layer over job_ingestion, but architecturally standalone).

Modules with **zero Celery tasks** (pure request/response): `auth`, `user_profile`, `admin`, `search`, `tracking`, `analytics`, `ai_career_coach`, `workspace` — note `workspace` and `tracking` still *trigger* tasks/events in other modules synchronously, they just don't own any `tasks.py` themselves.

---

## 4. Complete API Flow

Format per user-facing flow: `Frontend Page → API Endpoint → Service → Repository → DB Tables → Response`.

**Resume Upload**
```
ResumeManagerPage → POST /resumes → ResumeParserService.upload_resume → ResumeRepository/ResumeVersionRepository
  → INSERT resume, resume_version (S3/local write via StorageProvider) → 201 { id, versions:[...] }
```

**Resume Parsing (async, not directly page-driven)**
```
Celery parse_resume_task → ResumeParserService internals → ResumeParsedDataRepository
  → INSERT resume_parsed_data, UPDATE resume_version.status → (chains to next task, no HTTP response)
```

**ATS Resume Generation**
```
ATSResumeGeneratorPage → POST /ats-resumes → ATSResumeGeneratorService.create_ats_resume
  → ATSResumeRepository/ATSResumeVersionRepository → INSERT ats_resume, ats_resume_version (status=pending)
  → 201 { id, versions:[{status:"pending"}] }   (optimize/files happen async; poll GET .../versions/{id})
```

**Job Search**
```
SearchPage → GET /search?q=&mode=hybrid&filters → SearchService.search → SearchRepository
  (Postgres FTS ts_rank_cd + pgvector cosine_distance, fused) → reads job/company/location/skill/job_embedding
  → 200 PaginatedResponse[JobSearchResult]  (cached in Redis via CacheClient keyed search:v1:query:*)
```

**Matching**
```
MatchingPage → POST /matching/resumes/{id}/compute → MatchingService.enqueue_batch_compute
  → Celery matching.compute_resume_matches (queue matching) → MatchingRepository
  → UPSERT resume_job_match rows (reads resume_embedding, job_embedding, skill, job_skill)
  → 202 Accepted; frontend then GET /matching/resumes/{id}/matches or listens on WS /matching/ws
```

**Auto Apply**
```
AutoApplyPage → POST /auto-apply/tasks → AutoApplyService.create_task → AutoApplyTaskRepository
  → INSERT auto_apply_task(status=queued) → 201 { id, status:"queued" }
  → Celery auto_apply.submit_application → resume_selection + adapter (HTTP or Playwright)
  → INSERT auto_apply_attempt, auto_apply_screenshot, auto_apply_log_entry
  → on success: INSERT tracking.application, tracking.application_status_history
  → UPDATE auto_apply_task.status = submitted | requires_review | failed
```

**Application Tracking**
```
TrackerBoardPage → GET /tracking/applications?filters → TrackingService.list → ApplicationRepository
  → SELECT application (+status_history/notes) → 200 PaginatedResponse[ApplicationRead]
TrackerBoardPage → POST /tracking/applications/{id}/status → TrackingService.change_status
  → UPDATE application.status, INSERT application_status_history
  → side effects: NotificationService.notify(APPLICATION_STATUS_CHANGED) + Redis publish → WS
```

**Notifications**
```
AppShell bell icon → GET /notifications?unread_only → NotificationService (read path) → NotificationRepository
  → SELECT notification → 200 PaginatedResponse[NotificationRead]
```

**AI Career Coach**
```
CoachToolPage → POST /career-coach/resumes/{id}/review → AICareerCoachService.review_resume
  → ResumeContextRepository (reads resume_parsed_data) → AIService.extract_structured (LLM)
  → INSERT career_insight (result cached as JSONB) → 200 ResumeReviewResult
```

**Analytics**
```
AnalyticsPage → GET /analytics/dashboard?filters → AnalyticsService.get_dashboard → AnalyticsRepository
  → read-only aggregate SELECTs across application/resume_version/ats_resume_version/job/company/skill
  → 200 AnalyticsDashboardResponse (no writes, no owned tables)
```

**Workspace (Dashboard)**
```
WorkspacePage → GET /workspace → WorkspaceService.get_summary
  → fan-out reads: user_profile, matching, resume_parser, tracking, ai_career_coach, notification repositories
  → 200 WorkspaceSummary (single synchronous aggregation, no AI call, no writes)
```

**Admin**
```
DashboardPage(admin) → GET /admin/dashboard → AdminService.get_dashboard → AdminRepository
  → aggregate counts across user/company/job/application/report/feature_flag + recent audit_log/system logs
  → 200 AdminDashboardResponse
```

---

## 5. Event Flow

**Important finding:** the formal `EventBus` (`app/events/bus.py`, in-process asyncio pub/sub) is fully implemented and wired into `main.py`'s startup (`register_all_handlers()`), but that function body is literally `pass`, and a repo-wide search finds **zero** `event_bus.publish(...)` or `event_bus.subscribe(...)` calls anywhere in `app/modules/**`. It is dead scaffolding for a "Phase 1" that hasn't happened. All actual "events" in this system are one of:

- **Celery task chaining** (a task's completion directly enqueues the next task) — deterministic, ordered, but tightly coupled (module A's task literally imports and calls `.delay()` on module B's task).
- **Redis pub/sub → WebSocket** — for live UI updates only (not used for backend-to-backend coordination).
- **NotificationService.notify(...)** — the closest thing to a real "domain event with subscribers," but it's a direct method call with a hardcoded enum of event types, not a pub/sub topic.

### Redis pub/sub events (real-time UI channel)

| Event (`type` field) | Publisher | Channel | Subscriber |
|---|---|---|---|
| `pipeline_progress` | every step of the resume pipeline (`resume_parser.tasks._advance_pipeline`, also called from `matching.tasks`) | `workspace:updates:{user_id}` | `GET /workspace/ws` → `PipelineBanner` (frontend) |
| `ats_resume_ready` | `ats_resume_generator.tasks.generate_ats_resume_files_task` | `workspace:updates:{user_id}` | `/workspace/ws` |
| `workspace_ready` | `workspace.tasks.update_workspace_task` | `workspace:updates:{user_id}` | `/workspace/ws` |
| `application_status_changed` | `tracking.service.TrackingService._notify_and_publish_status_change` (called from both the tracking router AND email_tracking) | `workspace:updates:{user_id}` | `/workspace/ws` |
| `match_computed` | `matching.service.MatchingService.compute_match` (single-pair sync compute) | `matching:updates:{user_id}` | `/matching/ws` and `/workspace/ws` (workspace WS subscribes to both channels) |
| `matches_computed` | `matching.tasks.compute_resume_matches_task` (batch) | `matching:updates:{user_id}` | same |
| `recommendations_updated` | `matching.tasks.refresh_recommendations_task` / `matching.service.refresh_recommendations` | `matching:updates:{user_id}` | same |

### Notification "events" (`NotificationEventType` enum — DB row + optional email, dispatched by direct method call, not pub/sub)

| Event Type | Publisher | Trigger | Execution order |
|---|---|---|---|
| `RESUME_READY` | `workspace.tasks.update_workspace_task` (last pipeline step) | resume finishes full pipeline | parse→embed→score→ats_baseline→match→recommend→dashboard→**notify** |
| `NEW_MATCHING_JOB` | `matching.tasks.match_new_job_for_active_users_task` | new job ingested scores ≥ `MATCHING_NEW_JOB_NOTIFY_THRESHOLD` (75.0) for an active user | ingest→normalize→skills→embed→**match+notify** |
| `APPLICATION_STATUS_CHANGED` | `tracking.service.TrackingService.change_status` | manual status edit OR Gmail-classified email match | status change→**notify**+**publish** |
| `EMAIL_REVIEW_NEEDED` | `email_tracking.service.EmailTrackingService.process_email` | classified email is low-confidence or ambiguous match | classify→(no confident match)→**notify** |
| `GMAIL_REAUTH_REQUIRED` | `email_tracking.tasks._flag_needs_reauth` | Gmail OAuth token expired/revoked (AUTH failure category) | poll fails→**notify**, account flagged `NEEDS_REAUTH` |

---

## 6. Background Workers

All tasks defined with `base=BaseTask` (`app/workers/base_task.py`: `retry_backoff=True`, `retry_backoff_max=600`, `retry_jitter=True`, default `max_retries=3`, logs on success/failure). Broker/backend: Redis db1/db2.

| Task name | Queue | Trigger | Input | Output | Retry |
|---|---|---|---|---|---|
| `resume_parser.parse_resume` | `parsing` | `ResumeParserService.upload_resume/upload_new_version` | resume_version_id, user_id | `ResumeParsedData` row, status→PARSED; chains `generate_embedding` | none configured |
| `resume_parser.generate_embedding` | `embedding` | previous task success | resume_version_id | `ResumeEmbedding` row (pgvector); chains `calculate_score` | none |
| `resume_parser.calculate_score` | `embedding` | previous task success | resume_version_id | `resume_score`/`resume_score_breakdown`; chains `run_ats_baseline_analysis` | `autoretry_for=(Exception,)`, max 3 |
| `resume_parser.run_ats_baseline_analysis` | `embedding` | previous task success | resume_version_id | `ats_baseline_score`, `ats_baseline_missing_skills`; chains `matching.compute_resume_matches` | `autoretry_for=(Exception,)`, max 3 |
| `job_ingestion.sweep_companies` | `ingestion` | **Celery Beat**, every `JOB_INGESTION_SWEEP_INTERVAL_MINUTES` (360m) | none | fans out `fetch_company_jobs` per due company | n/a |
| `job_ingestion.fetch_company_jobs` | `ingestion` | sweep, or manual `POST /companies/{id}/crawl` | company_id | upserts `job` rows, `IngestionRun`; chains `normalize_job` per job | `autoretry_for=(httpx.HTTPError, ServiceUnavailableError)` |
| `job_ingestion.normalize_job` | `ingestion` | previous task | job_id | parsed salary/location/work_mode, status→ACTIVE; chains `extract_job_skills` | none |
| `job_ingestion.extract_job_skills` | `skills` | previous task | job_id | `Skill` rows + `job_skill` links (LLM-extracted); chains `generate_job_embedding` | none |
| `job_ingestion.generate_job_embedding` | `embedding` | previous task | job_id | `JobEmbedding` row; chains `matching.match_new_job_for_active_users` | none |
| `matching.compute_resume_matches` | `matching` | resume pipeline chain, or manual `POST /resumes/{id}/compute` | resume_version_id | `ResumeJobMatch` rows; chains `refresh_recommendations` | none |
| `matching.refresh_recommendations` | `matching` | previous task, or manual `POST /recommendations/refresh` | resume_version_id (opt) | `Recommendation` rows; chains `workspace.update_dashboard` (if resume_version_id) | none |
| `matching.sweep_stale_matches` | `matching` | **Celery Beat**, fixed 3600s | none | marks stale `ResumeJobMatch.status=STALE` | none |
| `matching.periodic_batch_sweep` | `matching` | **Celery Beat**, every `MATCHING_RECOMMENDATION_REFRESH_INTERVAL_MINUTES` (1440m) | none | fans out `compute_resume_matches` per active user | none |
| `matching.match_new_job_for_active_users` | `matching` | `job_ingestion.generate_job_embedding` ("smart trigger") | job_id | `ResumeJobMatch` rows + `NEW_MATCHING_JOB` notification if score≥75 | `autoretry_for=(Exception,)`, max 3 |
| `ats_resume_generator.optimize` | `ats_optimization` | `ATSResumeGeneratorService.create_ats_resume/create_new_version` | ats_resume_version_id | LLM-optimized summary/skills/projects, keyword+ats score; chains `generate_files` | none |
| `ats_resume_generator.generate_files` | `ats_files` | previous task | ats_resume_version_id | DOCX+PDF in S3/local, `ATSResumeFile` rows; publishes `ats_resume_ready` | none |
| `auto_apply.submit_application` | `auto_apply` | `AutoApplyService.create_task` (or retry) | task_id | form submission, `AutoApplyAttempt`, screenshots, `tracking.Application` on success | `autoretry_for=(TransientApplyError, PlatformApplyError)`, max `AUTO_APPLY_MAX_ATTEMPTS` |
| `email_tracking.sweep_accounts` | `email_tracking` | **Celery Beat**, every `GMAIL_SYNC_INTERVAL_MINUTES` (10m) | none | fans out `poll_account` per connected Gmail account | none |
| `email_tracking.poll_account` | `email_tracking` | sweep task | gmail_account_id | new message ids; chains `fetch_and_classify_email` per message; flips `NEEDS_REAUTH` on auth failure | `autoretry_for=(TransientGmailError,)` |
| `email_tracking.fetch_and_classify_email` | `email_tracking` | poll task | account_id, message_id | classified `TrackedEmail`, possible `Application` status change | `autoretry_for=(TransientGmailError, ServiceUnavailableError)` |
| `notification.send_email` | `notifications` | `NotificationService.notify` (EMAIL channel) | notification_id | email sent via SMTP/SES, status SENT/FAILED | `autoretry_for=(TransientNotificationError,)`, max `NOTIFICATION_MAX_RETRIES` |
| `notification.send_resume_ready` | `notifications` | `workspace.update_dashboard` (pipeline's final link) | user_id, resume_version_id | `RESUME_READY` notification (idempotency-guarded) | `autoretry_for=(Exception,)`, max 3 |
| `workspace.update_dashboard` | `matching` (shares the matching worker/queue) | `matching.refresh_recommendations` | resume_version_id, user_id | publishes `workspace_ready`; chains `notification.send_resume_ready` | `autoretry_for=(Exception,)`, max 3 |

**Celery Beat schedule (4 entries):** `job-ingestion-sweep` (360m default), `matching-periodic-batch-sweep` (1440m default), `matching-stale-sweep` (fixed 3600s), `gmail-sync-sweep` (10m default).

---

## 7. Database Flow

**40 tables** across 12 functional groups (search and analytics own no tables — pure read layers). Every table has a UUID PK + `created_at`/`updated_at` (`TimestampMixin`); tables prefixed with a ✝ also carry a nullable `deleted_at` (`SoftDeleteMixin`). Almost no `user_id` column anywhere in the schema is a real foreign key to `users.id` — auth was modularized last (migration 0012 of 14) and older tables were never retrofitted with the constraint, so user-ownership is enforced at the **service layer** (comparing JWT-derived `user_id` against the row), not the database.

| Table | Created by | Updated by | Deleted by | Key relationships |
|---|---|---|---|---|
| `users` ✝(no soft-delete actually) | `AuthService.register` | `AuthService` (is_active toggles — none currently exposed) | n/a | ← `refresh_token` |
| `refresh_token` | `AuthService.issue_tokens` | `AuthService.refresh` (rotate/revoke) | `AuthService.logout*` (revoked_at set, not hard-deleted) | → users |
| `user_profile` | `UserProfileService.complete_onboarding` | same | never | — |
| `resume` ✝ | `ResumeParserService.upload_resume` | `update_resume` | `delete_resume` (soft) | ← resume_version |
| `resume_version` | `upload_resume`/`upload_new_version` | Celery pipeline tasks (status/scores) | `delete_version` (hard, blocked if last) | → resume; ← parsed_data, embedding, pipeline_run |
| `resume_parsed_data` | `parse_resume_task` | never (immutable per version) | cascade w/ version | → resume_version (1:1) |
| `resume_embedding` | `generate_resume_embedding_task` | never | cascade | → resume_version (1:1) |
| `resume_pipeline_run` | `ResumePipelineRunRepository.start` | every task step (`advance`) | never | → resume_version (1:1) |
| `company` ✝ | `CompanyService.register_company` | `update_company`, ingestion tasks | `delete_company` (soft) | ← intelligence, job, ingestion_run |
| `company_intelligence` | auto-created on company register / crawl config | crawl tasks (last_crawled_at etc.) | cascade | → company (1:1) |
| `location` | `normalize_job_task` get-or-create | never | never | ← job |
| `skill` | `extract_job_skills_task` / resume ATS baseline get-or-create | never | never | ↔ job (via job_skill) |
| `job` ✝ | `fetch_company_jobs_task` | `normalize_job_task`, `bulk_close_stale`; admin moderation (soft-delete via status=REMOVED) | admin job moderation | → company, location; ← job_embedding, job_skill; referenced by matching/tracking/ats/auto_apply |
| `job_skill` (join) | `extract_job_skills_task` | never | cascade w/ job or skill | job ↔ skill |
| `job_embedding` | `generate_job_embedding_task` | never | cascade | → job (1:1) |
| `ingestion_run` | `fetch_company_jobs_task` | same task (counters, status) | never | → company |
| `match_preference` | `MatchingService.upsert_preferences` (via onboarding) | same | never | — |
| `resume_job_match` | `batch_compute_matches` / single-pair compute | `sweep_stale_matches` (status→STALE) | never | → resume_version, job |
| `recommendation` | `refresh_recommendations` | never (delete-then-reinsert) | replaced wholesale each refresh | → job/company/skill (one of) |
| `application` ✝ | `TrackingService.create` or `auto_apply` on submit success | `change_status`, `update` | `delete` (soft), admin delete | → job (SET NULL); ← status_history, notes; ← tracked_email, auto_apply_task |
| `application_status_history` | `TrackingService.change_status` | never (append-only) | never | → application |
| `application_note` | notes endpoints | `update_note` | `delete_note` | → application |
| `ats_resume` ✝ | `ATSResumeGeneratorService.create_ats_resume` | `update_ats_resume` | `delete_ats_resume` (soft) | → resume (source), job (target, SET NULL) |
| `ats_resume_version` ✝ | `create_ats_resume`/`create_new_version` | `optimize_ats_resume_task` (scores/content), status transitions | `delete_version` | → ats_resume, resume_version (source) |
| `ats_resume_file` | `generate_ats_resume_files_task` | insert-or-replace by format | cascade | → ats_resume_version |
| `applicant_profile` | `AutoApplyService` on first `PUT /profile` | `PUT /profile` | never | → ats_resume, resume (defaults) |
| `auto_apply_task` ✝ | `create_task` | Celery `submit_application` (status machine) | soft, admin delete | → job, ats_resume, resume_version, application (tracking_application_id) |
| `auto_apply_attempt` | `submit_application_task` | same (finish status) | never | → task |
| `auto_apply_screenshot` | `submit_application_task` | never | never | → attempt |
| `auto_apply_log_entry` | `submit_application_task` | never (append-only) | never | → attempt |
| `gmail_account` | `EmailTrackingService.complete_connect` | poll tasks (tokens, sync state), `disconnect` | never (status=DISCONNECTED instead) | ← tracked_email |
| `tracked_email` | `fetch_and_classify_email_task` | `resolve_tracked_email` | never | → gmail_account, application (SET NULL), application_status_history |
| `notification` | `NotificationService.notify` | `mark_read`/`mark_all_read`, task status updates | never | — |
| `notification_preference` | `update_preferences` | same | never | — |
| `career_insight` | every AI Career Coach analysis endpoint | never (new row per call — history log) | never | → resume_version (SET NULL) |
| `coach_conversation` ✝ | `POST /chat` (new conversation) | `last_message_at` bump | `DELETE /conversations/{id}` (soft) | ← coach_message |
| `coach_message` | `send_message` | never (append-only) | cascade | → coach_conversation |
| `admin_actor` | `bootstrap_actor` / `create_actor` | `update_actor` | never (is_active toggle instead) | ← platform_user_status, verification, moderation, reports, audit_log, feature_flag (as actor FK) |
| `platform_user_status` | `update_user_status` (upsert) | same | never | → admin_actor (changed_by) |
| `company_verification` | admin views (get-or-create) | `update_company_verification` | never | → company (1:1), admin_actor |
| `job_moderation` | admin views (get-or-create) | `update_job_moderation` (also soft-deletes/restores `job`) | never | → job (1:1), admin_actor |
| `content_report` | `POST /admin/reports` (any user, not just admin) | `update_report` | never | → admin_actor (resolved_by) |
| `audit_log` | every mutating `AdminService` call | never (append-only) | never | → admin_actor |
| `feature_flag` | `create_feature_flag` | `update_feature_flag` | `delete_feature_flag` (hard) | → admin_actor (updated_by) |

**Hub tables** (most cross-module inbound FKs): `job` (7 dependents: job_skill, job_embedding, resume_job_match, recommendation, application, ats_resume, auto_apply_task, job_moderation), `resume_version` (7 dependents), `application` (4 dependents), `admin_actor` (6 dependents).

---

## 8. AI Flow

Every LLM call in the system funnels through **one class**, `app/services/ai/service.py::AIService`, which wraps whichever provider `AI_PROVIDER` selects (Ollama by default) and adds `tenacity`-based retry on `TransientProviderError`. It exposes exactly two methods, `extract_structured(text, schema, system, max_tokens)` (structured-output / JSON mode) and `complete_chat(system, messages, max_tokens)` (free-form). Every structured caller reuses one of its own Pydantic response models' `.model_json_schema()` as the literal output contract handed to the LLM — no separate prompt-schema definitions exist.

| Caller | Input | Prompt (system) | Output schema | Stored where | Depends on |
|---|---|---|---|---|---|
| `resume_parser.tasks.parse_resume_task` | raw resume text (pypdf/python-docx extracted) | `EXTRACTION_SYSTEM_PROMPT` | `ParsedResumeData` (summary, skills, experience, education, projects, certifications, achievements, languages) | `resume_parsed_data` | text extraction must succeed first |
| `job_ingestion.tasks.extract_job_skills_task` | job title + description | skill-extraction system prompt | list of skill strings | `skill` + `job_skill` rows | job must be normalized first |
| `ats_resume_generator.tasks.optimize_ats_resume_task` | source resume raw_text + job_description, via `_build_optimize_prompt` | `OPTIMIZE_SYSTEM_PROMPT` (hard rule: never invent employers/dates/outcomes, every skill must be evidenced, no new/removed projects) | `OptimizedContent` (summary, skills, projects, keywords) | `ats_resume_version.optimized_*`, `keywords` | source resume must have `parsed_data` |
| `email_tracking.classifier` (`classify_email`) | email subject/snippet/sender | `EMAIL_CLASSIFICATION_SYSTEM_PROMPT` | `EmailClassificationResult` (classification, is_relevant, confidence, extracted_company_name, extracted_job_title, reasoning) | `tracked_email.classification/confidence/raw_metadata` | Gmail message must be fetched first |
| `ai_career_coach` — `review_resume`, `career_suggestions`, `skill_gap_analysis`, `learning_recommendations`, `career_roadmap`, `interview_preparation`, `salary_prediction`, `market_trend_analysis`, `country_demand` | resume context text (`_resume_context_text`) + target role + optional job-market stats from `job_ingestion` | per-feature system prompt | per-feature Pydantic result model (e.g. `ResumeReviewResult`, `SkillGapResult`) via `.model_json_schema()` | `career_insight.result` (JSONB, one row per call — a permanent cache/history) | resume must be parsed; some features (salary, market trends) blend deterministic Python stats with LLM narrative |
| `ai_career_coach.send_message` (chat) | user message + up to `AI_CAREER_COACH_CHAT_MAX_HISTORY_MESSAGES` prior turns + resume context | `_CHAT_SYSTEM_PROMPT` | free-form text (`complete_chat`, not structured) | `coach_message` (both user + assistant turns) | none — always available once a conversation exists |

**Embeddings** go through the parallel `EmbeddingService` (`app/services/embedding/service.py`), which adds Redis cache-aside (key = sha256 of provider+input_type+text, TTL `EMBEDDING_CACHE_TTL_SECONDS`, default 7 days) on top of the same retry pattern:

| Caller | Input | Output | Stored where |
|---|---|---|---|
| `resume_parser.tasks.generate_resume_embedding_task` | concatenated summary/skills/experience/education | 384-dim vector (local sentence-transformers by default) | `resume_embedding.embedding` |
| `job_ingestion.tasks.generate_job_embedding_task` | job title + description | 384-dim vector | `job_embedding.embedding` |
| `search.service` (query time) | user's search query string | 384-dim vector (`input_type="query"`) | not stored — used transiently for the cosine-distance query |

---

## 9. Search Flow

Owned entirely by the `search` module, which owns **no tables** — it queries `job_ingestion`'s tables directly.

- **Keyword search**: Postgres full-text search. `job.search_vector` is a generated `tsvector` column (`setweight(title,'A') || setweight(description,'B')`), queried via `plainto_tsquery('english', q)` and ranked with `ts_rank_cd`. A secondary trigram signal uses `pg_trgm`'s `%` similarity operator against `job.title` for typo tolerance.
- **Semantic search**: pgvector. Query text → `EmbeddingService.generate_embedding(text, input_type="query")` → `JobEmbedding.embedding.cosine_distance(query_vector)`, `similarity = 1 - distance`, ordered ascending by distance. An HNSW index backs this (added in migration 0003).
- **Vector search** is the same mechanism as semantic search above — this codebase doesn't distinguish a separate "vector search" concept from "semantic search."
- **Hybrid search** (the default `mode`): candidates are pulled independently from both paths (capped at `SEARCH_KEYWORD_CANDIDATE_LIMIT` / `SEARCH_SEMANTIC_CANDIDATE_LIMIT`), each score min-max normalized to [0,1], then fused as a **weighted sum** (not reciprocal-rank-fusion): `hybrid_score = SEARCH_HYBRID_KEYWORD_WEIGHT * keyword_norm + SEARCH_HYBRID_SEMANTIC_WEIGHT * semantic_norm` (defaults 0.4/0.6).
- **Ranking**: `sort_by` supports `relevance` (the fused/keyword/semantic score depending on mode), `salary`, `match_score` (requires a non-empty query and `mode != keyword`), `newest`, `company`.
- **Filtering**: status=ACTIVE + not soft-deleted always enforced; then optional `work_mode`, `employment_type`, `company_id`/`company_name` (ILIKE), location (join), salary range-overlap, skills (EXISTS subquery, match-all or match-any), `experience_level` (heuristically inferred from the job title via regex patterns — there is no dedicated experience-level column), `date_posted` cutoff.
- **Caching**: every read path (`search`, `autocomplete`, `suggestions`, `trending/skills`, `trending/jobs`) is cached via `app/integrations/cache.py::CacheClient` (Redis-backed, deterministic keys from a sha256 hash of sorted filter params) with independently configurable TTLs.
- **No writes, no events, no Celery tasks** in this module — it's a pure, cached read path over data other modules produce.

---

## 10. Matching Flow

```
Resume Embedding (pgvector, from resume_parser)
     ↓
Job Embedding (pgvector, from job_ingestion)         [candidate set: semantic_candidate_jobs,
     ↓                                                 top-N by cosine distance, N=MATCHING_CANDIDATE_LIMIT]
Cosine Similarity  →  semantic_score ∈ [0,1]
     ↓
Skill Overlap  →  skill_score = matched_job_skills / total_job_skills * 100
     ↓
Weighted Base  =  semantic_score*100*MATCHING_SEMANTIC_WEIGHT (0.6) + skill_score*MATCHING_SKILL_WEIGHT (0.4)
     ↓
Rule Engine (each boost is None if the user set no preference, else 0..cap):
     country_boost · work_mode_boost · employment_type_boost
     · salary_boost (range overlap + currency match) · experience_boost (title-derived level vs years-of-exp)
     ↓
Final Score  =  clamp(base + Σ boosts, 0, 100)
     ↓
Strengths / Weaknesses  (templated narrative from score thresholds — no LLM involved)
     ↓
Recommendations  (job/company/skill types, floor score 50.0, replace-wholesale on each refresh,
                   excludes previously-dismissed candidates)
```

This is **not** a machine-learned ranking model — it's embedding-similarity + deterministic skill overlap + a hand-tuned rule engine, all weights/caps configurable via environment variables. Real-time delivery to the browser rides Redis pub/sub (`matching:updates:{user_id}`) → `/matching/ws`, independent of the scoring computation itself.

---

## 11. Application Flow

```
Job Selected (frontend: SearchPage/JobDetailPage "Apply" action)
     ↓
POST /auto-apply/tasks  →  AutoApplyTask(status=queued) created; validates job exists/not closed,
     no duplicate active task for the same job
     ↓  Celery auto_apply.submit_application
Resume Selection (resume_selection.select_resume) — priority order:
     1. explicit override  2. tailored ATS resume matching target_job_id
     3. tailored ATS resume matching company name  4. most recent completed tailored resume
     5. most recent parsed plain resume  →  raises if none usable
     ↓
Adapter Dispatch (get_apply_adapter, by Company.intelligence.ats_platform):
     Greenhouse/Lever → HTTP form POST (httpx + BeautifulSoup)
     Workday → Playwright browser automation (multi-step Angular SPA)
     Custom/Unknown → Playwright generic fallback (CSS-selector-driven)
     ↓
Field Discovery  →  ATS Resume Field Validation (canonical-hint matching + custom_answers bank)
     ↓
   ┌─── Gate: mapping incomplete OR validation issues ───┐
   │        → status = REQUIRES_REVIEW (STOPS HERE)       │   ← MANUAL CHECKPOINT
   │        user edits ApplicantProfile/custom_answers,    │
   │        clicks "Retry" → POST /tasks/{id}/retry        │
   └────────────────────────────────────────────────────┘
     ↓ (mapping/validation OK)
Auto Apply (adapter.apply — actually fills and submits the form/browser flow, unattended)
     ↓
Confirmation Detection (heuristic: success phrases in response text/HTML, or URL patterns — no LLM)
     ↓
Tracking  →  creates tracking.Application(status=APPLIED, source=AUTO_APPLY) +
     ApplicationStatusHistory; links back via auto_apply_task.tracking_application_id
     ↓
Notifications  →  APPLICATION_STATUS_CHANGED (via TrackingService, the same path a manual status
     edit or Gmail-classified email uses) + Redis publish → workspace WebSocket
```

Note there is **no pre-submission human-approval gate** — once field mapping and validation pass, the form is submitted unattended. The only human-in-the-loop checkpoint is the `REQUIRES_REVIEW` state for incomplete/invalid data.

---

## 12. Dashboard Data Flow

The live dashboard is `/workspace` (`WorkspacePage`), fed by a single `GET /workspace` call plus a WebSocket for push updates. `WorkspaceService.get_summary()` deliberately does no AI calls and no writes — it's a fan-out read composed entirely from other modules' existing services/repositories:

| Widget | Data source | Refresh mechanism |
|---|---|---|
| Resume Score / Health | `resume_parser.ResumeVersionRepository` (`resume_score`, `ats_baseline_score`) | Redraws on `pipeline_progress`/`workspace_ready` WS event, otherwise on page load |
| ATS Score | `resume_parser.ResumeVersionRepository.ats_baseline_score` (baseline) or `ats_resume_generator` (per tailored resume, via its own page — not on the workspace widget) | same |
| Applications (stats/recent/follow-up) | `tracking.TrackingService.get_dashboard()` (own aggregate queries) | Page load only; WS event `application_status_changed` triggers a query invalidation in the frontend, causing a refetch |
| Recommendations | `matching.MatchingRepository.list_recommendations(type=JOB)` | Redraws on `recommendations_updated`/`matches_computed` WS events |
| Notifications | `notification.NotificationRepository` (unread, limit 5 + count) | Page load; no push event dedicated to new notifications specifically (piggybacks on the general workspace refresh) |
| Analytics | **Not on the workspace dashboard** — lives on its own separate `/analytics` page, fetched independently via `GET /analytics/dashboard` | Page load only, no WS |
| Pinned Items | `pinsStore` (Zustand, **client-side only** — no backend endpoint; "save for later" state lives only in `localStorage`, never synced) | n/a — purely local |
| Saved Jobs | Same as Pinned Items — `pinsStore`, client-only, no backend "saved job" table or endpoint | n/a |
| Career Coach highlights | `ai_career_coach.CareerInsightRepository.list_by_user(limit=3)` (most recent cached insights — does not generate new ones) | Page load only |
| Next Best Actions | `WorkspaceService._build_next_actions()` — a local, deterministic Python priority rules engine (upload_resume=100, generate_ats_resume=90 if baseline<70, close_skill_gaps=70, follow_up_applications=60, review_recommendations=50, check_coach=10 fallback), top 4 shown | Recomputed on every `GET /workspace` call |

The legacy `features/dashboard/DashboardPage.tsx` (older widgets: `PinnedItemsPanel`, `QuickActions`, `RecentActivityList`, `ScoreTile`, `SuggestionsFeed`, `WeeklyGoalCard`) still exists in the codebase but **is not mounted by any route** — `/dashboard` redirects straight to `/workspace`. It appears to be dead code left over from before the Workspace aggregation was built.

---

## 13. Route Map

### Frontend routes (`frontend/src/routes/AppRoutes.tsx` + `paths.ts`)

| Path | Shell | Guard | Page |
|---|---|---|---|
| `/`, `/features`, `/pricing`, `/about`, `/contact` | MarketingShell | public | Landing/Features/Pricing/About/Contact |
| `/login`, `/register`, `/forgot-password`, `/reset-password`, `/verify-email` | AuthShell | public | Login/Register/ForgotPassword/ResetPassword/VerifyEmail (**last 3 have no backing backend endpoint** — see §17) |
| `/onboarding` | none (bare) | protected | OnboardingWizardPage |
| `/dashboard` | AppShell | protected | redirect → `/workspace` |
| `/workspace` | AppShell | protected | WorkspacePage |
| `/resumes`, `/resumes/:id`, `/resumes/:id/versions/:vid` | AppShell | protected | ResumeManager/Detail/VersionViewer |
| `/resumes/ats-generator` | AppShell | protected | ATSResumeGeneratorPage |
| `/jobs`, `/jobs/:id`, `/jobs/saved`, `/jobs/recommended` | AppShell | protected | Search/JobDetail/SavedJobs/RecommendedJobs |
| `/companies`, `/companies/:id` | AppShell | protected | CompanyExplorer/CompanyDetail |
| `/matching` | AppShell | protected | MatchingPage |
| `/auto-apply`, `/auto-apply/:taskId` | AppShell | protected | AutoApplyPage/TaskDetail |
| `/tracker`, `/tracker/calendar`, `/tracker/emails`, `/tracker/:applicationId` | AppShell | protected | TrackerBoard/Calendar/Emails/Timeline |
| `/notifications` | AppShell | protected | NotificationPage |
| `/coach`, `/coach/chat`, `/coach/insights`, `/coach/tools/:toolId` | AppShell | protected | Coach Home/Chat/Insights/Tool |
| `/analytics` | AppShell | protected | AnalyticsPage |
| `/settings`, `/settings/profile`, `/settings/security`, `/settings/api-keys` | AppShell | protected | Settings pages |
| `/admin/*` (12 sub-pages) | AppShell+AdminLayout | protected + permission-gated | Admin console |
| `*` | none | public | NotFoundPage |

Orphaned: `src/pages/ServerErrorPage.tsx` (no route references it). Declared-but-unrouted path builders in `paths.ts`: `atsResumeDetail()`, `resumeCompare()`, `coachSkillGap`, `coachRoadmap`, `coachSalary`, `coachLearning`.

### Backend endpoints (grouped by module; full detail in §4 sections above and in the module facts)

`/api/v1/auth` (register, login, refresh, logout, me) · `/api/v1/onboarding` (status, complete) · `/api/v1/users` (status stub) · `/api/v1/resumes` (10 endpoints — CRUD + versions + download) · `/api/v1/job-ingestion` (companies, crawl trigger, jobs) · `/api/v1/search` (search, autocomplete, suggestions, trending) · `/api/v1/matching` (preferences, compute, matches, recommendations, WS) · `/api/v1/ats-resumes` (CRUD + versions + diff + download) · `/api/v1/auto-apply` (profile, tasks, retry/cancel, attempts, screenshots) · `/api/v1/tracking` (applications CRUD, status, notes, statistics, dashboard) · `/api/v1/email-tracking` (gmail connect/callback, accounts, emails, resolve) · `/api/v1/notifications` (list, read, preferences) · `/api/v1/career-coach` (9 analysis endpoints, insights, chat) · `/api/v1/analytics` (dashboard, CSV/PDF export) · `/api/v1/workspace` (summary, progress, WS) · `/api/v1/admin` (23 endpoints — actors, users, companies, jobs, applications, reports, feature flags, audit/system logs, analytics) · `/api/v1/health`, `/api/v1/health/ready` · `/api/v1/storage/local/{key}` (signed local-storage download proxy).

---

## 14. State Management

- **Server state**: `@tanstack/react-query` v5 everywhere — `queryClient` defaults `retry: 1`, `refetchOnWindowFocus: false`, `staleTime: 30s`. No global query key registry; each feature manages its own keys/invalidation.
- **Client/UI state**: Zustand v5, 5 stores — `themeStore` (light/dark/system, applies `data-theme` attribute directly, no Context), `uiStore` (sidebar collapsed, command palette/shortcuts open), `toastStore` (transient, not persisted), `pinsStore` (client-only "save for later" — **no backend equivalent**), `goalsStore` (client-only weekly application goal — **no backend equivalent**).
- **React Context**: `AuthContext` (`features/auth/AuthProvider.tsx` — user/isAuthenticated/isLoading + login/register/logout), `AdminAuthContext` (role/permissions for `/admin/*`), plus Radix's `TooltipProvider` (UI-library context, not app data).
- **Cache**: Redis backs the backend's `CacheClient` (search results, autocomplete, trending, embeddings, Gmail OAuth state) — the frontend has no separate cache layer beyond React Query's in-memory cache.
- **Persistence**: access token lives **only in memory** (`tokenStore.ts`, a module-level variable + pub/sub, never localStorage); the refresh token is an httpOnly cookie the frontend never touches directly (`withCredentials: true`, scoped server-side to `/auth`). Zustand's `persist` middleware writes `themeStore`, `uiStore` (sidebar only), `pinsStore`, and `goalsStore` to `localStorage`.

---

## 15. Complete Project Tree

```
job-seaker/
├── backend/
│   ├── alembic/versions/            14 linear migrations (0001–0014)
│   ├── app/
│   │   ├── main.py                  FastAPI app factory + lifespan
│   │   ├── core/                    config, constants, security (JWT/bcrypt), crypto (Fernet), exceptions, logging
│   │   ├── db/                      base_class (Base + mixins), session (async engine), base (Alembic metadata glue)
│   │   ├── dependency_injection/     container.py (Celery-scope service locator), providers.py (single wiring point)
│   │   ├── events/                   base.py (Event contract), bus.py (EventBus — UNUSED), handlers/ (empty)
│   │   ├── factories/                ai/email/embedding/storage provider factories (env-var dispatch)
│   │   ├── integrations/             cache.py (Redis cache-aside), redis_client.py, vector_store.py (pgvector const), http_client.py
│   │   ├── interfaces/               AIProvider/EmailProvider/EmbeddingProvider/StorageProvider Protocols
│   │   ├── middleware/               exception handlers, logging, request-context (request_id)
│   │   ├── models/mixins.py          re-export shim → db/base_class
│   │   ├── modules/                  14 modules (see below)
│   │   ├── providers/                concrete provider implementations (ai/email/embedding/storage)
│   │   ├── repositories/base.py      generic async CRUD base
│   │   ├── schemas/common.py         BaseResponse/PaginatedResponse/ErrorResponse envelopes
│   │   ├── services/                 cross-cutting: ai/, embedding/, notification/ (push_sender, templates, errors), recommendation/ (STUB)
│   │   ├── utils/datetime_utils.py
│   │   └── workers/                  celery_app.py (queues, beat schedule), base_task.py, tasks/ (empty placeholder)
│   ├── docker/postgres/init/         pgvector, uuid-ossp, pg_trgm extension bootstrap
│   ├── tests/                        pytest (asyncio_mode=auto), factories/, modules/, providers/, services/
│   ├── requirements.txt, pyproject.toml (ruff/black/mypy config), Dockerfile
│   └── modules/ (14, each with models.py/repository.py/router.py/schemas.py/service.py + module-specific extras):
│       auth · user_profile · resume_parser (+extraction.py, tasks.py)
│       · ats_resume_generator (+scoring.py, diffing.py, documents.py, tasks.py)
│       · job_ingestion (+adapters/, normalization.py, tasks.py)
│       · search (+scoring.py) · matching (+scoring.py, realtime.py, tasks.py)
│       · auto_apply (+adapters/, confirmation.py, errors.py, field_mapping.py, resume_selection.py, validation.py, tasks.py)
│       · tracking · email_tracking (+classifier.py, gmail_*_client.py, matching.py, errors.py, tasks.py)
│       · notification (+tasks.py) · analytics (+reports.py) · ai_career_coach
│       · workspace (+realtime.py, tasks.py) · admin (+permissions.py, system_logs.py, deps.py)
├── frontend/
│   ├── src/
│   │   ├── api/client.ts             axios instance + auth interceptors
│   │   ├── app/                      config.ts (apiBaseUrl), queryClient.ts
│   │   ├── components/               command/, common/, layout/ (AppShell, AuthShell, MarketingShell), motion/, ui/ (design system)
│   │   ├── features/                 19 feature folders mirroring backend modules (+ marketing, settings, dashboard[orphaned])
│   │   ├── layouts/                  EMPTY — real shells live in components/layout/
│   │   ├── lib/                      cn.ts, errorMessage.ts, format.ts, useDebouncedValue.ts
│   │   ├── pages/                    NotFoundPage (wired), ServerErrorPage (orphaned)
│   │   ├── routes/                   AppRoutes.tsx, paths.ts
│   │   ├── stores/                   goalsStore, pinsStore, themeStore, toastStore, uiStore (all Zustand)
│   │   ├── styles/                   base.css, index.css, legacy.css, tokens.css
│   │   ├── main.tsx, App.tsx
│   ├── package.json, vite.config.ts, Dockerfile
├── nginx/nginx.conf                  single reverse-proxy config
├── docker-compose.yml                 postgres, redis, ollama(+pull), api, 7 workers, beat, frontend, nginx
└── .env.example / backend/.env.example / frontend/.env.example
```

---

## 16. Automation Flow (Manual Actions)

| Manual action today | Current behaviour | Ideal behaviour | Missing trigger |
|---|---|---|---|
| Generate ATS Resume | User must open `/resumes/ats-generator` and explicitly submit a job description; nothing links a high-scoring match to this action | Auto-suggest (or auto-generate) a tailored ATS resume the moment a `ResumeJobMatch` clears a high-score threshold for a job the user is interested in | No hook from `matching.tasks.compute_resume_matches`/`match_new_job_for_active_users` into `ats_resume_generator` |
| Create Auto-Apply task | User must click "Apply" per job, even after an ATS resume is marked `ats_resume_ready` | Offer one-click apply directly from the "ready" notification/recommendation | No hook from `ats_resume_generator`'s `ats_resume_ready` event into `auto_apply` |
| Retry a `REQUIRES_REVIEW` auto-apply task | User must notice the status, edit `ApplicantProfile`/`custom_answers`, click Retry | Same (this one is *correctly* manual — it's a legitimate human-in-the-loop gate for missing/invalid PII) | N/A — working as intended, not a gap |
| Visit AI Career Coach | User must proactively open `/coach` and click each tool (review, skill-gap, roadmap, etc.) | Auto-run at least "resume review" + "job readiness" right after a resume finishes its pipeline, surfaced on the workspace dashboard | No hook from the resume pipeline (or `workspace.update_dashboard`) into `ai_career_coach` |
| Refresh recommendations | Mostly automatic (pipeline + periodic sweep), but user can also force `POST /recommendations/refresh` | Fine as-is — automatic path already covers the common case | N/A |
| Crawl a company's jobs | Admin/operator must call `POST /companies/{id}/crawl` for a brand-new company (Beat sweep only covers companies already marked "due") | Auto-crawl once on company registration | `register_company` doesn't enqueue an initial `fetch_company_jobs` |
| Save/pin a job or resume | Stored only in browser `localStorage` (`pinsStore`) — lost on a new device/browser | Persist server-side so pins sync across devices | No backend table/endpoint for pinned/saved items at all |
| Weekly application goal | Stored only in browser `localStorage` (`goalsStore`) | Persist server-side, and let `analytics`/`workspace` reference it for a real progress widget | No backend table/endpoint for goals |
| Verify email / reset password | Frontend renders `VerifyEmailPage`/`ForgotPasswordPage`/`ResetPasswordPage`, but there is **no backend endpoint** for any of the three — these pages are effectively non-functional | Real endpoints backing all three flows | `auth` module has no verify/forgot/reset routes at all |
| Notice a failed notification email | Only surfaced via `admin/system-logs` (an admin-only aggregation view) | Surface to the affected user too, or auto-retry more aggressively | No user-facing surfacing of `notification.status=FAILED` |
| Dismiss vs act on `EMAIL_REVIEW_NEEDED` | User must open `/tracker/emails` and manually resolve each ambiguous email | Fine as designed — ambiguity inherently needs a human | N/A |

---

## 17. Missing Connections

1. **The formal `EventBus` (`app/events/bus.py`) is entirely disconnected.** It's fully implemented (asyncio pub/sub, `Event` base class, `subscribe`/`publish`) and instantiated at startup (`register_all_handlers()` runs on every boot), but that function is a literal `pass`, and no module anywhere calls `event_bus.publish(...)` or `event_bus.subscribe(...)`. The codebase's actual "event-driven" behavior is really **hardcoded Celery task chaining** plus two ad-hoc Redis pub/sub channels — functional, but not the general-purpose, decoupled event system the scaffolding implies. Any new cross-module reaction today requires editing the upstream task's source to add another `.delay()` call, not registering an independent handler.
2. **Resume Upload → Resume Parsing → Embeddings → Matching → Recommendations → Dashboard → Notification is, in fact, already fully automatic** (a single hardcoded Celery chain) — contrary to the example given in the prompt, this particular pipeline is *not* a gap in this codebase. This is worth stating explicitly since it's the best-automated flow in the system.
3. **ATS Resume Generation is isolated from Matching.** A high `ResumeJobMatch` score never triggers ATS-resume creation; the user must manually start it from scratch for every job of interest.
4. **Auto-Apply is isolated from ATS Resume Generation.** A completed, `ats_resume_ready` tailored resume never surfaces a "ready to apply" action; the user creates the Auto-Apply task independently and must re-navigate to find/select the right resume (though `resume_selection.py` does correctly *prefer* a completed ATS resume once a task is created).
5. **Career Coach is fully isolated** — no upstream module (resume parsing, matching, tracking) ever triggers a coach insight. It is 100% pull-based; a user who never opens `/coach` gets zero value from it, even though the backend already has all the context (parsed resume, job market data) needed to generate at least a "resume review" and "job readiness score" automatically.
6. **Notifications are not truly event-driven** — `NotificationService.notify(...)` is a direct synchronous method call hardcoded into 3 call sites (`workspace.update_dashboard`, `matching.match_new_job_for_active_users`, `tracking.change_status`, `email_tracking.process_email`/`_flag_needs_reauth`), not a subscriber reacting to a published domain event. Adding a new notification trigger means editing the source of whichever module should fire it.
7. **Dashboard (Workspace) does not proactively refresh on its own timer** — it updates only via (a) explicit navigation/refetch or (b) a handful of specific WebSocket event types. Widgets not covered by any WS event (e.g., Analytics, which lives on a separate page entirely) are page-load-only.
8. **Application Tracker only receives automatic status updates from one source (Gmail email classification)** — there's no equivalent automatic ingestion from, say, a calendar integration or an ATS webhook; everything else requires a manual status edit.
9. **`services/recommendation/service.py::RecommendationService` is dead code** — constructor is a no-op and its one public method raises `NotImplementedError("... not implemented in Phase 0.")`. It is never imported or wired anywhere; all real recommendation logic actually lives in `modules/matching/service.py`. This looks like an abandoned earlier design that was superseded but never deleted.
10. **`features/dashboard/` (frontend) is orphaned** — a full page + 6 components exist but no route mounts it; `/dashboard` redirects to `/workspace` instead. Likely a leftover from before Workspace absorbed its responsibilities.
11. **`pages/ServerErrorPage.tsx` (frontend) is orphaned** — built but not referenced by any route or error boundary.
12. **Pinned/Saved items and Weekly Goals exist only in browser `localStorage`** — there is no backend table or endpoint for either, so this state is lost on a new device/browser and invisible to any backend logic (e.g., recommendations can't take "pinned jobs" into account).
13. **No backend endpoints exist for email verification, forgot-password, or reset-password**, despite the frontend having fully-built pages for all three (`VerifyEmailPage`, `ForgotPasswordPage`, `ResetPasswordPage`). `register` currently issues a valid access token immediately with no verification step at all.
14. **Push notifications are a no-op placeholder** (`services/notification/push_sender.py::PushSender.send()` just logs) — the `PUSH` channel exists end-to-end in the data model (`NotificationChannel.PUSH`) and preference system but delivers nothing.
15. **Job crawling doesn't auto-start for a newly registered company** — `CompanyService.register_company` does not enqueue an initial `fetch_company_jobs_task`; the company sits idle until Beat's periodic sweep happens to consider it "due," or an operator manually calls the crawl endpoint.
16. **Documentation/code drift**: several module docstrings (`matching/router.py`, and similar comments in other modules) describe a "trusted `X-User-Id` header" as the current auth mechanism for internal APIs — but the actual, single, working implementation (`api/deps.py::get_current_user_id`) uses verified JWT bearer tokens everywhere, with no `X-User-Id` header logic present anywhere in the codebase. This is a stale comment, not a real inconsistency in the security model, but it could mislead a future contributor.
17. **Analytics and Admin are pull-only**, with no push/websocket updates and no scheduled refresh — both are always a manual page-load away from current data.
18. **Search and Job Ingestion have no "saved search" or alerting feature** — a user can search/filter jobs but there's no way to persist a query and get notified when new matching jobs appear (the only proactive job-alert mechanism is the unrelated `matching.match_new_job_for_active_users` "smart trigger," which is driven by resume-embedding similarity, not by a saved keyword search).

---

## 18. Recommended Workflow

This proposes the ideal end-to-end flow given what's already built vs. what's still disconnected (§17). It reuses the existing automatic resume pipeline as-is (it's already good) and extends true event-driven automation to the modules that are currently isolated (ATS Resume, Auto Apply, Career Coach, Notifications), using the **existing but unused `EventBus`** as the actual mechanism — rather than more hardcoded `.delay()` chains — so future modules can subscribe without editing upstream code.

```
Register  (with real email verification — new endpoint, blocks login until verified)
     ↓
Login
     ↓
Onboarding  (target roles, preferences)
     ↓
Resume Upload
     ↓
Automatic Parsing  (existing — unchanged)
     ↓
Automatic Embedding  (existing — unchanged)
     ↓
Automatic Resume Analysis        ← NEW: publish "resume.analyzed" event
     ↓                              subscriber: ai_career_coach auto-generates a first
     ↓                              RESUME_REVIEW + JOB_READINESS_SCORE CareerInsight
Automatic ATS Score  (existing baseline — unchanged)
     ↓
Automatic Skill Extraction  (existing, job-side — unchanged)
     ↓
Automatic Job Matching  (existing — unchanged)         ← publish "matches.computed" event
     ↓                                                     subscriber: for any match ≥ configurable
     ↓                                                     threshold, auto-queue an ATS Resume
     ↓                                                     generation job (not auto-submit — just
     ↓                                                     pre-generate so it's one click to apply)
Automatic Recommendations  (existing — unchanged)
     ↓
Dashboard Update  (existing WS bridge — unchanged, extend to cover Analytics widgets too)
     ↓
Career Coach Update              ← surfaced automatically on the dashboard (no click needed)
     ↓                              instead of requiring the user to open /coach first
Notifications                    ← re-platform NotificationService.notify() as an EventBus
     ↓                              subscriber (still same DB row + email/push delivery),
     ↓                              so new triggers are additive, not edits to upstream modules
Job Search  (existing hybrid search — unchanged)         ← ADD: saved-search + alert subscription
     ↓
Auto Apply  (existing adapter/validation pipeline — unchanged)
     ↓                              from the pre-generated ATS resume, one-click launch
Status Tracking  (existing — unchanged)
     ↓
Email Parsing  (existing Gmail sweep — unchanged)
     ↓
Analytics Update                 ← ADD: push analytics deltas over the existing workspace
     ↓                              WebSocket instead of leaving it page-load-only
Career Insights                  ← periodic re-generation (e.g. weekly) via Celery Beat,
                                     not purely reactive to a single resume upload
```

**Key structural recommendation**: promote the already-built-but-unused `EventBus` from dead scaffolding to the actual coupling mechanism between modules, starting with the five real domain events already implicit in the code (`resume.parsed`, `resume.embedded`, `matches.computed`, `ats_resume.ready`, `application.status_changed`). Each currently-hardcoded `task.delay(...)` chain call becomes a `publish()` instead, and new consumers (Career Coach auto-insights, Auto-Apply pre-staging, saved-search alerts) subscribe independently — closing §17's gaps without every new feature needing to edit an unrelated upstream module's source file.

---

*This report reflects the implementation as read on 2026-07-24. No code, configuration, or data was modified during this audit.*
