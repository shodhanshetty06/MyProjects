# AI-Powered Job Search Platform — Phase 0 Foundation

Enterprise-grade project foundation only. **No business logic is implemented in this phase.**
Every module, service, and endpoint is a wired-but-empty skeleton meant to be filled in during
subsequent phases.

## Architecture

Clean Architecture, layered per module:

```
router (interface)  →  service (application/use-case)  →  repository (data access)  →  models (domain/ORM)
                                     ↑
                                schemas (I/O contracts, pydantic)
```

Cross-cutting concerns (config, logging, DI, exception handling, middleware, DB session,
Redis, Celery, S3, event bus) live under `backend/app/core`, `backend/app/db`,
`backend/app/integrations`, `backend/app/middleware`, and `backend/app/events`, and are
consumed by every module via dependency injection (`app/api/deps.py`).

## Stack

| Concern        | Choice                                   |
|-----------------|-------------------------------------------|
| API framework   | FastAPI                                   |
| ORM             | SQLAlchemy 2.0 (async)                    |
| Migrations      | Alembic                                   |
| Database        | PostgreSQL + pgvector                     |
| Cache / broker  | Redis                                     |
| Background jobs | Celery (worker + beat)                    |
| Object storage  | AWS S3 (boto3)                            |
| Frontend        | React + TypeScript + Vite                 |
| Containerization| Docker / Docker Compose                   |

## Repository layout

```
job-search-platform/
├── backend/
│   ├── app/
│   │   ├── core/            # config, logging, DI, exceptions, constants
│   │   ├── api/              # versioned routers (v1, v2 placeholder), deps
│   │   ├── middleware/        # request context, logging, exception handling
│   │   ├── db/                 # engine, session, declarative base, mixins
│   │   ├── models/            # shared ORM mixins
│   │   ├── schemas/           # shared pydantic contracts (BaseResponse, etc.)
│   │   ├── repositories/      # BaseRepository (generic CRUD)
│   │   ├── services/          # BaseService + cross-cutting service skeletons
│   │   │   ├── notification/
│   │   │   ├── search/
│   │   │   ├── recommendation/
│   │   │   └── ai/
│   │   ├── events/            # in-process event bus
│   │   ├── workers/           # Celery app, base task, task registry
│   │   ├── integrations/      # Redis, S3, pgvector clients
│   │   ├── utils/
│   │   └── modules/           # feature modules (Clean Architecture per module)
│   │       ├── auth/
│   │       ├── user_profile/
│   │       ├── resume_parser/
│   │       ├── job_ingestion/
│   │       ├── search/
│   │       ├── matching/
│   │       ├── ats_resume_generator/
│   │       ├── auto_apply/
│   │       ├── tracking/
│   │       ├── notification/
│   │       ├── ai_career_coach/
│   │       ├── analytics/
│   │       └── admin/
│   ├── alembic/
│   ├── tests/
│   └── Dockerfile
├── frontend/
│   └── src/
│       ├── api/                # generated/typed API client
│       ├── features/           # one folder per backend module
│       ├── components/
│       ├── layouts/
│       ├── routes/
│       └── store/
├── docker-compose.yml
└── .env.example
```

## Getting started

```bash
cp .env.example .env
docker compose up --build
```

- API: http://localhost:8000
- API docs (Swagger): http://localhost:8000/api/v1/docs
- ReDoc: http://localhost:8000/api/v1/redoc
- Health check: http://localhost:8000/api/v1/health
- Frontend: http://localhost:3000

Run migrations:

```bash
docker compose exec api alembic upgrade head
```

## Adding a new module

Each entry under `backend/app/modules/<name>/` follows the same skeleton:

```
<name>/
├── __init__.py
├── models.py        # SQLAlchemy models (empty — no tables yet)
├── schemas.py        # Pydantic request/response contracts
├── repository.py     # extends BaseRepository
├── service.py         # extends BaseService, injected with repository
└── router.py           # FastAPI APIRouter, injected with service
```

Register the module's router in `backend/app/api/v1/router.py`.
