"""Aggregates every ORM model so Alembic autogenerate can see the full schema.

Import each module's `models` here as models are added. This file has no
runtime purpose beyond making `Base.metadata` complete — it should only
ever contain imports.
"""

from app.db.base_class import Base  # noqa: F401
from app.modules.auth.models import *  # noqa: F401,F403
from app.modules.job_ingestion.models import *  # noqa: F401,F403

# Module models will be imported here as they are implemented, e.g.:
# from app.modules.user_profile.models import *  # noqa: F401,F403
from app.modules.resume_parser.models import *  # noqa: F401,F403
from app.modules.matching.models import *  # noqa: F401,F403
from app.modules.tracking.models import *  # noqa: F401,F403
from app.modules.ats_resume_generator.models import *  # noqa: F401,F403
from app.modules.ai_career_coach.models import *  # noqa: F401,F403
