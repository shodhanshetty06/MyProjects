"""Application-wide constants that are not environment-configurable."""
from enum import Enum


class APIVersion(str, Enum):
    V1 = "v1"
    V2 = "v2"


class HealthStatus(str, Enum):
    OK = "ok"
    DEGRADED = "degraded"
    DOWN = "down"


REQUEST_ID_HEADER = "X-Request-ID"
DEFAULT_PAGE_SIZE = 20
MAX_PAGE_SIZE = 100
