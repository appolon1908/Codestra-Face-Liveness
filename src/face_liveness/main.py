"""Process entrypoint: ``python -m face_liveness.main``."""

from __future__ import annotations

import os

import uvicorn

from .api import create_app
from .config import Settings
from .logging_config import configure_logging


def build_app():  # type: ignore[no-untyped-def]
    settings = Settings()
    configure_logging(settings.log_level)
    return create_app(settings)


def main() -> None:
    uvicorn.run(
        "face_liveness.main:build_app",
        factory=True,
        host=os.environ.get("LIVENESS_HOST", "0.0.0.0"),  # noqa: S104 - container bind
        port=int(os.environ.get("LIVENESS_PORT", "8080")),
        log_config=None,
        access_log=False,
        proxy_headers=False,
        server_header=False,
    )


if __name__ == "__main__":
    main()
