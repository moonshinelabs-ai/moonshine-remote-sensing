"""Only use loguru if it's present in the environment."""
from typing import Any

logger: Any = None
try:
    from loguru import logger as loguru_logger

    logger = loguru_logger
except:
    import logging

    logger = logging.getLogger("moonshine")
