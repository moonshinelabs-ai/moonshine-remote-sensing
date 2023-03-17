"""Only use loguru if it's present in the environment."""
from typing import Any

logger: Any = None
try:
    from loguru import logger as logger
except:
    import logging

    logger = logging.getLogger("moonshine")
