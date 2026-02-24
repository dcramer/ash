"""Config RPC method handlers."""

import logging
import tomllib
from typing import TYPE_CHECKING, Any

from ash.config.paths import get_config_path

if TYPE_CHECKING:
    from ash.config import AshConfig
    from ash.rpc.server import RPCServer
    from ash.skills import SkillRegistry

logger = logging.getLogger(__name__)


def register_config_methods(
    server: "RPCServer",
    config: "AshConfig",
    skill_registry: "SkillRegistry | None" = None,
) -> None:
    """Register config-related RPC methods.

    Args:
        server: RPC server to register methods on.
        config: Application configuration instance.
        skill_registry: Optional skill registry to reload.
    """

    async def config_reload(params: dict[str, Any]) -> dict[str, Any]:
        """Reload configuration from disk.

        This reloads skill configurations to pick up new API keys
        or settings without restarting the server.

        Note: Some config changes (models, providers) require a restart.
        """
        config_path = get_config_path()

        if not config_path.exists():
            return {"success": False, "error": f"Config not found: {config_path}"}

        try:
            with config_path.open("rb") as f:
                raw_config = tomllib.load(f)

            # Reload skill configurations
            skills_config = raw_config.get("skills", {})
            from ash.config.models import SkillConfig, SkillDefaultsConfig

            config.skill_defaults = SkillDefaultsConfig.model_validate(
                skills_config.get("defaults", {})
            )

            # Update existing skill configs and add new ones
            for skill_name, skill_data in skills_config.items():
                if skill_name in {
                    "defaults",
                    "sources",
                    "auto_sync",
                    "update_interval_minutes",
                    "update_interval",
                }:
                    continue
                if isinstance(skill_data, dict):
                    config.skills[skill_name] = SkillConfig.model_validate(skill_data)

            # Reload skill definitions from workspace if registry available
            if skill_registry is not None:
                skill_registry.reload_all(config.workspace)
                logger.info(
                    "skills_reloaded",
                    extra={
                        "skill_count": len(skill_registry),
                        "workspace": str(config.workspace),
                    },
                )

            logger.info("config_reloaded")
            return {"success": True}

        except Exception as e:
            logger.error("config_reload_failed", exc_info=True)
            return {"success": False, "error": str(e)}

    # Register handlers
    server.register("config.reload", config_reload)

    logger.debug("Registered config RPC methods")
