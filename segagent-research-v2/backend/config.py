from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_DIR = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    """Environment-backed research configuration.

    Patient content is never included in traces unless the explicit
    ``trace_content`` switch is enabled.
    """

    model_config = SettingsConfigDict(
        env_prefix="SEGAGENT_", env_file=".env", extra="ignore"
    )

    data_dir: Path = PROJECT_DIR / ".research_data"
    knowledge_dir: Path = PROJECT_DIR / "knowledge"
    checkpoint_db: Path = PROJECT_DIR / ".research_data" / "checkpoints.sqlite"
    voxtell_model: Path | None = None
    llm_model: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    planner: Literal["qwen", "rule"] = "qwen"
    device: str = "auto"
    max_steps: int = Field(default=8, ge=2, le=24)
    montage_slices: int = Field(default=6, ge=2, le=16)
    overlay_slices: int = Field(default=3, ge=1, le=8)
    max_new_tokens: int = Field(default=512, ge=64, le=4096)
    require_mask_approval: bool = True
    retrieve_k: int = Field(default=4, ge=1, le=20)
    semantic_weight: float = Field(default=0.45, ge=0.0, le=1.0)
    trace_content: bool = False
    api_base_url: str = "http://127.0.0.1:8000"
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    @property
    def allowed_origins(self) -> list[str]:
        return [item.strip() for item in self.cors_origins.split(",") if item.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.checkpoint_db.parent.mkdir(parents=True, exist_ok=True)
    return settings

