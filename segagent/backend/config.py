from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
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
    # Qwen3.5 is natively multimodal and exposes a thinking channel. The 9B
    # checkpoint is the deployable default; the 27B checkpoint remains an
    # evaluation candidate and can be selected with SEGAGENT_LLM_MODEL.
    llm_model: str = "Qwen/Qwen3.5-9B"
    # Thinking is a product requirement: the raw model channel is streamed and
    # displayed separately from clinical evidence. Templates without a thinking
    # switch simply ignore the argument.
    enable_thinking: bool = True
    planner: Literal["qwen", "rule"] = "qwen"
    # ``device`` is retained as the backwards-compatible fallback. Production
    # deployments should place the two model stacks explicitly so VoxTell does
    # not collide with a sharded Qwen process.
    device: str = "auto"
    llm_device: str | None = None
    llm_input_device: str | None = None
    voxtell_device: str | None = None
    llm_device_map: Literal[
        "auto", "balanced", "balanced_low_0", "sequential"
    ] = "balanced_low_0"
    # Accelerate-style memory map. JSON environment example:
    # {"0":"18GiB","1":"18GiB","2":"18GiB","cpu":"64GiB"}
    # Omitting a CUDA device excludes it from automatic placement.
    llm_max_memory: dict[str, str] = Field(default_factory=dict)
    # Keep a single in-process generation active. Activations and KV caches are
    # request-local, so concurrent calls can exhaust otherwise adequate GPUs.
    llm_max_concurrent_generations: int = Field(default=1, ge=1, le=4)
    # These are aggregate model-input controls, not viewer controls. All masks
    # remain independently visible in the frontend.
    llm_max_prompt_images: int = Field(default=12, ge=3, le=64)
    llm_max_artifacts: int = Field(default=2, ge=0, le=16)
    llm_mask_axial_views: int = Field(default=1, ge=1, le=4)
    llm_min_image_tokens: int = Field(default=256, ge=64, le=4096)
    llm_max_image_tokens: int = Field(default=512, ge=64, le=4096)
    # Large OAR sets remain one semantic tool request, but VoxTell processes the
    # prompts in bounded internal chunks to limit its activation peak.
    voxtell_batch_size: int = Field(default=4, ge=1, le=32)
    max_steps: int = Field(default=8, ge=2, le=24)
    montage_slices: int = Field(default=3, ge=2, le=16)
    overlay_slices: int = Field(default=3, ge=1, le=8)
    # 512 tokens can end inside <think> before the structured decision is
    # emitted. Leave enough room for both reasoning and the validated payload.
    max_new_tokens: int = Field(default=1536, ge=256, le=8192)
    retrieve_k: int = Field(default=4, ge=1, le=20)
    semantic_weight: float = Field(default=0.45, ge=0.0, le=1.0)
    trace_content: bool = False
    api_base_url: str = "http://127.0.0.1:8000"
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    @property
    def allowed_origins(self) -> list[str]:
        return [item.strip() for item in self.cors_origins.split(",") if item.strip()]

    @property
    def resolved_llm_device(self) -> str:
        return (self.llm_device or self.device).strip()

    @property
    def resolved_voxtell_device(self) -> str:
        return (self.voxtell_device or self.device).strip()

    @model_validator(mode="after")
    def validate_image_budget(self) -> "Settings":
        if self.llm_min_image_tokens > self.llm_max_image_tokens:
            raise ValueError(
                "llm_min_image_tokens cannot exceed llm_max_image_tokens"
            )
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.checkpoint_db.parent.mkdir(parents=True, exist_ok=True)
    return settings
