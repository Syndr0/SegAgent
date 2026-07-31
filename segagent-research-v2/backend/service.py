from __future__ import annotations

from functools import lru_cache

from .config import Settings, get_settings
from .expert import VoxTellBackend
from .intent import IntentClassifier
from .knowledge import HybridKnowledgeBase
from .observability import Tracing
from .planner import QwenStructuredPlanner, RulePlanner
from .qc import ContourQCEngine
from .storage import ResearchStore
from .tools import (
    ContourQCTool,
    EvidenceCritic,
    MaskInspectionTool,
    ProtocolLookupTool,
    SegmentationTool,
)
from .workflow import SegAgentGraph, WorkflowService


class AppServices:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.store = ResearchStore(settings.data_dir)
        self.expert = VoxTellBackend(
            settings.voxtell_model,
            settings.resolved_voxtell_device,
            settings.voxtell_batch_size,
        )
        self.knowledge = HybridKnowledgeBase(
            settings.knowledge_dir,
            embed_fn=self.expert.embed,
            semantic_weight=settings.semantic_weight,
        )
        if settings.planner == "rule":
            self.planner = RulePlanner(self.knowledge)
        else:
            self.planner = QwenStructuredPlanner(
                model_id=settings.llm_model,
                store=self.store,
                montage_slices=settings.montage_slices,
                max_new_tokens=settings.max_new_tokens,
                device=settings.resolved_llm_device,
                input_device=settings.llm_input_device,
                enable_thinking=settings.enable_thinking,
                device_map=settings.llm_device_map,
                max_memory=settings.llm_max_memory,
                max_concurrent_generations=settings.llm_max_concurrent_generations,
                max_prompt_images=settings.llm_max_prompt_images,
                max_artifacts=settings.llm_max_artifacts,
                mask_axial_views=settings.llm_mask_axial_views,
                min_image_tokens=settings.llm_min_image_tokens,
                max_image_tokens=settings.llm_max_image_tokens,
                known_anatomy=sorted(
                    {
                        str(organ)
                        for protocol in self.knowledge.protocols
                        for organ in protocol.get("oars", [])
                        if organ
                    }
                ),
            )
        self.protocol_tool = ProtocolLookupTool(self.knowledge, settings.retrieve_k)
        self.segmentation_tool = SegmentationTool(
            self.store, self.expert, settings.overlay_slices
        )
        self.inspection_tool = MaskInspectionTool(self.store)
        self.qc_engine = ContourQCEngine(self.store, self.expert, self.knowledge)
        self.qc_tool = ContourQCTool(self.qc_engine)
        self.critic = EvidenceCritic()
        self.intent_classifier = IntentClassifier.from_knowledge(self.knowledge)
        self.graph = SegAgentGraph(
            settings,
            self.planner,
            self.protocol_tool,
            self.segmentation_tool,
            self.critic,
            Tracing(),
            self.inspection_tool,
            self.qc_tool,
        )
        self.workflow = WorkflowService(self.store, self.graph, self.intent_classifier)


@lru_cache(maxsize=1)
def get_services() -> AppServices:
    return AppServices(get_settings())
