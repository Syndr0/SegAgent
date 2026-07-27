from __future__ import annotations

from functools import lru_cache

from .config import Settings, get_settings
from .expert import VoxTellBackend
from .intent import IntentClassifier
from .knowledge import HybridKnowledgeBase
from .observability import Tracing
from .planner import QwenStructuredPlanner, RulePlanner
from .storage import ResearchStore
from .tools import EvidenceCritic, ProtocolLookupTool, SegmentationTool
from .workflow import SegAgentGraph, WorkflowService


class AppServices:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.store = ResearchStore(settings.data_dir)
        self.expert = VoxTellBackend(settings.voxtell_model, settings.device)
        self.knowledge = HybridKnowledgeBase(
            settings.knowledge_dir,
            embed_fn=self.expert.embed,
            semantic_weight=settings.semantic_weight,
        )
        if settings.planner == "rule":
            self.planner = RulePlanner(self.knowledge)
        else:
            self.planner = QwenStructuredPlanner(
                settings.llm_model,
                self.store,
                settings.montage_slices,
                settings.max_new_tokens,
                settings.device,
            )
        self.protocol_tool = ProtocolLookupTool(self.knowledge, settings.retrieve_k)
        self.segmentation_tool = SegmentationTool(
            self.store, self.expert, settings.overlay_slices
        )
        self.critic = EvidenceCritic()
        self.intent_classifier = IntentClassifier.from_knowledge(self.knowledge)
        self.graph = SegAgentGraph(
            settings,
            self.planner,
            self.protocol_tool,
            self.segmentation_tool,
            self.critic,
            Tracing(),
        )
        self.workflow = WorkflowService(self.store, self.graph, self.intent_classifier)


@lru_cache(maxsize=1)
def get_services() -> AppServices:
    return AppServices(get_settings())

