from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator


class Tracing:
    """Small OpenTelemetry facade with a no-op fallback.

    Content is represented only by counts and identifiers unless callers make
    an explicit, separate decision to record sensitive payloads.
    """

    def __init__(self, service_name: str = "segagent-research-v2"):
        self.tracer = None
        try:
            from opentelemetry import trace
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

            provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
            try:
                provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
            except Exception:
                pass
            if trace.get_tracer_provider().__class__.__name__ == "ProxyTracerProvider":
                trace.set_tracer_provider(provider)
            self.tracer = trace.get_tracer(service_name)
        except Exception:
            self.tracer = None

    @contextmanager
    def span(self, name: str, **attributes) -> Iterator[object | None]:
        if self.tracer is None:
            yield None
            return
        with self.tracer.start_as_current_span(name) as span:
            for key, value in attributes.items():
                if value is not None:
                    span.set_attribute(key, value)
            yield span

