"""Monitoring, metrics, and observability for the RAG pipeline.

Provides structured metric tracking for latency, throughput, and
quality signals. Exports metrics as JSON for dashboarding and
supports Prometheus-compatible exposition.
"""

import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any

from src.config import PROJECT_ROOT

logger = logging.getLogger(__name__)


@dataclass
class MetricSample:
    """A single metric observation."""

    name: str
    value: float
    timestamp: float
    labels: dict[str, str] = field(default_factory=dict)


class MetricsCollector:
    """Thread-safe metrics collector for pipeline observability.

    Tracks:
        - Query latency (p50, p90, p99)
        - Retrieval quality scores per query
        - Embedding throughput
        - Error rates by component
        - Token usage and cost estimates
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._counters: dict[str, float] = defaultdict(float)
        self._histograms: dict[str, list[float]] = defaultdict(list)
        self._gauges: dict[str, float] = {}

    def increment(self, name: str, value: float = 1.0, labels: dict[str, str] | None = None) -> None:
        """Increment a counter metric."""
        key = self._key(name, labels)
        with self._lock:
            self._counters[key] += value

    def observe(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
        """Record an observation for a histogram metric."""
        key = self._key(name, labels)
        with self._lock:
            self._histograms[key].append(value)

    def set_gauge(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
        """Set a gauge metric to a specific value."""
        key = self._key(name, labels)
        with self._lock:
            self._gauges[key] = value

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of all collected metrics."""
        with self._lock:
            summary: dict[str, Any] = {
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "histograms": {},
            }

            for name, values in self._histograms.items():
                if not values:
                    continue
                sorted_vals = sorted(values)
                n = len(sorted_vals)
                summary["histograms"][name] = {
                    "count": n,
                    "mean": sum(sorted_vals) / n,
                    "min": sorted_vals[0],
                    "max": sorted_vals[-1],
                    "p50": sorted_vals[int(n * 0.5)],
                    "p90": sorted_vals[int(n * 0.9)],
                    "p99": sorted_vals[min(int(n * 0.99), n - 1)],
                }

            return summary

    def export_prometheus(self) -> str:
        """Export metrics in Prometheus text exposition format."""
        lines = []
        with self._lock:
            for name, value in self._counters.items():
                metric_name = name.replace(".", "_").replace("-", "_")
                lines.append(f"# TYPE {metric_name} counter")
                lines.append(f"{metric_name} {value}")

            for name, value in self._gauges.items():
                metric_name = name.replace(".", "_").replace("-", "_")
                lines.append(f"# TYPE {metric_name} gauge")
                lines.append(f"{metric_name} {value}")

            for name, values in self._histograms.items():
                if not values:
                    continue
                metric_name = name.replace(".", "_").replace("-", "_")
                sorted_vals = sorted(values)
                n = len(sorted_vals)
                lines.append(f"# TYPE {metric_name} summary")
                lines.append(f'{metric_name}{{quantile="0.5"}} {sorted_vals[int(n * 0.5)]}')
                lines.append(f'{metric_name}{{quantile="0.9"}} {sorted_vals[int(n * 0.9)]}')
                lines.append(f'{metric_name}{{quantile="0.99"}} {sorted_vals[min(int(n * 0.99), n - 1)]}')
                lines.append(f"{metric_name}_count {n}")
                lines.append(f"{metric_name}_sum {sum(sorted_vals)}")

        return "\n".join(lines)

    def save_snapshot(self, path: str | None = None) -> Path:
        """Save current metrics to a JSON file."""
        if path is None:
            output = PROJECT_ROOT / "logs" / "metrics_snapshot.json"
        else:
            output = Path(path)

        output.parent.mkdir(parents=True, exist_ok=True)
        summary = self.get_summary()
        summary["exported_at"] = time.time()

        with open(output, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info("Metrics snapshot saved to %s", output)
        return output

    @staticmethod
    def _key(name: str, labels: dict[str, str] | None = None) -> str:
        if not labels:
            return name
        label_str = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"


class LatencyTracker:
    """Context manager for tracking operation latency."""

    def __init__(self, collector: MetricsCollector, operation: str) -> None:
        self.collector = collector
        self.operation = operation
        self._start: float = 0

    def __enter__(self) -> "LatencyTracker":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args) -> None:
        elapsed_ms = (time.perf_counter() - self._start) * 1000
        self.collector.observe(f"latency_ms.{self.operation}", elapsed_ms)
        self.collector.increment(f"requests.{self.operation}")


# Global metrics instance
metrics = MetricsCollector()


def health_check() -> dict[str, Any]:
    """Run a system health check.

    Returns:
        Health status dict with component readiness.
    """
    status: dict[str, Any] = {
        "status": "healthy",
        "timestamp": time.time(),
        "components": {},
    }

    # Check FAISS index
    index_path = PROJECT_ROOT / "data" / "vectordb" / "faiss.index"
    status["components"]["vector_index"] = {
        "status": "ready" if index_path.exists() else "not_built",
        "path": str(index_path),
    }

    # Check paper data
    papers_path = PROJECT_ROOT / "data" / "raw" / "papers.json"
    status["components"]["paper_data"] = {
        "status": "ready" if papers_path.exists() else "not_ingested",
        "path": str(papers_path),
    }

    # Check API key
    import os
    has_key = bool(os.getenv("ANTHROPIC_API_KEY"))
    status["components"]["anthropic_api"] = {
        "status": "configured" if has_key else "missing_key",
    }

    # Overall status
    component_statuses = [c["status"] for c in status["components"].values()]
    if any(s in ("not_built", "missing_key") for s in component_statuses):
        status["status"] = "degraded"

    return status
