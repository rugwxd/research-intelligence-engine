"""Tests for monitoring and metrics collection."""

import json

import pytest

from src.monitoring import LatencyTracker, MetricsCollector, health_check


@pytest.fixture
def collector():
    return MetricsCollector()


class TestMetricsCollector:
    def test_increment_counter(self, collector):
        collector.increment("requests")
        collector.increment("requests")
        summary = collector.get_summary()
        assert summary["counters"]["requests"] == 2.0

    def test_increment_with_labels(self, collector):
        collector.increment("errors", labels={"component": "retriever"})
        collector.increment("errors", labels={"component": "generator"})
        summary = collector.get_summary()
        assert 'errors{component="retriever"}' in summary["counters"]
        assert 'errors{component="generator"}' in summary["counters"]

    def test_observe_histogram(self, collector):
        for val in [10, 20, 30, 40, 50]:
            collector.observe("latency_ms", val)

        summary = collector.get_summary()
        hist = summary["histograms"]["latency_ms"]
        assert hist["count"] == 5
        assert hist["mean"] == 30.0
        assert hist["min"] == 10
        assert hist["max"] == 50

    def test_set_gauge(self, collector):
        collector.set_gauge("index_size", 1000)
        summary = collector.get_summary()
        assert summary["gauges"]["index_size"] == 1000

        collector.set_gauge("index_size", 2000)
        summary = collector.get_summary()
        assert summary["gauges"]["index_size"] == 2000

    def test_export_prometheus(self, collector):
        collector.increment("requests_total")
        collector.set_gauge("index_vectors", 500)
        collector.observe("latency_ms", 100)

        prom_output = collector.export_prometheus()
        assert "requests_total" in prom_output
        assert "index_vectors" in prom_output
        assert "latency_ms" in prom_output

    def test_save_snapshot(self, collector, tmp_path):
        collector.increment("test_counter")
        output = collector.save_snapshot(str(tmp_path / "metrics.json"))

        assert output.exists()
        data = json.loads(output.read_text())
        assert "counters" in data
        assert "exported_at" in data


class TestLatencyTracker:
    def test_tracks_latency(self, collector):
        import time

        with LatencyTracker(collector, "test_op"):
            time.sleep(0.01)

        summary = collector.get_summary()
        assert "latency_ms.test_op" in summary["histograms"]
        assert summary["histograms"]["latency_ms.test_op"]["mean"] >= 10

    def test_increments_request_count(self, collector):
        with LatencyTracker(collector, "test_op"):
            pass

        summary = collector.get_summary()
        assert summary["counters"]["requests.test_op"] == 1.0


class TestHealthCheck:
    def test_health_check_returns_status(self):
        status = health_check()
        assert "status" in status
        assert "components" in status
        assert "vector_index" in status["components"]
        assert "paper_data" in status["components"]
        assert "anthropic_api" in status["components"]
