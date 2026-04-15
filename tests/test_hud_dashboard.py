from __future__ import annotations


def test_hud_dashboard_exposes_indexing_and_ops(jarvis_app) -> None:
    dashboard = jarvis_app.hud_runtime_service.dashboard()
    runtime_names = {panel["name"] for panel in dashboard["runtimes"]}
    service_names = {service["name"] for service in dashboard["services"]}
    assert "indexing" in runtime_names
    assert "ops" in runtime_names
    assert "hud_runtime" in service_names
    assert "health_summary" in dashboard
