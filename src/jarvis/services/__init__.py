"""Runtime service interfaces."""

from .runtime import JarvisRuntimeService

from .result_views import (
    summarize_autonomy_view,
    summarize_error,
    summarize_model_response,
    summarize_ops_status,
    summarize_research_report,
    summarize_research_task,
    summarize_science_result,
    summarize_security_result,
    summarize_system_operation,
    summarize_system_search,
    summarize_writing_receipt,
)

__all__ = [
    "JarvisRuntimeService",
    "summarize_autonomy_view",
    "summarize_error",
    "summarize_model_response",
    "summarize_ops_status",
    "summarize_research_report",
    "summarize_research_task",
    "summarize_science_result",
    "summarize_security_result",
    "summarize_system_operation",
    "summarize_system_search",
    "summarize_writing_receipt",
]
