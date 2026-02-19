"""
Proactive System - Doris takes initiative.

Sources feed events → Evaluator decides → Executor acts → Notifier tells the user.
"""

from .models import ProactiveEvent, ProactiveAction, EvaluationResult
from .db import init_proactive_db, get_pending_events, save_event, save_action
from .evaluator import evaluate_event
from .executor import execute_action
from .notifier import notify_action, handle_correction
from .scheduler import ProactiveScheduler, init_scheduler, shutdown_scheduler, get_scheduler

__all__ = [
    # Models
    "ProactiveEvent",
    "ProactiveAction",
    "EvaluationResult",
    # Database
    "init_proactive_db",
    "get_pending_events",
    "save_event",
    "save_action",
    # Core functions
    "evaluate_event",
    "execute_action",
    "notify_action",
    "handle_correction",
    # Scheduler
    "ProactiveScheduler",
    "init_scheduler",
    "shutdown_scheduler",
    "get_scheduler",
]
