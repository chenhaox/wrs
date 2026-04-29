"""
SEALP Layout Optimization
===========================

Workspace layout representation, feasibility evaluation, and
optimization for sequence-aware dual-arm assembly planning.

Modules
-------
layout
    ``WorkspaceLayout`` dataclass and YAML serialization.
reachability
    IK reachability checks for target poses.
manipulability
    Yoshikawa manipulability scoring.
feasibility
    Per-step feasibility evaluation and ``FeasibilityReport``.
metrics
    Standardized layout evaluation metrics.
optimizer
    Abstract optimizer interface.
optimizer_random
    Random search baseline optimizer.
"""

from .layout import WorkspaceLayout
from .feasibility import FeasibilityReport, StepFeasibility, evaluate_layout
from .metrics import LayoutMetrics, compute_metrics
