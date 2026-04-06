"""
Motion Primitive Library
=========================

Reusable motion primitives for assembly operations.
Each primitive wraps WRS planners into a high-level interface.

Available primitives:
    - ``TransportPrimitive``     — single-arm pick-transport-place
    - ``DualTransportPrimitive`` — dual-arm cooperative transport
"""

from .base import MotionPrimitive
from .transport import TransportPrimitive
from .dual_transport import DualTransportPrimitive
