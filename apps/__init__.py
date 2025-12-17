"""
Unified application package that bundles both Needle and TVM helpers.
"""
from .models import GCNModel

__all__ = ["GCNModel", "GraphConvNetwork"]


def __getattr__(name):
    if name == "GraphConvNetwork":
        # Lazy import to avoid pulling TVM unless requested
        from .tvm.models import GraphConvNetwork

        return GraphConvNetwork
    raise AttributeError(f"module 'apps' has no attribute '{name}'")
