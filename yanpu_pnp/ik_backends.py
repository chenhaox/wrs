DEFAULT_IK_BACKEND = "tracik"

_TRACIK_ALIASES = {"t", "tracik", "pytracik"}
_IKFAST_ALIASES = {"fast", "ikfast", "pyikfast"}
_NUMERICAL_ALIASES = {"n", "numik", "numerical"}


def normalize_backend(value=None):
    if value is None:
        return DEFAULT_IK_BACKEND
    backend = str(value).strip().lower()
    if backend in _TRACIK_ALIASES:
        return "tracik"
    if backend in _IKFAST_ALIASES:
        return "ikfast"
    if backend in _NUMERICAL_ALIASES:
        return "numerical"
    return backend


def is_multi_solution_backend(value=None):
    return normalize_backend(value) == "ikfast"


def robot_solver_name(value=None):
    backend = normalize_backend(value)
    if backend == "tracik":
        return "tracik"
    if backend == "ikfast":
        return "ikfast"
    if backend == "numerical":
        return "n"
    return backend


def resolve_backend(base_cfg=None, search_cfg=None, override=None):
    if override is not None:
        return normalize_backend(override)
    if search_cfg is not None and search_cfg.get("ik_backend") is not None:
        return normalize_backend(search_cfg["ik_backend"])
    if base_cfg is not None and base_cfg.get("pnp", {}).get("ik_backend") is not None:
        return normalize_backend(base_cfg["pnp"]["ik_backend"])
    return DEFAULT_IK_BACKEND


def set_backend(cfg, backend):
    cfg.setdefault("pnp", {})["ik_backend"] = normalize_backend(backend)
    return cfg
