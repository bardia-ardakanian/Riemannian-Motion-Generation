"""
Single source of truth for "which flow is this checkpoint?".

Before this, rmg_eval.load_model inferred the flow from which optional keys happened
to be present in the checkpoint dict (`mixture`, `flow6d`, `endpoint_euclidean`,
`euclidean`, `param == "endpoint"`, `vfm`, else base). That chain is how the 6D model
was briefly evaluated with the wrong flow and how a flow whose sample() had a different
signature crashed a sweep half way through.

Every trainer should write ckpt["flow"] = <one of FLOWS>. load_flow() prefers that tag
and only falls back to the legacy inference for older checkpoints.

    from flow_registry import FLOWS, build_flow, load_flow, describe
"""
import importlib

# name -> (module, class, human description, extra kwargs the flow accepts)
FLOWS = {
    "vel_int": ("rmg_flow",                    "RMGFlow",
                "velocity target, intrinsic geometry (the baseline / RMG)",
                ("sigma_trans", "sigma_rot", "canon_prior")),
    "vel_ext": ("rmg_flow_euclidean",          "RMGFlowEuclidean",
                "velocity target, extrinsic (ambient) geometry",
                ("sigma_trans", "sigma_rot")),
    "ep_int":  ("rmg_flow_endpoint",           "RMGFlowEndpoint",
                "endpoint target, intrinsic geometry (Algorithm 1)",
                ("sigma_trans", "sigma_rot", "canon_prior")),
    "ep_ext":  ("rmg_flow_endpoint_euclidean", "RMGFlowEndpointEuclidean",
                "endpoint target, extrinsic geometry (Algorithm 2)",
                ("sigma_trans", "sigma_rot")),
    "6d":      ("rmg_flow_6d",                 "RMGFlow6D",
                "velocity target on the 6D rotation representation (extrinsic)",
                ("sigma_trans", "sigma_rot")),
    "vfm":     ("rmg_flow_vfm",                "RMGFlowVFM",
                "variational endpoint (von Mises-Fisher), intrinsic",
                ("sigma_trans", "sigma_rot")),
    "vfm_mix": ("rmg_flow_vfm_mix",            "RMGFlowVFMMix",
                "variational endpoint, mixture of vMF",
                ("sigma_trans", "sigma_rot", "K")),
}

# Network input/output width per flow. The default frame vector is 91 numbers
# (3 translation + 22*4 quaternion); flows that deviate MUST be listed here, or the
# model is built with the wrong head and the reshape fails at the first batch.
J = 22
IN_DIM = {"6d": 3 + J * 6}                       # everything else: 91
OUT_DIM = {
    "6d":      3 + J * 6,                        # 135
    "vfm":     6 + J * 5,                        # 116: trans delta+log_sigma, per joint tangent+log_kappa
    "vfm_mix": lambda K=4: 6 + J * K * 6,        # 534 at K=4
}


def io_dims(name, K=4):
    """(in_dim, out_dim) for building RMGTransformer, or (None, None) for the default."""
    o = OUT_DIM.get(name)
    if callable(o):
        o = o(K)
    return IN_DIM.get(name), o


def build_flow(name, **kw):
    """Instantiate a flow by registry name, passing only kwargs it accepts."""
    if name not in FLOWS:
        raise KeyError(f"unknown flow {name!r}; known: {sorted(FLOWS)}")
    mod, cls, _desc, accepted = FLOWS[name]
    Flow = getattr(importlib.import_module(mod), cls)
    return Flow(**{k: v for k, v in kw.items() if k in accepted})


def flow_name_from_checkpoint(ckpt):
    """Prefer the explicit tag; fall back to the legacy key sniffing for old checkpoints."""
    if ckpt.get("flow") in FLOWS:
        return ckpt["flow"]
    if ckpt.get("mixture"):
        return "vfm_mix"
    if ckpt.get("flow6d"):
        return "6d"
    if ckpt.get("endpoint_euclidean"):
        return "ep_ext"
    if ckpt.get("euclidean"):
        return "vel_ext"
    if ckpt.get("vfm"):
        return "vfm"
    if ckpt.get("param") == "endpoint":
        return "ep_int"
    return "vel_int"


def load_flow(ckpt):
    """Build the flow described by a loaded checkpoint dict."""
    name = flow_name_from_checkpoint(ckpt)
    return name, build_flow(
        name,
        sigma_trans=ckpt.get("sigma_trans", 1.0),
        sigma_rot=ckpt.get("sigma_rot", 1.0),
        canon_prior=ckpt.get("canon_prior", False),
        K=ckpt.get("K", 4),
    )


#: flows whose prior() returns points ON the manifold (unit quaternions). The
#: extrinsic flows deliberately return ambient Gaussians that are NOT unit norm.
INTRINSIC_PRIOR = {"vel_int", "ep_int", "vfm", "vfm_mix"}


def describe(name):
    return FLOWS[name][2]


if __name__ == "__main__":
    for n in FLOWS:
        print(f"{n:8s} {describe(n)}")
