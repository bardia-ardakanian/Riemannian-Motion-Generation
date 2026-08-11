"""
Smoke test for every registered flow. Run this BEFORE starting a sweep: it is what
would have caught the VFM sample() signature mismatch that killed a sweep half way
through, and the 6D input-dimension mismatch.

For each flow it checks:
  - the module imports and the class instantiates
  - prior() returns the right shapes, finite, quaternions unit-norm
  - training_loss() returns a finite scalar and backprops
  - sample() accepts the standard keyword set (including solver=) and returns
    (trans, quats) with unit quaternions and no NaNs
  - one guided and one unguided sample

    python test_flows.py            # all flows
    python test_flows.py vel_int 6d # a subset
"""
import sys, traceback
import torch

from flow_registry import FLOWS, build_flow, io_dims, INTRINSIC_PRIOR, describe
from rmg_model import RMGTransformer

B, L, J, TEXT_DIM = 2, 8, 22, 1024


def check(name):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ind, outd = io_dims(name)
    io = {}
    if ind is not None:
        io["in_dim"] = ind
    if outd is not None:
        io["out_dim"] = outd
    model = RMGTransformer(dim=64, num_layers=2, num_heads=4, ff_mult=2, **io).to(dev)
    flow = build_flow(name, sigma_trans=1.0, sigma_rot=1.0)

    trans1 = torch.randn(B, L, 3, device=dev)
    quats1 = torch.randn(B, L, J, 4, device=dev)
    quats1 = quats1 / quats1.norm(dim=-1, keepdim=True)
    text = torch.randn(B, TEXT_DIM, device=dev)
    mask = torch.ones(B, L, dtype=torch.bool, device=dev)

    # prior
    t0, q0 = flow.prior(B, L, dev)
    assert t0.shape == (B, L, 3), f"prior trans shape {t0.shape}"
    assert q0.shape[:3] == (B, L, J), f"prior quat shape {q0.shape}"
    assert torch.isfinite(t0).all() and torch.isfinite(q0).all(), "prior has non-finite values"
    if name in INTRINSIC_PRIOR:
        n = q0.norm(dim=-1)
        assert (n - 1).abs().max() < 1e-4, \
            f"intrinsic prior must lie on S^3 (max norm deviation {(n-1).abs().max():.2e})"

    # training loss + backward
    loss = flow.training_loss(model, trans1, quats1, text=text, mask=mask, p_drop=0.1)
    assert loss.ndim == 0, f"loss is not a scalar: shape {tuple(loss.shape)}"
    assert torch.isfinite(loss), "loss is not finite"
    loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all()
               for p in model.parameters()), "no finite gradients reached the model"

    # sampling, with the full keyword set the eval harness uses
    with torch.no_grad():
        for guidance in (1.0, 2.5):
            tr, q = flow.sample(model, B, L, mask=mask, text=text, guidance=guidance,
                                n_steps=2, device=dev, solver="euler")
            assert tr.shape == (B, L, 3), f"sample trans shape {tr.shape}"
            assert q.shape == (B, L, J, 4), f"sample quat shape {q.shape}"
            assert torch.isfinite(tr).all() and torch.isfinite(q).all(), "sample has non-finite values"
            nn = q.norm(dim=-1)
            assert (nn - 1).abs().max() < 1e-3, \
                f"sampled quaternions not unit norm (max dev {(nn-1).abs().max():.2e})"
    return True


def main():
    names = sys.argv[1:] or list(FLOWS)
    width = max(len(n) for n in names)
    ok = True
    print(f"smoke-testing {len(names)} flow(s) on "
          f"{'cuda' if torch.cuda.is_available() else 'cpu'}\n")
    for n in names:
        try:
            check(n)
            print(f"  [PASS] {n:<{width}}  {describe(n)}")
        except Exception as e:
            ok = False
            print(f"  [FAIL] {n:<{width}}  {type(e).__name__}: {e}")
            traceback.print_exc(limit=3)
    print("\nall flows OK" if ok else "\nSOME FLOWS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
