"""
Smooth loss tests — two independent sections:

  SECTION A: Pure math unit tests (no project imports)
  ─────────────────────────────────────────────────────
  Tests _smooth_loss logic directly using vanilla PyTorch.
  Run these anywhere Python + torch is available:

      python test_smooth_loss.py --unit

  SECTION B: Integration tests (requires full project environment)
  ────────────────────────────────────────────────────────────────
  Tests compute_loss() with a synthetic batch.
  Requires pytorch_lightning, torch_geometric, etc.
  Run on Compute Canada (or wherever the env is set up):

      python test_smooth_loss.py --integration
      python test_smooth_loss.py          # runs both
"""

import sys
import torch


# ╔══════════════════════════════════════════════════════════════════╗
# ║  SECTION A — pure math unit tests (no project imports needed)   ║
# ╚══════════════════════════════════════════════════════════════════╝

# Inline copy of DonutNLL._smooth_loss — zero project dependencies.
def _smooth_loss_fn(pos: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Verbatim copy of DonutNLL._smooth_loss for standalone testing."""
    vel    = pos[:, :, 1:, :] - pos[:, :, :-1, :]   # [N, K, T-1, 2]
    acc    = vel[:, :, 1:, :] - vel[:, :, :-1, :]   # [N, K, T-2, 2]
    acc_sq = (acc ** 2).sum(dim=-1)                  # [N, K, T-2]
    acc_mask = mask[:, 2:].float().unsqueeze(1)       # [N, 1, T-2]
    acc_sq   = acc_sq * acc_mask
    denom    = acc_mask.sum() * acc_sq.shape[1]
    return acc_sq.sum() / denom.clamp(min=1.0)


# ── helpers ──────────────────────────────────────────────────────────────────

def make_pos(N=3, K=6, T=80):
    return torch.randn(N, K, T, 2)

def make_mask(N=3, T=80):
    return torch.ones(N, T, dtype=torch.bool)


# ── unit tests ───────────────────────────────────────────────────────────────

def test_linear_trajectory_gives_zero_loss():
    """Constant-velocity trajectory → zero acceleration → loss == 0."""
    N, K, T = 2, 6, 80
    t   = torch.arange(T, dtype=torch.float32)
    vel = torch.tensor([1.5, 0.3])
    pos = (t[:, None] * vel[None, :])               # [T, 2]
    pos = pos[None, None].expand(N, K, T, 2).clone()

    loss = _smooth_loss_fn(pos, make_mask(N, T))
    assert loss.item() < 1e-8, f"Expected ~0, got {loss.item():.6e}"
    print(f"[PASS] linear trajectory (zero accel)          loss = {loss.item():.2e}")


def test_discontinuous_trajectory_gives_high_loss():
    """Sudden direction reversal → large acceleration → high loss."""
    N, K, T = 1, 1, 10
    pos = torch.zeros(N, K, T, 2)
    pos[0, 0, :5, 0] = torch.arange(5, dtype=torch.float32)
    pos[0, 0, 5:, 0] = 4 - torch.arange(5, dtype=torch.float32) * 3.0

    loss = _smooth_loss_fn(pos, make_mask(N, T))
    assert loss.item() > 1.0, f"Expected large loss, got {loss.item():.4f}"
    print(f"[PASS] discontinuous trajectory (high accel)   loss = {loss.item():.4f}")


def test_masking_excludes_invalid_timesteps():
    """Masking out the spike region must reduce the loss."""
    N, K, T = 1, 1, 10
    pos = torch.zeros(N, K, T, 2)
    pos[0, 0, 5:, 0] = 100.0   # huge spike

    full_mask    = torch.ones(N, T, dtype=torch.bool)
    partial_mask = torch.ones(N, T, dtype=torch.bool)
    partial_mask[0, 5:] = False

    loss_full    = _smooth_loss_fn(pos, full_mask)
    loss_partial = _smooth_loss_fn(pos, partial_mask)

    assert loss_partial.item() < loss_full.item(), (
        f"Partial mask should give lower loss: {loss_partial.item():.4f} vs {loss_full.item():.4f}"
    )
    print(f"[PASS] masking                                  full={loss_full.item():.4f}  partial={loss_partial.item():.4f}")


def test_all_masked_no_nan():
    """All-False mask → denom clamped → no NaN / Inf, loss == 0."""
    pos  = make_pos() * 100.0
    mask = torch.zeros(3, 80, dtype=torch.bool)
    loss = _smooth_loss_fn(pos, mask)
    assert not torch.isnan(loss) and not torch.isinf(loss), "NaN/Inf with all-False mask"
    assert loss.item() == 0.0, f"Expected 0.0, got {loss.item()}"
    print(f"[PASS] all-masked no NaN                        loss = {loss.item()}")


def test_output_is_scalar():
    """Return value must be a 0-dim tensor."""
    loss = _smooth_loss_fn(make_pos(), make_mask())
    assert loss.shape == torch.Size([]), f"Expected scalar, got {loss.shape}"
    print(f"[PASS] output is scalar                         shape = {loss.shape}")


def test_smoother_trajectory_lower_loss():
    """Smooth sine wave must have lower loss than the same + Gaussian noise."""
    N, K, T = 1, 1, 80
    t      = torch.linspace(0, 4 * 3.14159, T)
    smooth = torch.zeros(N, K, T, 2)
    smooth[0, 0, :, 0] = torch.sin(t) * 5.0
    smooth[0, 0, :, 1] = torch.cos(t) * 5.0
    noisy  = smooth.clone() + torch.randn_like(smooth) * 2.0

    mask = make_mask(N, T)
    l_smooth = _smooth_loss_fn(smooth, mask)
    l_noisy  = _smooth_loss_fn(noisy,  mask)
    assert l_smooth.item() < l_noisy.item(), (
        f"Smooth < noisy violated: {l_smooth.item():.4f} vs {l_noisy.item():.4f}"
    )
    print(f"[PASS] smoother → lower loss                    smooth={l_smooth.item():.4f}  noisy={l_noisy.item():.4f}")


def run_unit_tests():
    print("─" * 60)
    print("SECTION A: Unit tests (no project imports needed)")
    print("─" * 60)
    test_linear_trajectory_gives_zero_loss()
    test_discontinuous_trajectory_gives_high_loss()
    test_masking_excludes_invalid_timesteps()
    test_all_masked_no_nan()
    test_output_is_scalar()
    test_smoother_trajectory_lower_loss()
    print()


# ╔══════════════════════════════════════════════════════════════════╗
# ║  SECTION B — integration tests (requires full project env)      ║
# ╚══════════════════════════════════════════════════════════════════╝

def run_integration_tests():
    print("─" * 60)
    print("SECTION B: Integration tests (requires pytorch_lightning etc.)")
    print("─" * 60)

    try:
        from predictors.donut_nll import DonutNLL
        from distributions import DistributionFactory
    except ImportError as e:
        print(f"[SKIP] Could not import project modules: {e}")
        print("       Run this section on Compute Canada where the env is set up.")
        print()
        return

    # ── minimal model factory ─────────────────────────────────────────────────

    def make_model(lambda_smooth=0.0):
        return DonutNLL(
            dataset='waymo', t_per_tok=10, t_hist=11, t_pred=20,
            num_modes=6, refine=False, overpredict=False, hidden_dim=32,
            edge_limit=0.9999, map_enc_radius=50, map_enc_layers=1,
            dec_attn_order='t', dec_attn_repetitions=1,
            dec_radius_r=50, dec_radius_s=50,
            position_distribution='gaussian', loss_type='traj_nll',
            target_loss_only=False, lr=5e-4, weight_decay=1e-4,
            decay_epochs=32, lambda_smooth=lambda_smooth,
        )

    def make_traj_distr(model, factory, logits_3d=False):
        N, K, T = 4, model.num_modes, model.t_pred
        logits = torch.randn(N, K, T) if logits_3d else torch.randn(N, K)
        return factory.build(
            pos=torch.randn(N, K, T, 2), pos_params=torch.randn(N, K, T, 2),
            head=torch.randn(N, K, T),   head_params=torch.randn(N, K, T),
            logits=logits, cumsum_uncertainty=False,
        )

    def make_batch(model):
        N, T = 4, model.t_pred
        return {
            'agent': {
                'position':     torch.randn(N, model.t_hist + T, 3),
                'heading':      torch.randn(N, model.t_hist + T),
                'predict_mask': torch.ones(N, model.t_hist + T, dtype=torch.bool),
                'category':     torch.ones(N, dtype=torch.long),
            }
        }

    factory = DistributionFactory(pos_family='gaussian')

    # B-1: lambda=0 → no 'smooth' key (backward compat)
    m   = make_model(0.0)
    td  = [[make_traj_distr(m, factory)]]
    out = m.compute_loss(make_batch(m), td)
    assert 'loss'   in out,         "Missing 'loss' key"
    assert 'smooth' not in out,     "Unexpected 'smooth' key when lambda=0"
    assert not torch.isnan(out['loss']), "NaN loss"
    print(f"[PASS] lambda=0 backward compat                 loss={out['loss'].item():.4f}")

    # B-2: lambda>0 → 'smooth' key present and non-negative
    m2   = make_model(1.0)
    td2  = [[make_traj_distr(m2, factory)]]
    out2 = m2.compute_loss(make_batch(m2), td2)
    assert 'smooth' in out2,                  "Missing 'smooth' key when lambda>0"
    assert out2['smooth'].item() >= 0.0,      "Smooth loss must be non-negative"
    assert not torch.isnan(out2['loss']),     "NaN loss with lambda>0"
    print(f"[PASS] lambda=1.0                               loss={out2['loss'].item():.4f}  smooth={out2['smooth'].item():.4f}")

    # B-3: all three loss_type variants
    for lt in ('traj_nll', 'step_nll', 'wta'):
        m3 = make_model(0.05)
        m3.loss_type = lt
        use_3d = (lt == 'step_nll')
        td3 = [[make_traj_distr(m3, factory, logits_3d=use_3d)]]
        out3 = m3.compute_loss(make_batch(m3), td3)
        assert not torch.isnan(out3['loss']), f"NaN loss for loss_type={lt}"
        print(f"[PASS] loss_type={lt:<10}                  loss={out3['loss'].item():.4f}")

    # B-4: _smooth_loss on model matches inline fn on same tensor
    pos_t = torch.randn(4, 6, 20, 2)
    msk_t = torch.ones(4, 20, dtype=torch.bool)
    from predictors.donut_nll import DonutNLL as _DN
    l_model  = _DN._smooth_loss(pos_t, msk_t)
    l_inline = _smooth_loss_fn(pos_t, msk_t)
    assert abs(l_model.item() - l_inline.item()) < 1e-6, (
        f"Model and inline implementations differ: {l_model.item()} vs {l_inline.item()}"
    )
    print(f"[PASS] model._smooth_loss matches inline fn     Δ={abs(l_model.item()-l_inline.item()):.2e}")

    print()


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    run_unit = '--integration' not in sys.argv
    run_intg = '--unit'        not in sys.argv

    if run_unit:
        run_unit_tests()
    if run_intg:
        run_integration_tests()

    print("✓ Done.")
