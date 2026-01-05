import torch
import torch.nn.functional as F

def orthogonal_loss(
    h_s: torch.Tensor,
    h_p: torch.Tensor,
    mode: str = "corr",
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Orthogonality loss between shared/private features.

    Args:
        h_s: [B, Ds] shared features
        h_p: [B, Dp] private features
        mode:
          - "corr": penalize squared cross-correlation (distribution-level)
          - "cos":  penalize mean squared cosine similarity (sample-level, requires Ds==Dp)
        eps: numerical stability constant
    """
    if h_s.dim() != 2 or h_p.dim() != 2:
        raise ValueError(f"Expected 2D tensors [B, D], got {h_s.shape} and {h_p.shape}")
    if h_s.size(0) != h_p.size(0):
        raise ValueError(f"Batch mismatch: {h_s.size(0)} vs {h_p.size(0)}")

    if mode == "cos":
        if h_s.size(1) != h_p.size(1):
            raise ValueError(f"'cos' mode requires same feature dim, got {h_s.size(1)} vs {h_p.size(1)}")
        h_s_norm = F.normalize(h_s, p=2, dim=1, eps=eps)
        h_p_norm = F.normalize(h_p, p=2, dim=1, eps=eps)
        cos_sim = torch.sum(h_s_norm * h_p_norm, dim=1)  # [B]
        return (cos_sim ** 2).mean()

    if mode == "corr":
        hs = h_s - h_s.mean(dim=0, keepdim=True)
        hp = h_p - h_p.mean(dim=0, keepdim=True)

        # normalize each feature dimension (column) to unit norm
        hs = hs / (hs.norm(p=2, dim=0, keepdim=True) + eps)
        hp = hp / (hp.norm(p=2, dim=0, keepdim=True) + eps)

        # cross-correlation: [Ds, Dp]
        corr = hs.t() @ hp

        # scale-invariant objective
        return (corr ** 2).mean()

    raise ValueError(f"Unknown mode: {mode}. Expected 'corr' or 'cos'.")
