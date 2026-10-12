"""CORAL loss（论文 Eq.1-3）。

L_CORAL = 1/(4 d^2) * || C_S - C_T ||_F^2
C = 1/(n-1) * (D^T D - 1/n (1^T D)^T (1^T D))    —— 即无偏的 batch 协方差
梯度（Eq.4/5）由 autograd 自动得到，与论文推导一致。
"""
import torch


def covariance(x: torch.Tensor) -> torch.Tensor:
    """x: [n, d] -> [d, d]，按论文 Eq.(2)/(3) 计算。"""
    n = x.size(0)
    ones_t_x = x.sum(dim=0, keepdim=True)          # 1^T D, [1, d]
    return (x.t() @ x - ones_t_x.t() @ ones_t_x / n) / (n - 1)


def coral_loss(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    d = source.size(1)
    diff = covariance(source) - covariance(target)
    return (diff * diff).sum() / (4 * d * d)


if __name__ == "__main__":
    # 自检：与 torch.cov 结果一致，且论文 Eq.(4) 的解析梯度与 autograd 一致
    torch.manual_seed(0)
    s = torch.randn(64, 31, dtype=torch.float64, requires_grad=True)
    t = torch.randn(48, 31, dtype=torch.float64) * 2 + 1
    assert torch.allclose(covariance(s), torch.cov(s.t()))
    loss = coral_loss(s, t)
    loss.backward()
    n, d = s.shape
    cs, ct = covariance(s.detach()), covariance(t)
    centered = s.detach() - s.detach().mean(0, keepdim=True)
    grad_eq4 = centered @ (cs - ct) / (d * d * (n - 1))
    assert torch.allclose(s.grad, grad_eq4)
    print(f"coral_loss={loss.item():.6f}  自检通过（协方差与 Eq.4 梯度均正确）")
