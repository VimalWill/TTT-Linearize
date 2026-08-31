# -*- coding: utf-8 -*-
"""LaCT operator with an l2-regression inner objective.

Upstream's `block_causal_lact_swiglu` descends the dot-product bias

    L(f_W(k), v) = -f_W(k)^T v                          (LaCT Eq. 7)

whose gradient never looks at what the memory currently retrieves, so every
association is written unconditionally -- purely Hebbian accumulation, held
finite only by the weight renormalisation. This variant descends

    L(f_W(k), v) = 1/2 * || f_W(k) - v ||^2

the l2 regression bias used by TTT-Linear / Titans / DeltaNet. Its gradient
carries the residual (v - f_W(k)), so a key that already reads out correctly
produces no update: error correction rather than accumulation.

Structurally the change is a single substitution -- everywhere upstream seeds
the inner backward pass with `v`, seed it with `v - f_W(k)` instead. The cost is
one extra bmm per chunk (the readout at k, which the dot-product form is able to
skip; cf. the FLOPs count in the paper's Appendix A, 18 -> 20 units, ~11%).

Kept identical to upstream: the SwiGLU fast-weight function, the
apply-then-update order, per-token/per-matrix learning rates, momentum,
optional Muon, and the channel-wise weight renormalisation (Eq. 8).
"""

import torch
import torch.nn.functional as F

from .ttt_ops import silu_backprop, zeropower_via_newtonschulz5


def swiglu_l2_grads(w0, w1, w2, ki, vi, lr0i, lr1i, lr2i):
    """Descent directions for one chunk under the l2 regression bias.

    Args:
        w0, w1, w2: fast weights. w0, w2: [b, dh, dk]; w1: [b, dv, dh]
        ki: keys,   [b, l, dk]
        vi: values, [b, dv, l]  (transposed, as in the operator's loop)
        lr0i, lr1i, lr2i: per-token learning rates, [b, l, 1] or [b, l, d]

    Returns:
        dw0, dw1, dw2 -- to be *added* to the fast weights, i.e. already
        negated relative to the gradient of L.
    """
    kT = ki.transpose(1, 2)                                  # [b, dk, l]

    # forward at the keys
    gate_before_act = torch.bmm(w0, kT)                      # [b, dh, l]
    hidden_before_mul = torch.bmm(w2, kT)                    # [b, dh, l]
    gate = F.silu(gate_before_act, inplace=False)
    hidden = gate * hidden_before_mul                        # [b, dh, l]

    # The extra term relative to the dot-product bias: what the memory
    # currently retrieves for these keys. err is the descent direction of
    # 1/2||pred - v||^2 wrt pred, negated.
    #
    # In fp32, deliberately. This is the only subtraction in either operator,
    # and stage 1 drives pred -> v by construction, so err is a difference of
    # two nearly-equal numbers precisely when training is working. In bf16
    # (8 mantissa bits) its relative error grows without bound as the model
    # improves -- the residual turns to noise and the run diverges. One fp32
    # bmm per chunk is a rounding error against the operator's 18 units.
    acc = torch.promote_types(torch.float32,
                              torch.promote_types(w1.dtype, vi.dtype))
    with torch.autocast(device_type=vi.device.type, enabled=False):
        pred = torch.bmm(w1.to(acc), hidden.to(acc))         # [b, dv, l]
        err = vi.to(acc) - pred                              # [b, dv, l]

    # from here on: upstream's backward pass with `vi` replaced by `err`
    dhidden = torch.bmm(w1.transpose(1, 2), err)             # [b, dh, l]
    dhidden_before_mul = dhidden * gate
    dgate_before_act = silu_backprop(dhidden * hidden_before_mul, gate_before_act)

    dw1 = torch.bmm(err, (hidden.transpose(1, 2) * lr1i).type_as(err))
    dw0 = torch.bmm(dgate_before_act, (ki * lr0i).type_as(dgate_before_act))
    dw2 = torch.bmm(dhidden_before_mul, (ki * lr2i).type_as(dhidden_before_mul))
    return dw0, dw1, dw2


@torch.compile()
@torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16)
def block_causal_lact_swiglu_l2(
    w0: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lr0: torch.Tensor,
    lr1: torch.Tensor,
    lr2: torch.Tensor,
    chunk_size: int = 2048,
    use_muon: bool = False,
    momentum: torch.Tensor = None,   # [b, s, 1]
):
    """Drop-in replacement for `block_causal_lact_swiglu` with the l2 bias.

    Same signature, same apply-then-update (shifted block causal) order, same
    outputs: o of shape [b, l, dv].
    """
    w0_norm = w0.norm(dim=2, keepdim=True)
    w1_norm = w1.norm(dim=2, keepdim=True)
    w2_norm = w2.norm(dim=2, keepdim=True)

    if momentum is not None:
        dw0_momentum = torch.zeros_like(w0)
        dw1_momentum = torch.zeros_like(w1)
        dw2_momentum = torch.zeros_like(w2)

    q = q.transpose(1, 2)   # [b, dk, l]
    v = v.transpose(1, 2)   # [b, dv, l]

    output = torch.zeros_like(v)

    e_index = 0
    seq_len = k.shape[1]
    for i in range(0, seq_len - chunk_size, chunk_size):
        s_index = i
        e_index = s_index + chunk_size

        ki = k[:, s_index:e_index, :]
        vi = v[:, :, s_index:e_index]
        qi = q[:, :, s_index:e_index]
        lr0i = lr0[:, s_index:e_index, :]
        lr1i = lr1[:, s_index:e_index, :]
        lr2i = lr2[:, s_index:e_index, :]

        # apply first: weights fit on chunks strictly before this one
        h = torch.bmm(w2, qi)
        gate = F.silu(torch.bmm(w0, qi), inplace=True)
        output[:, :, s_index:e_index] = torch.bmm(w1, gate * h)

        dw0, dw1, dw2 = swiglu_l2_grads(w0, w1, w2, ki, vi, lr0i, lr1i, lr2i)

        if momentum is not None:
            m_i = momentum[:, s_index:e_index, :]
            m_i = m_i.mean(dim=1, keepdim=True)

            dw0 = dw0 + dw0_momentum * m_i
            dw1 = dw1 + dw1_momentum * m_i
            dw2 = dw2 + dw2_momentum * m_i
            dw0_momentum = dw0
            dw1_momentum = dw1
            dw2_momentum = dw2

        if use_muon:
            dw0 = zeropower_via_newtonschulz5(dw0)
            dw1 = zeropower_via_newtonschulz5(dw1)
            dw2 = zeropower_via_newtonschulz5(dw2)

        w0 = w0 + dw0
        w1 = w1 + dw1
        w2 = w2 + dw2

        # channel-wise l2 norm, conceptually like post-norm (Eq. 8)
        w0 = w0 / (w0.norm(dim=2, keepdim=True) + 1e-5) * w0_norm
        w1 = w1 / (w1.norm(dim=2, keepdim=True) + 1e-5) * w1_norm
        w2 = w2 / (w2.norm(dim=2, keepdim=True) + 1e-5) * w2_norm

    # tail chunk: read out with the final weights, no further update
    s_index = e_index
    e_index = seq_len
    qi = q[:, :, s_index:e_index]
    h = torch.bmm(w2, qi)
    gate = F.silu(torch.bmm(w0, qi), inplace=True)
    output[:, :, s_index:e_index] = torch.bmm(w1, gate * h)

    return output.transpose(1, 2)


__all__ = ['block_causal_lact_swiglu_l2', 'swiglu_l2_grads']
