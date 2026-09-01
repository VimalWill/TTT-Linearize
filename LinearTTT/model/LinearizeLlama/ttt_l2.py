# -*- coding: utf-8 -*-
"""LaCT operator with an l2-regression inner objective and a retention gate.

Upstream's `block_causal_lact_swiglu` descends the dot-product bias

    L(f_W(k), v) = -f_W(k)^T v                                  (LaCT Eq. 7)

and controls the fast-weight magnitude by projecting every row back onto its
*initial* norm after each update (LaCT Eq. 8). Those two choices belong
together: a direction-only objective is unbounded, so a fixed-norm sphere is
exactly the right constraint, and the output scale is set downstream by
`ttt_norm` + `ttt_scale_proj` anyway.

This module implements the other coherent pairing, from Titans / Atlas
(Behrouz et al., arXiv:2505.23735). The objective becomes l2 regression

    L(f_W(k), v) = 1/2 * || f_W(k) - v ||^2                      (Atlas Eq. 9)

whose gradient carries the residual (v - f_W(k)), so a key that already reads
out correctly produces no update -- error correction rather than Hebbian
accumulation. Crucially, the magnitude control changes with it: Atlas Eq. 32
uses a *learned retention gate* rather than a norm projection,

    M_t = alpha_t * M_{t-1} - eta_t * NewtonSchulz(S_t)
    S_t = theta_t * S_{t-1} + grad(...)

with alpha_t in (0, 1) from a per-token projection. Decay bounds the memory
multiplicatively instead of pinning it to a sphere, so ||W|| is free to grow
until f_W(k) can actually reach v.

Pairing l2 with Eq. 8 instead -- which is what a naive `dot -> l2` swap gives
you -- does not work: the rows are pinned at ||W_0||, the reachable readout
magnitude sits ~14x below ||v|| on a trained checkpoint, and the update spends
itself inflating ||pred|| without aligning it. Measured, the inner loss climbs
*above* 1.0 (worse than predicting zero) while training diverges. The l2 update
is self-limiting once the residual closes, so it does not need the sphere.

Kept identical to upstream: the SwiGLU fast-weight function, the
apply-then-update order, per-token/per-matrix learning rates, momentum, and the
optional Muon step.
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

    gate_before_act = torch.bmm(w0, kT)                      # [b, dh, l]
    hidden_before_mul = torch.bmm(w2, kT)                    # [b, dh, l]
    gate = F.silu(gate_before_act, inplace=False)
    hidden = gate * hidden_before_mul                        # [b, dh, l]

    # The term the dot-product bias does not have: what the memory currently
    # retrieves for these keys.
    #
    # In fp32, deliberately. This is the only subtraction in either operator,
    # and training drives pred -> v by construction, so err is a difference of
    # two nearly-equal numbers precisely when the memory is working. In bf16
    # (8 mantissa bits) its relative error grows without bound as that happens.
    # One fp32 bmm per chunk is a rounding error against the operator's 18 units.
    # Only the subtraction needs the extra precision; err is then cast back to
    # vi's dtype and used exactly where upstream uses vi, so the rest of the
    # pass keeps upstream's single-dtype structure rather than threading a
    # second dtype through the bmms.
    acc = torch.promote_types(torch.float32,
                              torch.promote_types(w1.dtype, vi.dtype))
    with torch.autocast(device_type=vi.device.type, enabled=False):
        pred = torch.bmm(w1.to(acc), hidden.to(acc))         # [b, dv, l]
        err = (vi.to(acc) - pred).to(vi.dtype)               # [b, dv, l]

    # upstream's backward pass, seeded with the residual instead of v
    dhidden = torch.bmm(w1.transpose(1, 2), err)
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
    momentum: torch.Tensor = None,    # [b, s, 1]
    retention: torch.Tensor = None,   # [b, s, 1], alpha in (0, 1)
):
    """Drop-in replacement for `block_causal_lact_swiglu` with the l2 bias.

    Same signature plus `retention`, same apply-then-update (shifted block
    causal) order, same output shape [b, l, dv].

    `retention` replaces upstream's channel-wise renormalisation: instead of
    W <- L2Norm(W + dW) * ||W_0||, this does W <- alpha * W + dW. Passing
    retention=None falls back to no magnitude control at all, which is only
    sensible for debugging -- the l2 update is self-limiting but nothing then
    bounds the accumulated drift.
    """
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
            m_i = momentum[:, s_index:e_index, :].mean(dim=1, keepdim=True)

            dw0 = dw0 + dw0_momentum * m_i
            dw1 = dw1 + dw1_momentum * m_i
            dw2 = dw2 + dw2_momentum * m_i
            dw0_momentum = dw0
            dw1_momentum = dw1
            dw2_momentum = dw2

        if use_muon:
            # Atlas Eq. 32 applies eta_t OUTSIDE Newton-Schulz for a reason:
            # NS returns the nearest semi-orthogonal matrix, so it discards the
            # magnitude of its input -- including the per-token lr already folded
            # into dw. Upstream can ignore this because Eq. 8 rescales W right
            # afterwards (hence its "conclusion: 1.0 is good" note on the muon
            # lr), but with the retention gate an unscaled orthogonal update is
            # O(1) against ||W|| and blows up immediately. Reapply the lr as the
            # chunk-mean per head, which is the eta_t of Eq. 32.
            eta0 = lr0i.mean(dim=1, keepdim=True)
            eta1 = lr1i.mean(dim=1, keepdim=True)
            eta2 = lr2i.mean(dim=1, keepdim=True)
            dw0 = zeropower_via_newtonschulz5(dw0).type_as(dw0) * eta0
            dw1 = zeropower_via_newtonschulz5(dw1).type_as(dw1) * eta1
            dw2 = zeropower_via_newtonschulz5(dw2).type_as(dw2) * eta2

        # Atlas Eq. 32: multiplicative decay of the old memory, in place of
        # LaCT Eq. 8's projection back onto the initial row norms.
        if retention is not None:
            a_i = retention[:, s_index:e_index, :].mean(dim=1, keepdim=True)
            w0 = w0 * a_i + dw0
            w1 = w1 * a_i + dw1
            w2 = w2 * a_i + dw2
        else:
            w0 = w0 + dw0
            w1 = w1 + dw1
            w2 = w2 + dw2

    # tail chunk: read out with the final weights, no further update
    s_index = e_index
    e_index = seq_len
    qi = q[:, :, s_index:e_index]
    h = torch.bmm(w2, qi)
    gate = F.silu(torch.bmm(w0, qi), inplace=True)
    output[:, :, s_index:e_index] = torch.bmm(w1, gate * h)

    return output.transpose(1, 2)


__all__ = ['block_causal_lact_swiglu_l2', 'swiglu_l2_grads']
