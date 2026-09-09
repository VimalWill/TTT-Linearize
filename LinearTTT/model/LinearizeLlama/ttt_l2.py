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

    # What the memory currently retrieves for these keys -- the term the
    # dot-product bias does not have.
    #
    # fp32 deliberately: training drives pred -> v, so err is a difference of
    # nearly-equal numbers exactly when the memory works, and bf16's 8 mantissa
    # bits lose it. Only the subtraction needs the precision; err is cast back
    # to vi's dtype and used where upstream uses vi.
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
    return_state: bool = False,
    init_momentum: tuple = None,
    return_momentum: bool = False,
):
    """Drop-in replacement for `block_causal_lact_swiglu` with the l2 bias.

    Same signature plus `retention`, same apply-then-update (shifted block
    causal) order, same output shape [b, l, dv].

    `return_state` additionally returns the converged (w0, w1, w2), which a
    shared memory and the decode path both need; upstream returns only output.

    `retention` replaces upstream's renormalisation: W <- alpha * W + dW rather
    than W <- L2Norm(W + dW) * ||W_0||. retention=None means no magnitude
    control at all -- debugging only.
    """
    if momentum is not None:
        # Momentum buffers persist ACROSS chunks, so an incremental caller
        # resuming mid-sequence must hand them back in or the state drifts from
        # what a single pass produces.
        if init_momentum is not None:
            dw0_momentum, dw1_momentum, dw2_momentum = init_momentum
        else:
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
            # Atlas Eq. 32 applies eta_t OUTSIDE Newton-Schulz: NS returns the
            # nearest semi-orthogonal matrix and so discards its input's
            # magnitude, including the per-token lr folded into dw. Upstream
            # gets away with it because Eq. 8 rescales W afterwards; with the
            # retention gate an unscaled orthogonal update is O(1) against ||W||
            # and blows up. Reapply the lr as the per-head chunk mean.
            eta0 = lr0i.mean(dim=1, keepdim=True)
            eta1 = lr1i.mean(dim=1, keepdim=True)
            eta2 = lr2i.mean(dim=1, keepdim=True)
            dw0 = zeropower_via_newtonschulz5(dw0).type_as(dw0) * eta0
            dw1 = zeropower_via_newtonschulz5(dw1).type_as(dw1) * eta1
            dw2 = zeropower_via_newtonschulz5(dw2).type_as(dw2) * eta2

        # Atlas Eq. 32: multiplicative decay in place of Eq. 8's projection.
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

    out = output.transpose(1, 2)
    if return_momentum:
        mom = ((dw0_momentum, dw1_momentum, dw2_momentum)
               if momentum is not None else None)
        return out, w0, w1, w2, mom
    return (out, w0, w1, w2) if return_state else out


__all__ = ['block_causal_lact_swiglu_l2', 'swiglu_l2_grads']
