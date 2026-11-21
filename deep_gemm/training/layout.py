"""Checkpoint conversions for gate/up training-parameter layouts."""


def to_g8_weight(weight):
    """Convert [..., 2*I, H] split gate/up rows to [gate8, up8, ...]."""
    if weight.ndim < 2 or weight.shape[-2] % 16:
        raise ValueError("G8 weights require a 2*I row dimension divisible by16")
    shape = tuple(weight.shape)
    return weight.reshape(*shape[:-2], 2, shape[-2] // 16, 8, shape[-1]).transpose(-4, -3).contiguous().view(shape)


def from_g8_weight(weight):
    """Convert G8 weights or their matching gradients to split checkpoint rows."""
    if weight.ndim < 2 or weight.shape[-2] % 16:
        raise ValueError("G8 weights require a 2*I row dimension divisible by16")
    shape = tuple(weight.shape)
    return weight.reshape(*shape[:-2], shape[-2] // 16, 2, 8, shape[-1]).transpose(-4, -3).contiguous().view(shape)
