from typing import Tuple

from . import operators
from .autodiff import Context
from .fast_ops import FastOps
from .tensor import Tensor
from .tensor_functions import Function, rand


# List of functions in this file:
# - avgpool2d: Tiled average pooling 2D
# - argmax: Compute the argmax as a 1-hot tensor
# - Max: New Function for max operator
# - max: Apply max reduction
# - softmax: Compute the softmax as a tensor
# - logsoftmax: Compute the log of the softmax as a tensor - See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
# - maxpool2d: Tiled max pooling 2D
# - dropout: Dropout positions based on random noise, include an argument to turn off


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Reshape an image tensor for 2D pooling

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.

    """
    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0
    # TODO: Implement for Task 4.3.
    new_width = width // kw
    new_height = height // kh

    reshaped = input.contiguous().view(batch, channel, height, new_width, kw)
    tiled = (
        reshaped.permute(0, 1, 3, 2, 4)
        .contiguous()
        .view(batch, channel, new_height, new_width, kh * kw)
    )

    return tiled, new_height, new_width


# TODO: Implement for Task 4.3.
def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Average pooling operation.

    Args:
    ----
        input: Tensor of size batch x channel x height x width.
        kernel: Tuple of (kernel_height, kernel_width).

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width after average pooling.

    """
    batch, channel, _, _ = input.shape
    tiled, new_height, new_width = tile(input, kernel)
    return tiled.mean(dim=4).view(batch, channel, new_height, new_width)


max_reduce = FastOps.reduce(operators.max, -1e9)


def argmax(input: Tensor, dim: int) -> Tensor:
    """Compute the argmax as a 1-hot tensor."""
    out = max_reduce(input, dim)
    return out == input


class Max(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, dim: Tensor) -> Tensor:
        """Forward of max is max reduction"""
        ctx.save_for_backward(input, dim)
        return max_reduce(input, int(dim.item()))

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Backward of max is argmax"""
        input, dim = ctx.saved_values
        out = argmax(input, int(dim.item())) * grad_output
        return (out, dim)


def max(input: Tensor, dim: int) -> Tensor:
    """Compute the maximum value along a dimension."""
    return Max.apply(input, input._ensure_tensor(dim))


def softmax(input: Tensor, dim: int) -> Tensor:
    """Compute the softmax function along a specified dimension."""
    exp_input = input.exp()
    sum_exp = exp_input.sum(dim=dim)
    return exp_input / sum_exp


def logsoftmax(input: Tensor, dim: int) -> Tensor:
    """Compute the log of the softmax function."""
    max_input = max(input, dim)
    shifted_input = input - max_input
    log_sum_exp = shifted_input.exp().sum(dim=dim).log()
    return shifted_input - log_sum_exp


def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Apply 2D max pooling to the input tensor."""
    batch, channel, _, _ = input.shape
    tiled, new_height, new_width = tile(input, kernel)
    return max(tiled, dim=4).view(batch, channel, new_height, new_width)


def dropout(input: Tensor, rate: float, ignore: bool = False) -> Tensor:
    """Apply dropout to the input tensor."""
    if ignore:
        return input
    return input * (rand(input.shape) > rate)
