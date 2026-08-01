from __future__ import annotations

import math
import os
import tempfile
import threading
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from .renderer import PARAM_COUNT, SoftShapeRenderer


class CudaRendererUnavailable(RuntimeError):
    """Raised when the optional CuPy/NVRTC renderer cannot be used."""


_CUDA_SOURCE = r"""
#define CHECKPOINT_SIZE 16
#define PARAM_GRADS 10

__device__ __forceinline__ float stable_sigmoid(float value) {
    if (value >= 0.0f) {
        return 1.0f / (1.0f + expf(-value));
    }
    const float exponential = expf(value);
    return exponential / (1.0f + exponential);
}

__device__ __forceinline__ float primitive_distance(
    int shape_type,
    float local_x,
    float local_y,
    float half_x,
    float half_y,
    float pixel_scale
) {
    half_x = fmaxf(half_x, 1.0e-5f);
    half_y = fmaxf(half_y, 1.0e-5f);

    if (shape_type == 0) {
        const float normalized_x = local_x / half_x;
        const float normalized_y = local_y / half_y;
        const float radius = sqrtf(
            normalized_x * normalized_x + normalized_y * normalized_y + 1.0e-8f
        );
        return (1.0f - radius) * fminf(half_x, half_y) * pixel_scale;
    }

    if (shape_type == 1) {
        const float q_x = fabsf(local_x) - half_x;
        const float q_y = fabsf(local_y) - half_y;
        const float outside_x = fmaxf(q_x, 0.0f);
        const float outside_y = fmaxf(q_y, 0.0f);
        const float outside = sqrtf(
            outside_x * outside_x + outside_y * outside_y + 1.0e-8f
        );
        const float inside = fminf(fmaxf(q_x, q_y), 0.0f);
        return -(outside + inside) * pixel_scale;
    }

    const float normalized_x = local_x / half_x;
    const float normalized_y = local_y / half_y;
    const float inverse_sqrt_five = 0.4472135954999579f;
    const float edge_zero = (2.0f * normalized_x + normalized_y + 1.0f) * inverse_sqrt_five;
    const float edge_one = 1.0f - normalized_y;
    const float edge_two = (-2.0f * normalized_x + normalized_y + 1.0f) * inverse_sqrt_five;
    return fminf(edge_zero, fminf(edge_one, edge_two))
        * fminf(half_x, half_y) * pixel_scale;
}

__device__ __forceinline__ float shape_mask(
    int shape_index,
    float point_x,
    float point_y,
    const float* centers,
    const float* half_sizes,
    const float* sine_angles,
    const float* cosine_angles,
    const int* shape_types,
    float pixel_scale,
    float softness
) {
    const float delta_x = point_x - centers[shape_index * 2];
    const float delta_y = point_y - centers[shape_index * 2 + 1];
    const float sine = sine_angles[shape_index];
    const float cosine = cosine_angles[shape_index];
    const float local_x = cosine * delta_x + sine * delta_y;
    const float local_y = -sine * delta_x + cosine * delta_y;
    const float distance = primitive_distance(
        shape_types[shape_index],
        local_x,
        local_y,
        half_sizes[shape_index * 2],
        half_sizes[shape_index * 2 + 1],
        pixel_scale
    );
    return stable_sigmoid(distance / softness);
}

__device__ __forceinline__ void distance_partials(
    int shape_type,
    float local_x,
    float local_y,
    float half_x,
    float half_y,
    float pixel_scale,
    float* distance,
    float* derivative_local_x,
    float* derivative_local_y,
    float* derivative_half_x,
    float* derivative_half_y
) {
    half_x = fmaxf(half_x, 1.0e-5f);
    half_y = fmaxf(half_y, 1.0e-5f);
    const bool x_is_minimum = half_x <= half_y;
    const float minimum_half = x_is_minimum ? half_x : half_y;
    const float minimum_x_gradient = x_is_minimum ? 1.0f : 0.0f;
    const float minimum_y_gradient = x_is_minimum ? 0.0f : 1.0f;

    if (shape_type == 0) {
        const float normalized_x = local_x / half_x;
        const float normalized_y = local_y / half_y;
        const float radius = sqrtf(
            normalized_x * normalized_x + normalized_y * normalized_y + 1.0e-8f
        );
        const float inverse_radius = 1.0f / radius;
        const float base = 1.0f - radius;
        *distance = base * minimum_half * pixel_scale;
        *derivative_local_x = -minimum_half * pixel_scale
            * local_x / (half_x * half_x) * inverse_radius;
        *derivative_local_y = -minimum_half * pixel_scale
            * local_y / (half_y * half_y) * inverse_radius;
        *derivative_half_x = pixel_scale * (
            minimum_half * local_x * local_x
                / (half_x * half_x * half_x) * inverse_radius
            + base * minimum_x_gradient
        );
        *derivative_half_y = pixel_scale * (
            minimum_half * local_y * local_y
                / (half_y * half_y * half_y) * inverse_radius
            + base * minimum_y_gradient
        );
        return;
    }

    if (shape_type == 1) {
        const float q_x = fabsf(local_x) - half_x;
        const float q_y = fabsf(local_y) - half_y;
        const float outside_x = fmaxf(q_x, 0.0f);
        const float outside_y = fmaxf(q_y, 0.0f);
        const float outside = sqrtf(
            outside_x * outside_x + outside_y * outside_y + 1.0e-8f
        );
        const float maximum_q = fmaxf(q_x, q_y);
        const float inside = fminf(maximum_q, 0.0f);
        float outside_q_x = q_x > 0.0f ? outside_x / outside : 0.0f;
        float outside_q_y = q_y > 0.0f ? outside_y / outside : 0.0f;
        float inside_q_x = 0.0f;
        float inside_q_y = 0.0f;
        if (maximum_q < 0.0f) {
            if (q_x > q_y) {
                inside_q_x = 1.0f;
            } else if (q_y > q_x) {
                inside_q_y = 1.0f;
            } else {
                inside_q_x = 0.5f;
                inside_q_y = 0.5f;
            }
        } else if (maximum_q == 0.0f) {
            if (q_x > q_y) {
                inside_q_x = 0.5f;
            } else if (q_y > q_x) {
                inside_q_y = 0.5f;
            } else {
                inside_q_x = 0.25f;
                inside_q_y = 0.25f;
            }
        }
        const float distance_q_x = -(outside_q_x + inside_q_x) * pixel_scale;
        const float distance_q_y = -(outside_q_y + inside_q_y) * pixel_scale;
        *distance = -(outside + inside) * pixel_scale;
        *derivative_local_x = distance_q_x
            * (local_x > 0.0f ? 1.0f : (local_x < 0.0f ? -1.0f : 0.0f));
        *derivative_local_y = distance_q_y
            * (local_y > 0.0f ? 1.0f : (local_y < 0.0f ? -1.0f : 0.0f));
        *derivative_half_x = -distance_q_x;
        *derivative_half_y = -distance_q_y;
        return;
    }

    const float normalized_x = local_x / half_x;
    const float normalized_y = local_y / half_y;
    const float inverse_sqrt_five = 0.4472135954999579f;
    const float edge_zero = (2.0f * normalized_x + normalized_y + 1.0f) * inverse_sqrt_five;
    const float edge_one = 1.0f - normalized_y;
    const float edge_two = (-2.0f * normalized_x + normalized_y + 1.0f) * inverse_sqrt_five;
    float edge_distance = edge_zero;
    float edge_x_gradient = 2.0f * inverse_sqrt_five;
    float edge_y_gradient = inverse_sqrt_five;
    if (edge_one < edge_distance) {
        edge_distance = edge_one;
        edge_x_gradient = 0.0f;
        edge_y_gradient = -1.0f;
    }
    if (edge_two < edge_distance) {
        edge_distance = edge_two;
        edge_x_gradient = -2.0f * inverse_sqrt_five;
        edge_y_gradient = inverse_sqrt_five;
    }
    *distance = edge_distance * minimum_half * pixel_scale;
    *derivative_local_x = minimum_half * pixel_scale * edge_x_gradient / half_x;
    *derivative_local_y = minimum_half * pixel_scale * edge_y_gradient / half_y;
    *derivative_half_x = pixel_scale * (
        minimum_half * edge_x_gradient * (-normalized_x / half_x)
        + edge_distance * minimum_x_gradient
    );
    *derivative_half_y = pixel_scale * (
        minimum_half * edge_y_gradient * (-normalized_y / half_y)
        + edge_distance * minimum_y_gradient
    );
}

extern "C" __global__ void sampled_forward(
    const float* centers,
    const float* half_sizes,
    const float* sine_angles,
    const float* cosine_angles,
    const float* colors,
    const float* background,
    const float* presence,
    const int* shape_types,
    const float* points,
    int shape_count,
    int point_count,
    int checkpoint_count,
    float pixel_scale,
    float softness,
    float* output,
    float* checkpoints
) {
    const int point_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (point_index >= point_count) {
        return;
    }

    const float point_x = points[point_index * 2];
    const float point_y = points[point_index * 2 + 1];
    float canvas_red = background[0];
    float canvas_green = background[1];
    float canvas_blue = background[2];

    for (int shape_index = 0; shape_index < shape_count; ++shape_index) {
        if ((shape_index & (CHECKPOINT_SIZE - 1)) == 0) {
            const int checkpoint_index = shape_index / CHECKPOINT_SIZE;
            checkpoints[(checkpoint_index * 3) * point_count + point_index] = canvas_red;
            checkpoints[(checkpoint_index * 3 + 1) * point_count + point_index] = canvas_green;
            checkpoints[(checkpoint_index * 3 + 2) * point_count + point_index] = canvas_blue;
        }
        const float mask = shape_mask(
            shape_index,
            point_x,
            point_y,
            centers,
            half_sizes,
            sine_angles,
            cosine_angles,
            shape_types,
            pixel_scale,
            softness
        );
        const float alpha = mask * presence[shape_index];
        canvas_red += alpha * (colors[shape_index * 3] - canvas_red);
        canvas_green += alpha * (colors[shape_index * 3 + 1] - canvas_green);
        canvas_blue += alpha * (colors[shape_index * 3 + 2] - canvas_blue);
    }

    output[point_index] = canvas_red;
    output[point_count + point_index] = canvas_green;
    output[point_count * 2 + point_index] = canvas_blue;
}

extern "C" __global__ void sampled_backward(
    const float* centers,
    const float* half_sizes,
    const float* sine_angles,
    const float* cosine_angles,
    const float* colors,
    const float* presence,
    const int* shape_types,
    const float* points,
    const float* output_gradient,
    const float* checkpoints,
    int shape_count,
    int point_count,
    int checkpoint_count,
    float pixel_scale,
    float softness,
    float* center_gradient,
    float* half_size_gradient,
    float* sine_gradient,
    float* cosine_gradient,
    float* color_gradient,
    float* presence_gradient,
    float* background_gradient
) {
    __shared__ float shared_shape_gradient[CHECKPOINT_SIZE * PARAM_GRADS];
    const int local_thread = threadIdx.x;
    const int point_index = blockIdx.x * blockDim.x + local_thread;
    const bool point_is_active = point_index < point_count;
    const float point_x = point_is_active ? points[point_index * 2] : 0.0f;
    const float point_y = point_is_active ? points[point_index * 2 + 1] : 0.0f;
    float canvas_gradient_red = point_is_active ? output_gradient[point_index] : 0.0f;
    float canvas_gradient_green = point_is_active ? output_gradient[point_count + point_index] : 0.0f;
    float canvas_gradient_blue = point_is_active ? output_gradient[point_count * 2 + point_index] : 0.0f;

    for (int checkpoint_index = checkpoint_count - 1; checkpoint_index >= 0; --checkpoint_index) {
        for (int shared_index = local_thread;
             shared_index < CHECKPOINT_SIZE * PARAM_GRADS;
             shared_index += blockDim.x) {
            shared_shape_gradient[shared_index] = 0.0f;
        }
        __syncthreads();

        const int first_shape = checkpoint_index * CHECKPOINT_SIZE;
        const int shapes_in_checkpoint = min(CHECKPOINT_SIZE, shape_count - first_shape);
        if (point_is_active) {
            float canvas_before_red[CHECKPOINT_SIZE];
            float canvas_before_green[CHECKPOINT_SIZE];
            float canvas_before_blue[CHECKPOINT_SIZE];
            float canvas_red = checkpoints[(checkpoint_index * 3) * point_count + point_index];
            float canvas_green = checkpoints[(checkpoint_index * 3 + 1) * point_count + point_index];
            float canvas_blue = checkpoints[(checkpoint_index * 3 + 2) * point_count + point_index];

            for (int local_shape = 0; local_shape < shapes_in_checkpoint; ++local_shape) {
                const int shape_index = first_shape + local_shape;
                canvas_before_red[local_shape] = canvas_red;
                canvas_before_green[local_shape] = canvas_green;
                canvas_before_blue[local_shape] = canvas_blue;
                const float mask = shape_mask(
                    shape_index,
                    point_x,
                    point_y,
                    centers,
                    half_sizes,
                    sine_angles,
                    cosine_angles,
                    shape_types,
                    pixel_scale,
                    softness
                );
                const float alpha = mask * presence[shape_index];
                canvas_red += alpha * (colors[shape_index * 3] - canvas_red);
                canvas_green += alpha * (colors[shape_index * 3 + 1] - canvas_green);
                canvas_blue += alpha * (colors[shape_index * 3 + 2] - canvas_blue);
            }

            for (int local_shape = shapes_in_checkpoint - 1; local_shape >= 0; --local_shape) {
                const int shape_index = first_shape + local_shape;
                const float center_x = centers[shape_index * 2];
                const float center_y = centers[shape_index * 2 + 1];
                const float delta_x = point_x - center_x;
                const float delta_y = point_y - center_y;
                const float sine = sine_angles[shape_index];
                const float cosine = cosine_angles[shape_index];
                const float local_x = cosine * delta_x + sine * delta_y;
                const float local_y = -sine * delta_x + cosine * delta_y;
                float distance;
                float distance_local_x;
                float distance_local_y;
                float distance_half_x;
                float distance_half_y;
                distance_partials(
                    shape_types[shape_index],
                    local_x,
                    local_y,
                    half_sizes[shape_index * 2],
                    half_sizes[shape_index * 2 + 1],
                    pixel_scale,
                    &distance,
                    &distance_local_x,
                    &distance_local_y,
                    &distance_half_x,
                    &distance_half_y
                );
                const float mask = stable_sigmoid(distance / softness);
                const float alpha = mask * presence[shape_index];
                const float alpha_gradient =
                    canvas_gradient_red
                        * (colors[shape_index * 3] - canvas_before_red[local_shape])
                    + canvas_gradient_green
                        * (colors[shape_index * 3 + 1] - canvas_before_green[local_shape])
                    + canvas_gradient_blue
                        * (colors[shape_index * 3 + 2] - canvas_before_blue[local_shape]);
                const float distance_gradient = alpha_gradient * presence[shape_index]
                    * mask * (1.0f - mask) / softness;
                const float local_x_gradient = distance_gradient * distance_local_x;
                const float local_y_gradient = distance_gradient * distance_local_y;
                const int shared_base = local_shape * PARAM_GRADS;

                atomicAdd(
                    &shared_shape_gradient[shared_base],
                    -cosine * local_x_gradient + sine * local_y_gradient
                );
                atomicAdd(
                    &shared_shape_gradient[shared_base + 1],
                    -sine * local_x_gradient - cosine * local_y_gradient
                );
                atomicAdd(
                    &shared_shape_gradient[shared_base + 2],
                    distance_gradient * distance_half_x
                );
                atomicAdd(
                    &shared_shape_gradient[shared_base + 3],
                    distance_gradient * distance_half_y
                );
                atomicAdd(
                    &shared_shape_gradient[shared_base + 4],
                    local_x_gradient * delta_y - local_y_gradient * delta_x
                );
                atomicAdd(
                    &shared_shape_gradient[shared_base + 5],
                    local_x_gradient * delta_x + local_y_gradient * delta_y
                );
                atomicAdd(
                    &shared_shape_gradient[shared_base + 6],
                    canvas_gradient_red * alpha
                );
                atomicAdd(
                    &shared_shape_gradient[shared_base + 7],
                    canvas_gradient_green * alpha
                );
                atomicAdd(
                    &shared_shape_gradient[shared_base + 8],
                    canvas_gradient_blue * alpha
                );
                atomicAdd(
                    &shared_shape_gradient[shared_base + 9],
                    alpha_gradient * mask
                );

                const float remaining_alpha = 1.0f - alpha;
                canvas_gradient_red *= remaining_alpha;
                canvas_gradient_green *= remaining_alpha;
                canvas_gradient_blue *= remaining_alpha;
            }
        }
        __syncthreads();

        for (int shared_index = local_thread;
             shared_index < shapes_in_checkpoint * PARAM_GRADS;
             shared_index += blockDim.x) {
            const int local_shape = shared_index / PARAM_GRADS;
            const int parameter = shared_index - local_shape * PARAM_GRADS;
            const int shape_index = first_shape + local_shape;
            const float value = shared_shape_gradient[shared_index];
            if (parameter == 0) {
                atomicAdd(&center_gradient[shape_index * 2], value);
            } else if (parameter == 1) {
                atomicAdd(&center_gradient[shape_index * 2 + 1], value);
            } else if (parameter == 2) {
                atomicAdd(&half_size_gradient[shape_index * 2], value);
            } else if (parameter == 3) {
                atomicAdd(&half_size_gradient[shape_index * 2 + 1], value);
            } else if (parameter == 4) {
                atomicAdd(&sine_gradient[shape_index], value);
            } else if (parameter == 5) {
                atomicAdd(&cosine_gradient[shape_index], value);
            } else if (parameter < 9) {
                atomicAdd(&color_gradient[shape_index * 3 + parameter - 6], value);
            } else {
                atomicAdd(&presence_gradient[shape_index], value);
            }
        }
        __syncthreads();
    }

    if (point_is_active) {
        atomicAdd(&background_gradient[0], canvas_gradient_red);
        atomicAdd(&background_gradient[1], canvas_gradient_green);
        atomicAdd(&background_gradient[2], canvas_gradient_blue);
    }
}
"""


_state_lock = threading.Lock()
_cupy: Any | None = None
_cupy_checked = False
_kernels: tuple[Any, Any] | None = None
_kernel_compile_attempted = False
_unavailable_reason: str | None = None
_torch_cuda_dll_handle: Any | None = None


def _load_cupy() -> Any | None:
    global _cupy, _cupy_checked, _unavailable_reason, _torch_cuda_dll_handle
    with _state_lock:
        if _cupy_checked:
            return _cupy
        _cupy_checked = True
        try:
            if os.name == "nt":
                torch_library_dir = Path(torch.__file__).resolve().parent / "lib"
                if torch_library_dir.is_dir():
                    _torch_cuda_dll_handle = os.add_dll_directory(str(torch_library_dir))
                    os.environ["PATH"] = (
                        str(torch_library_dir) + os.pathsep + os.environ.get("PATH", "")
                    )
                    if not os.environ.get("CUDA_PATH"):
                        runtime_shim = Path(tempfile.gettempdir()) / "shape_vectorizer_cuda_runtime"
                        (runtime_shim / "bin").mkdir(parents=True, exist_ok=True)
                        (runtime_shim / "include").mkdir(parents=True, exist_ok=True)
                        os.environ["CUDA_PATH"] = str(runtime_shim)
            import cupy
        except Exception as error:  # CuPy is an optional acceleration dependency.
            _unavailable_reason = f"CuPy is not importable: {error}"
            return None
        _cupy = cupy
        return cupy


def _get_kernels() -> tuple[Any, Any]:
    global _kernel_compile_attempted, _kernels, _unavailable_reason
    if _kernels is not None:
        return _kernels
    if _kernel_compile_attempted:
        raise CudaRendererUnavailable(
            _unavailable_reason or "CuPy NVRTC compilation previously failed"
        )
    cupy = _load_cupy()
    if cupy is None:
        raise CudaRendererUnavailable(_unavailable_reason or "CuPy is unavailable")
    if not torch.cuda.is_available():
        _unavailable_reason = "PyTorch CUDA is unavailable"
        raise CudaRendererUnavailable(_unavailable_reason)
    with _state_lock:
        if _kernels is not None:
            return _kernels
        if _kernel_compile_attempted:
            raise CudaRendererUnavailable(
                _unavailable_reason or "CuPy NVRTC compilation previously failed"
            )
        _kernel_compile_attempted = True
        try:
            module = cupy.RawModule(
                code=_CUDA_SOURCE,
                options=("--std=c++14",),
                name_expressions=("sampled_forward", "sampled_backward"),
            )
            _kernels = (
                module.get_function("sampled_forward"),
                module.get_function("sampled_backward"),
            )
        except Exception as error:
            _unavailable_reason = f"CuPy NVRTC compilation failed: {error}"
            raise CudaRendererUnavailable(_unavailable_reason) from error
    return _kernels


def cuda_renderer_available(*, probe_kernel: bool = False) -> bool:
    """Return whether the optional CUDA path is usable.

    ``probe_kernel=True`` also compiles the kernels, which can take a moment on
    the first call. Normal imports remain cheap and work without CuPy installed.
    """

    global _unavailable_reason
    if not torch.cuda.is_available():
        _unavailable_reason = "PyTorch CUDA is unavailable"
        return False
    cupy = _load_cupy()
    if cupy is None:
        return False
    if _kernel_compile_attempted and _kernels is None:
        return False
    try:
        if int(cupy.cuda.runtime.getDeviceCount()) <= 0:
            _unavailable_reason = "CuPy cannot see a CUDA device"
            return False
        if probe_kernel:
            _get_kernels()
    except Exception as error:
        if not isinstance(error, CudaRendererUnavailable):
            _unavailable_reason = f"CuPy CUDA probe failed: {error}"
        return False
    return True


def cuda_renderer_unavailable_reason(*, probe_kernel: bool = False) -> str | None:
    """Return a human-readable reason, or ``None`` when CUDA is available."""

    if cuda_renderer_available(probe_kernel=probe_kernel):
        return None
    return _unavailable_reason or "The optional CUDA renderer is unavailable"


def _cupy_view(cupy: Any, tensor: Tensor) -> Any:
    try:
        return cupy.from_dlpack(tensor.detach())
    except AttributeError:  # Compatibility with older CuPy releases.
        return cupy.fromDlpack(torch.utils.dlpack.to_dlpack(tensor.detach()))


def _validate_decoded_inputs(
    centers: Tensor,
    half_sizes: Tensor,
    sine_angles: Tensor,
    cosine_angles: Tensor,
    colors: Tensor,
    background: Tensor,
    presence: Tensor,
    shape_types: Tensor,
    points: Tensor,
) -> tuple[int, int]:
    if centers.ndim != 3 or centers.shape[0] != 1 or centers.shape[-1] != 2:
        raise ValueError("centers must have shape [1, shapes, 2]")
    shape_count = centers.shape[1]
    expected = {
        "half_sizes": (1, shape_count, 2),
        "sine_angles": (1, shape_count),
        "cosine_angles": (1, shape_count),
        "colors": (1, shape_count, 3),
        "background": (1, 3),
        "presence": (1, shape_count),
        "shape_types": (1, shape_count),
    }
    actual = {
        "half_sizes": tuple(half_sizes.shape),
        "sine_angles": tuple(sine_angles.shape),
        "cosine_angles": tuple(cosine_angles.shape),
        "colors": tuple(colors.shape),
        "background": tuple(background.shape),
        "presence": tuple(presence.shape),
        "shape_types": tuple(shape_types.shape),
    }
    for name, expected_shape in expected.items():
        if actual[name] != expected_shape:
            raise ValueError(f"{name} must have shape {expected_shape}, got {actual[name]}")
    if points.ndim != 2 or points.shape[-1] != 2:
        raise ValueError("points must have shape [point_count, 2]")
    if shape_count <= 0 or points.shape[0] <= 0:
        raise ValueError("At least one shape and one point are required")
    tensors = (
        centers,
        half_sizes,
        sine_angles,
        cosine_angles,
        colors,
        background,
        presence,
        shape_types,
        points,
    )
    if not all(tensor.is_cuda for tensor in tensors):
        raise ValueError("All custom-renderer inputs must be CUDA tensors")
    device = centers.device
    if any(tensor.device != device for tensor in tensors):
        raise ValueError("All custom-renderer inputs must be on the same CUDA device")
    float_tensors = tensors[:7] + (points,)
    if any(tensor.dtype != torch.float32 for tensor in float_tensors):
        raise ValueError("Custom-renderer floating-point inputs must use float32")
    if shape_types.dtype != torch.int32:
        raise ValueError("shape_types must use int32")
    if not all(tensor.is_contiguous() for tensor in tensors):
        raise ValueError("All custom-renderer inputs must be contiguous")
    return shape_count, points.shape[0]


class _CudaSampledRenderFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        centers: Tensor,
        half_sizes: Tensor,
        sine_angles: Tensor,
        cosine_angles: Tensor,
        colors: Tensor,
        background: Tensor,
        presence: Tensor,
        shape_types: Tensor,
        points: Tensor,
        pixel_scale: float,
        softness: float,
    ) -> Tensor:
        if softness <= 0.0:
            raise ValueError("softness must be positive")
        shape_count, point_count = _validate_decoded_inputs(
            centers,
            half_sizes,
            sine_angles,
            cosine_angles,
            colors,
            background,
            presence,
            shape_types,
            points,
        )
        forward_kernel, _ = _get_kernels()
        cupy = _load_cupy()
        assert cupy is not None
        checkpoint_count = (shape_count + 15) // 16
        output = torch.empty((1, 3, point_count), device=centers.device, dtype=torch.float32)
        checkpoints = torch.empty(
            (checkpoint_count, 3, point_count),
            device=centers.device,
            dtype=torch.float32,
        )
        device_index = centers.device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        stream_pointer = torch.cuda.current_stream(centers.device).cuda_stream
        with cupy.cuda.Device(device_index), cupy.cuda.ExternalStream(stream_pointer):
            arguments = (
                _cupy_view(cupy, centers),
                _cupy_view(cupy, half_sizes),
                _cupy_view(cupy, sine_angles),
                _cupy_view(cupy, cosine_angles),
                _cupy_view(cupy, colors),
                _cupy_view(cupy, background),
                _cupy_view(cupy, presence),
                _cupy_view(cupy, shape_types),
                _cupy_view(cupy, points),
                cupy.int32(shape_count),
                cupy.int32(point_count),
                cupy.int32(checkpoint_count),
                cupy.float32(pixel_scale),
                cupy.float32(softness),
                _cupy_view(cupy, output),
                _cupy_view(cupy, checkpoints),
            )
            threads = 128
            forward_kernel(((point_count + threads - 1) // threads,), (threads,), arguments)

        ctx.save_for_backward(
            centers,
            half_sizes,
            sine_angles,
            cosine_angles,
            colors,
            presence,
            shape_types,
            points,
            checkpoints,
        )
        ctx.pixel_scale = float(pixel_scale)
        ctx.softness = float(softness)
        ctx.background_shape = tuple(background.shape)
        return output

    @staticmethod
    def backward(ctx: Any, output_gradient: Tensor) -> tuple[Tensor | None, ...]:
        (
            centers,
            half_sizes,
            sine_angles,
            cosine_angles,
            colors,
            presence,
            shape_types,
            points,
            checkpoints,
        ) = ctx.saved_tensors
        _, backward_kernel = _get_kernels()
        cupy = _load_cupy()
        assert cupy is not None
        output_gradient = output_gradient.contiguous()
        center_gradient = torch.zeros_like(centers)
        half_size_gradient = torch.zeros_like(half_sizes)
        sine_gradient = torch.zeros_like(sine_angles)
        cosine_gradient = torch.zeros_like(cosine_angles)
        color_gradient = torch.zeros_like(colors)
        presence_gradient = torch.zeros_like(presence)
        background_gradient = torch.zeros(
            ctx.background_shape,
            device=centers.device,
            dtype=torch.float32,
        )
        shape_count = centers.shape[1]
        point_count = points.shape[0]
        checkpoint_count = checkpoints.shape[0]
        device_index = centers.device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        stream_pointer = torch.cuda.current_stream(centers.device).cuda_stream
        with cupy.cuda.Device(device_index), cupy.cuda.ExternalStream(stream_pointer):
            arguments = (
                _cupy_view(cupy, centers),
                _cupy_view(cupy, half_sizes),
                _cupy_view(cupy, sine_angles),
                _cupy_view(cupy, cosine_angles),
                _cupy_view(cupy, colors),
                _cupy_view(cupy, presence),
                _cupy_view(cupy, shape_types),
                _cupy_view(cupy, points),
                _cupy_view(cupy, output_gradient),
                _cupy_view(cupy, checkpoints),
                cupy.int32(shape_count),
                cupy.int32(point_count),
                cupy.int32(checkpoint_count),
                cupy.float32(ctx.pixel_scale),
                cupy.float32(ctx.softness),
                _cupy_view(cupy, center_gradient),
                _cupy_view(cupy, half_size_gradient),
                _cupy_view(cupy, sine_gradient),
                _cupy_view(cupy, cosine_gradient),
                _cupy_view(cupy, color_gradient),
                _cupy_view(cupy, presence_gradient),
                _cupy_view(cupy, background_gradient),
            )
            threads = 128
            backward_kernel(((point_count + threads - 1) // threads,), (threads,), arguments)

        return (
            center_gradient,
            half_size_gradient,
            sine_gradient,
            cosine_gradient,
            color_gradient,
            background_gradient,
            presence_gradient,
            None,
            None,
            None,
            None,
        )


def cuda_render_decoded_sampled(
    centers: Tensor,
    half_sizes: Tensor,
    sine_angles: Tensor,
    cosine_angles: Tensor,
    colors: Tensor,
    background: Tensor,
    shape_types: Tensor,
    points: Tensor,
    *,
    presence: Tensor | None = None,
    canvas_size: tuple[int, int],
    softness_px: float = 1.0,
) -> Tensor:
    """Render fixed-type decoded shapes at sampled points with custom CUDA.

    This low-level function requires contiguous float32 CUDA tensors and batch
    size one. Shape types are integer IDs: 0 ellipse, 1 rectangle, 2 triangle.
    Omitting ``presence`` is equivalent to giving every shape a weight of one.
    """

    height, width = canvas_size
    if height <= 0 or width <= 0:
        raise ValueError("canvas_size dimensions must be positive")
    if presence is None:
        presence = torch.ones_like(sine_angles, memory_format=torch.contiguous_format)
    return _CudaSampledRenderFunction.apply(
        centers,
        half_sizes,
        sine_angles,
        cosine_angles,
        colors,
        background,
        presence,
        shape_types,
        points,
        float(min(height, width)),
        float(softness_px),
    )


class CudaSampledShapeRenderer(nn.Module):
    """Fixed-type sampled/grid renderer with an automatic PyTorch fallback.

    The CUDA path intentionally treats each primitive type as fixed. Geometry,
    angle, color, presence, and background remain differentiable all the way
    back to the raw 12-parameter representation.
    """

    def __init__(
        self,
        min_size: float = 0.01,
        max_size: float = 0.5,
        softness_px: float = 1.0,
        type_temperature: float = 0.7,
        allow_fallback: bool = True,
    ) -> None:
        super().__init__()
        if not 0.0 < min_size < max_size <= 1.0:
            raise ValueError("Expected 0 < min_size < max_size <= 1")
        if softness_px <= 0.0:
            raise ValueError("softness_px must be positive")
        self.min_size = min_size
        self.max_size = max_size
        self.softness_px = softness_px
        self.type_temperature = type_temperature
        self.allow_fallback = allow_fallback
        self.last_backend: str | None = None
        self.last_fallback_reason: str | None = None
        self._fallback = SoftShapeRenderer(
            min_size=min_size,
            max_size=max_size,
            softness_px=softness_px,
            chunk_size=32,
            type_temperature=type_temperature,
            hard_types=True,
            learn_shape_types=False,
        )

    def _fallback_forward(
        self,
        raw_shapes: Tensor,
        raw_background: Tensor,
        sample_points: Tensor,
        canvas_size: tuple[int, int],
        use_presence: bool,
        softness_px: float,
        reason: str,
    ) -> Tensor:
        if not self.allow_fallback:
            raise CudaRendererUnavailable(reason)
        self.last_backend = "torch"
        self.last_fallback_reason = reason
        return self._fallback(
            raw_shapes,
            raw_background,
            sample_points=sample_points,
            canvas_size=canvas_size,
            use_presence=use_presence,
            softness_px=softness_px,
        )

    def _render_sampled(
        self,
        raw_shapes: Tensor,
        raw_background: Tensor,
        *,
        sample_points: Tensor,
        canvas_size: tuple[int, int],
        use_presence: bool = False,
        softness_px: float | None = None,
    ) -> Tensor:
        if raw_shapes.ndim != 3 or raw_shapes.shape[-1] != PARAM_COUNT:
            raise ValueError(f"raw_shapes must have shape [batch, shapes, {PARAM_COUNT}]")
        if raw_background.shape != (raw_shapes.shape[0], 3):
            raise ValueError("raw_background must have shape [batch, 3]")
        if sample_points.ndim != 2 or sample_points.shape[-1] != 2:
            raise ValueError("sample_points must have shape [point_count, 2]")
        softness = self.softness_px if softness_px is None else float(softness_px)
        if softness <= 0.0:
            raise ValueError("softness_px must be positive")
        if raw_shapes.shape[0] != 1:
            return self._fallback_forward(
                raw_shapes,
                raw_background,
                sample_points,
                canvas_size,
                use_presence,
                softness,
                "The custom CUDA path currently supports batch size one",
            )
        if not raw_shapes.is_cuda or not raw_background.is_cuda:
            return self._fallback_forward(
                raw_shapes,
                raw_background,
                sample_points,
                canvas_size,
                use_presence,
                softness,
                "Inputs are not CUDA tensors",
            )
        if sample_points.device != raw_shapes.device or raw_background.device != raw_shapes.device:
            raise ValueError("raw shapes, background, and sample points must share one device")
        if not cuda_renderer_available(probe_kernel=True):
            return self._fallback_forward(
                raw_shapes,
                raw_background,
                sample_points,
                canvas_size,
                use_presence,
                softness,
                cuda_renderer_unavailable_reason() or "The custom CUDA renderer is unavailable",
            )

        raw_float = raw_shapes.float()
        background_float = raw_background.float()
        centers = raw_float[..., 3:5].sigmoid().contiguous()
        sizes = self.min_size + (self.max_size - self.min_size) * raw_float[..., 5:7].sigmoid()
        half_sizes = (sizes * 0.5).contiguous()
        angles = math.pi * raw_float[..., 7].tanh()
        sine_angles = angles.sin().contiguous()
        cosine_angles = angles.cos().contiguous()
        colors = raw_float[..., 8:11].sigmoid().contiguous()
        background = background_float.sigmoid().contiguous()
        if use_presence:
            presence = raw_float[..., 11].sigmoid().contiguous()
        else:
            presence = torch.ones_like(raw_float[..., 11], memory_format=torch.contiguous_format)
        shape_types = raw_float[..., 0:3].argmax(dim=-1).to(torch.int32).contiguous()
        points = sample_points.to(device=raw_shapes.device, dtype=torch.float32).contiguous()
        self.last_backend = "cuda"
        self.last_fallback_reason = None
        return cuda_render_decoded_sampled(
            centers,
            half_sizes,
            sine_angles,
            cosine_angles,
            colors,
            background,
            shape_types,
            points,
            presence=presence,
            canvas_size=canvas_size,
            softness_px=softness,
        )

    def render_grid(
        self,
        raw_shapes: Tensor,
        raw_background: Tensor,
        height: int,
        width: int,
        *,
        use_presence: bool = False,
        softness_px: float | None = None,
    ) -> Tensor:
        """Render a complete grid using normalized pixel-center sample points."""

        if height <= 0 or width <= 0:
            raise ValueError("height and width must be positive")
        point_dtype = torch.float32 if raw_shapes.is_cuda else raw_shapes.dtype
        ys = (torch.arange(height, device=raw_shapes.device, dtype=point_dtype) + 0.5) / height
        xs = (torch.arange(width, device=raw_shapes.device, dtype=point_dtype) + 0.5) / width
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        points = torch.stack((xx, yy), dim=-1).reshape(-1, 2)
        rendered = self._render_sampled(
            raw_shapes,
            raw_background,
            sample_points=points,
            canvas_size=(height, width),
            use_presence=use_presence,
            softness_px=softness_px,
        )
        return rendered.reshape(raw_shapes.shape[0], 3, height, width)

    def forward(
        self,
        raw_shapes: Tensor,
        raw_background: Tensor,
        height: int | None = None,
        width: int | None = None,
        *,
        sample_points: Tensor | None = None,
        canvas_size: tuple[int, int] | None = None,
        use_presence: bool = False,
        softness_px: float | None = None,
    ) -> Tensor:
        """Render either a full grid or explicitly supplied normalized points."""

        if sample_points is None:
            if height is None or width is None:
                raise ValueError("height and width are required for full-grid rendering")
            return self.render_grid(
                raw_shapes,
                raw_background,
                height,
                width,
                use_presence=use_presence,
                softness_px=softness_px,
            )
        if canvas_size is None:
            if height is None or width is None:
                raise ValueError("canvas_size is required when rendering sampled points")
            canvas_size = (height, width)
        return self._render_sampled(
            raw_shapes,
            raw_background,
            sample_points=sample_points,
            canvas_size=canvas_size,
            use_presence=use_presence,
            softness_px=softness_px,
        )


__all__ = [
    "CudaRendererUnavailable",
    "CudaSampledShapeRenderer",
    "cuda_render_decoded_sampled",
    "cuda_renderer_available",
    "cuda_renderer_unavailable_reason",
]
