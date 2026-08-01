from __future__ import annotations

import csv
import os
import shutil
import subprocess
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence


NVIDIA_SMI_FIELDS = (
    "temperature.gpu",
    "power.draw",
    "power.limit",
    "memory.used",
    "memory.free",
    "memory.total",
    "utilization.gpu",
    "clocks_event_reasons.sw_thermal_slowdown",
    "clocks_event_reasons.hw_thermal_slowdown",
)


class GpuTelemetryError(RuntimeError):
    """Raised when GPU safety telemetry cannot be read reliably."""


class GpuSafetyError(RuntimeError):
    """Raised at a training-loop checkpoint when a safety limit is crossed."""

    def __init__(self, message: str, sample: GpuSample | None = None) -> None:
        super().__init__(message)
        self.sample = sample


@dataclass(frozen=True)
class GpuSample:
    temperature_c: float
    power_draw_w: float | None
    power_limit_w: float | None
    memory_used_mib: int
    memory_free_mib: int
    memory_total_mib: int
    utilization_percent: float | None
    software_thermal_slowdown: bool
    hardware_thermal_slowdown: bool

    @property
    def thermal_slowdown(self) -> bool:
        return self.software_thermal_slowdown or self.hardware_thermal_slowdown


@dataclass(frozen=True)
class GpuGuardConfig:
    gpu_index: int = 0
    poll_interval_seconds: float = 2.0
    warning_temperature_c: float = 78.0
    pause_temperature_c: float = 80.0
    cancel_temperature_c: float = 82.0
    sustained_hot_samples: int = 3
    cooldown_seconds: float = 10.0
    max_cooldown_cycles: int = 3
    minimum_free_vram_mib: int = 1024
    query_timeout_seconds: float = 3.0

    def __post_init__(self) -> None:
        if self.gpu_index < 0:
            raise ValueError("gpu_index must be non-negative")
        if self.poll_interval_seconds < 0 or self.cooldown_seconds < 0:
            raise ValueError("poll and cooldown intervals must be non-negative")
        if not (
            self.warning_temperature_c
            <= self.pause_temperature_c
            < self.cancel_temperature_c
        ):
            raise ValueError("temperature limits must satisfy warning <= pause < cancel")
        if self.sustained_hot_samples < 1 or self.max_cooldown_cycles < 1:
            raise ValueError("sample and cooldown counts must be positive")
        if self.minimum_free_vram_mib < 1:
            raise ValueError("minimum_free_vram_mib must be positive")
        if self.query_timeout_seconds <= 0:
            raise ValueError("query_timeout_seconds must be positive")


def _optional_number(value: str) -> float | None:
    normalized = value.strip().lower()
    if normalized in {"", "n/a", "[n/a]", "not supported"}:
        return None
    try:
        return float(value)
    except ValueError as exc:
        raise GpuTelemetryError(f"Invalid nvidia-smi number: {value!r}") from exc


def _required_number(value: str, field: str) -> float:
    parsed = _optional_number(value)
    if parsed is None:
        raise GpuTelemetryError(f"nvidia-smi did not report {field}")
    return parsed


def _slowdown_active(value: str) -> bool:
    normalized = value.strip().lower()
    return normalized not in {"", "0", "false", "n/a", "[n/a]", "not active"}


def parse_nvidia_smi_sample(output: str) -> GpuSample:
    lines = [line for line in output.splitlines() if line.strip()]
    if len(lines) != 1:
        raise GpuTelemetryError(f"Expected one nvidia-smi GPU row, received {len(lines)}")
    values = [item.strip() for item in next(csv.reader(lines))]
    if len(values) != len(NVIDIA_SMI_FIELDS):
        raise GpuTelemetryError(
            f"Expected {len(NVIDIA_SMI_FIELDS)} nvidia-smi fields, received {len(values)}"
        )

    temperature, draw, limit, used, free, total, utilization, sw_slow, hw_slow = values
    return GpuSample(
        temperature_c=_required_number(temperature, "GPU temperature"),
        power_draw_w=_optional_number(draw),
        power_limit_w=_optional_number(limit),
        memory_used_mib=round(_required_number(used, "used VRAM")),
        memory_free_mib=round(_required_number(free, "free VRAM")),
        memory_total_mib=round(_required_number(total, "total VRAM")),
        utilization_percent=_optional_number(utilization),
        software_thermal_slowdown=_slowdown_active(sw_slow),
        hardware_thermal_slowdown=_slowdown_active(hw_slow),
    )


def find_nvidia_smi() -> str:
    discovered = shutil.which("nvidia-smi")
    if discovered:
        return discovered
    windows_root = os.environ.get("SystemRoot") or os.environ.get("WINDIR")
    if windows_root:
        candidate = Path(windows_root) / "System32" / "nvidia-smi.exe"
        if candidate.is_file():
            return str(candidate)
    return "nvidia-smi"


def read_nvidia_smi_sample(
    *,
    gpu_index: int = 0,
    executable: str | None = None,
    timeout_seconds: float = 3.0,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> GpuSample:
    command = [
        executable or find_nvidia_smi(),
        f"--id={gpu_index}",
        f"--query-gpu={','.join(NVIDIA_SMI_FIELDS)}",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = runner(
            command,
            capture_output=True,
            text=True,
            check=True,
            timeout=timeout_seconds,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise GpuTelemetryError(f"Unable to query GPU {gpu_index} with nvidia-smi: {exc}") from exc
    return parse_nvidia_smi_sample(result.stdout)


WarningSink = Callable[[str], None]
SampleProvider = Callable[[], GpuSample]


def _default_warning_sink(message: str) -> None:
    warnings.warn(message, RuntimeWarning, stacklevel=3)


class GpuGuard:
    """Read-only, cooperative GPU safety guard for training and fitting loops.

    Call the guard at normal loop boundaries, or pass ``guard.callback`` to a
    training hook. Polling is rate-limited, so calling it every step is cheap.
    The guard never changes clocks, fan curves, or power limits.
    """

    def __init__(
        self,
        config: GpuGuardConfig | None = None,
        *,
        enabled: bool = True,
        sample_provider: SampleProvider | None = None,
        warning_sink: WarningSink | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config or GpuGuardConfig()
        self.enabled = enabled
        self._sample_provider = sample_provider or self._read_sample
        self._warning_sink = warning_sink or _default_warning_sink
        self._clock = clock
        self._sleep = sleep
        self._last_poll_time: float | None = None
        self._last_sample: GpuSample | None = None
        self._hot_samples = 0
        self._temperature_warning_active = False

    @property
    def last_sample(self) -> GpuSample | None:
        return self._last_sample

    def _read_sample(self) -> GpuSample:
        return read_nvidia_smi_sample(
            gpu_index=self.config.gpu_index,
            timeout_seconds=self.config.query_timeout_seconds,
        )

    def __enter__(self) -> GpuGuard:
        self.check(force=True)
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        return None

    def __call__(self, *_args: object, force: bool = False, **_kwargs: object) -> GpuSample | None:
        return self.check(force=force)

    def callback(self, *_args: object, **_kwargs: object) -> GpuSample | None:
        """Training-hook-compatible alias for :meth:`check`."""

        return self.check()

    def check(self, *, force: bool = False) -> GpuSample | None:
        if not self.enabled:
            return None
        now = self._clock()
        if (
            not force
            and self._last_poll_time is not None
            and now - self._last_poll_time < self.config.poll_interval_seconds
        ):
            return self._last_sample

        sample = self._take_sample(now)
        return self._evaluate(sample)

    def _take_sample(self, now: float | None = None) -> GpuSample:
        try:
            sample = self._sample_provider()
        except GpuTelemetryError:
            raise
        except Exception as exc:
            raise GpuTelemetryError(f"GPU telemetry provider failed: {exc}") from exc
        self._last_sample = sample
        self._last_poll_time = self._clock() if now is None else now
        return sample

    def _raise_for_hard_limit(self, sample: GpuSample) -> None:
        if sample.thermal_slowdown:
            raise GpuSafetyError("GPU thermal slowdown became active; cancelling the job", sample)
        if sample.temperature_c >= self.config.cancel_temperature_c:
            raise GpuSafetyError(
                f"GPU reached {sample.temperature_c:.0f} C; cancelling at the "
                f"{self.config.cancel_temperature_c:.0f} C safety limit",
                sample,
            )
        if sample.memory_free_mib < self.config.minimum_free_vram_mib:
            raise GpuSafetyError(
                f"Only {sample.memory_free_mib} MiB of global VRAM remains; cancelling to preserve "
                f"the {self.config.minimum_free_vram_mib} MiB reserve",
                sample,
            )

    def _evaluate(self, sample: GpuSample) -> GpuSample:
        self._raise_for_hard_limit(sample)

        if sample.temperature_c >= self.config.warning_temperature_c:
            if not self._temperature_warning_active:
                self._warning_sink(
                    f"GPU temperature is {sample.temperature_c:.0f} C "
                    f"(warning threshold {self.config.warning_temperature_c:.0f} C)"
                )
                self._temperature_warning_active = True
        else:
            self._temperature_warning_active = False

        if sample.temperature_c >= self.config.pause_temperature_c:
            self._hot_samples += 1
        else:
            self._hot_samples = 0

        if self._hot_samples >= self.config.sustained_hot_samples:
            return self._cool_down(sample)
        return sample

    def _cool_down(self, sample: GpuSample) -> GpuSample:
        for cycle in range(1, self.config.max_cooldown_cycles + 1):
            self._warning_sink(
                f"GPU stayed at or above {self.config.pause_temperature_c:.0f} C; "
                f"pausing for {self.config.cooldown_seconds:g} seconds "
                f"(cooldown {cycle}/{self.config.max_cooldown_cycles})"
            )
            self._sleep(self.config.cooldown_seconds)
            sample = self._take_sample()
            self._raise_for_hard_limit(sample)
            if sample.temperature_c < self.config.pause_temperature_c:
                self._hot_samples = 0
                if sample.temperature_c < self.config.warning_temperature_c:
                    self._temperature_warning_active = False
                return sample

        raise GpuSafetyError(
            f"GPU remained at or above {self.config.pause_temperature_c:.0f} C after "
            f"{self.config.max_cooldown_cycles} cooldown cycles; cancelling the job",
            sample,
        )
