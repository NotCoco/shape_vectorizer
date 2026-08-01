from __future__ import annotations

import subprocess
import unittest

from vectorlearner.gpu_guard import (
    GpuGuard,
    GpuGuardConfig,
    GpuSafetyError,
    GpuSample,
    parse_nvidia_smi_sample,
    read_nvidia_smi_sample,
)


def sample(
    *,
    temperature: float = 60.0,
    free_mib: int = 4096,
    software_slowdown: bool = False,
    hardware_slowdown: bool = False,
) -> GpuSample:
    return GpuSample(
        temperature_c=temperature,
        power_draw_w=120.0,
        power_limit_w=160.0,
        memory_used_mib=8188 - free_mib,
        memory_free_mib=free_mib,
        memory_total_mib=8188,
        utilization_percent=98.0,
        software_thermal_slowdown=software_slowdown,
        hardware_thermal_slowdown=hardware_slowdown,
    )


class GpuGuardTests(unittest.TestCase):
    def test_parser_reads_verified_nvidia_smi_shape(self) -> None:
        parsed = parse_nvidia_smi_sample(
            "48, 10.10, 160.00, 2682, 5267, 8188, 15, Not Active, Not Active\n"
        )
        self.assertEqual(parsed.temperature_c, 48.0)
        self.assertEqual(parsed.memory_free_mib, 5267)
        self.assertFalse(parsed.thermal_slowdown)

    def test_reader_only_uses_read_only_query_flags(self) -> None:
        captured: list[str] = []

        def runner(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
            captured.extend(command)
            return subprocess.CompletedProcess(
                command,
                0,
                stdout="48, 10.10, 160.00, 2682, 5267, 8188, 15, Not Active, Not Active\n",
                stderr="",
            )

        read_nvidia_smi_sample(executable="nvidia-smi", runner=runner)
        self.assertIn("--query-gpu=temperature.gpu,power.draw,power.limit,memory.used,memory.free,memory.total,utilization.gpu,clocks_event_reasons.sw_thermal_slowdown,clocks_event_reasons.hw_thermal_slowdown", captured)
        self.assertFalse(any(argument in {"-pl", "--power-limit"} for argument in captured))

    def test_warns_once_at_warning_temperature(self) -> None:
        messages: list[str] = []
        guard = GpuGuard(
            GpuGuardConfig(poll_interval_seconds=0),
            sample_provider=lambda: sample(temperature=78),
            warning_sink=messages.append,
        )
        guard.check()
        guard.check()
        self.assertEqual(len(messages), 1)
        self.assertIn("warning threshold", messages[0])

    def test_sustained_heat_pauses_until_below_pause_threshold(self) -> None:
        readings = iter(
            [
                sample(temperature=80),
                sample(temperature=80),
                sample(temperature=80),
                sample(temperature=79),
            ]
        )
        sleeps: list[float] = []
        guard = GpuGuard(
            GpuGuardConfig(poll_interval_seconds=0),
            sample_provider=lambda: next(readings),
            warning_sink=lambda _message: None,
            sleep=sleeps.append,
        )
        guard.check()
        guard.check()
        cooled = guard.check()
        self.assertEqual(sleeps, [10.0])
        self.assertIsNotNone(cooled)
        assert cooled is not None
        self.assertEqual(cooled.temperature_c, 79)

    def test_cancels_at_hard_temperature_or_thermal_slowdown(self) -> None:
        for reading in (sample(temperature=82), sample(software_slowdown=True)):
            with self.subTest(reading=reading):
                guard = GpuGuard(sample_provider=lambda reading=reading: reading)
                with self.assertRaises(GpuSafetyError):
                    guard.check(force=True)

    def test_enforces_one_gib_global_vram_reserve(self) -> None:
        guard = GpuGuard(sample_provider=lambda: sample(free_mib=1023))
        with self.assertRaisesRegex(GpuSafetyError, "1024 MiB reserve"):
            guard.check(force=True)

        safe_guard = GpuGuard(sample_provider=lambda: sample(free_mib=1024))
        self.assertIsNotNone(safe_guard.check(force=True))

    def test_context_manager_and_callback_are_training_loop_compatible(self) -> None:
        calls = 0

        def provider() -> GpuSample:
            nonlocal calls
            calls += 1
            return sample()

        guard = GpuGuard(
            GpuGuardConfig(poll_interval_seconds=0),
            sample_provider=provider,
        )
        with guard as active:
            self.assertIs(active, guard)
            active.callback(step=1, loss=0.5)
        self.assertEqual(calls, 2)


if __name__ == "__main__":
    unittest.main()
