"""Fail-closed host/GPU admission without fallback."""

from __future__ import annotations


class ResourceGateError(RuntimeError):
    pass


def admit_resources(device_kind, available_ram_bytes, required_ram_bytes, gpu_free_mib, required_gpu_mib):
    if available_ram_bytes < required_ram_bytes:
        raise ResourceGateError(
            f"insufficient host memory: available={available_ram_bytes}, required={required_ram_bytes}"
        )
    if device_kind == "GPU_REQUIRED":
        if gpu_free_mib is None:
            raise ResourceGateError("CUDA availability is required")
        if gpu_free_mib < required_gpu_mib:
            raise ResourceGateError(
                f"insufficient GPU memory: free={gpu_free_mib}, required={required_gpu_mib}"
            )
        return {
            "device_kind": device_kind, "device": "cuda:0", "threads": None,
            "available_ram_bytes": available_ram_bytes, "required_ram_bytes": required_ram_bytes,
            "gpu_free_mib": gpu_free_mib, "required_gpu_mib": required_gpu_mib,
            "cpu_fallback": False,
        }
    if device_kind == "CPU_INTENTIONAL":
        return {
            "device_kind": device_kind, "device": "cpu", "threads": 4,
            "available_ram_bytes": available_ram_bytes, "required_ram_bytes": required_ram_bytes,
            "gpu_free_mib": None, "required_gpu_mib": 0, "cpu_fallback": False,
        }
    raise ResourceGateError("unknown device kind")

