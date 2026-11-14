"""Import-safe shared PyTorch runtime inventory for native ROCm test scripts."""
from __future__ import annotations


def inspect_torch(torch) -> dict:
    info = {"torch_version": str(getattr(torch, "__version__", "unknown")),
            "hip_version": getattr(torch.version, "hip", None),
            "cuda_version": getattr(torch.version, "cuda", None),
            "device_count": int(torch.cuda.device_count()),
            "available": bool(torch.cuda.is_available()), "gpu_name": "", "gpu_arch": ""}
    if info["available"] and info["device_count"] == 1:
        info["gpu_name"] = str(torch.cuda.get_device_name(0))
        info["gpu_arch"] = str(getattr(torch.cuda.get_device_properties(0), "gcnArchName", ""))
    return info
