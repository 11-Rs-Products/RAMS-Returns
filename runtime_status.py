"""Read-only runtime capability checks for the RAMS status panel."""
from __future__ import annotations

import shutil
import subprocess
from typing import Any, Dict


def openshell_status() -> Dict[str, Any]:
    executable = shutil.which("openshell")
    if not executable:
        return {"available": False, "version": None, "gateway_reachable": False, "message": "OpenShell CLI not installed"}
    try:
        completed = subprocess.run([executable, "--version"], capture_output=True, text=True, timeout=5, check=False)
        output = (completed.stdout or completed.stderr).strip()
        if completed.returncode != 0:
            return {"available": False, "version": output or None, "gateway_reachable": False, "message": output or "OpenShell version check failed"}
        gateway = subprocess.run([executable, "status"], capture_output=True, text=True, timeout=5, check=False)
        gateway_output = (gateway.stdout or gateway.stderr).strip()
        return {"available": True, "version": output or None, "gateway_reachable": gateway.returncode == 0, "message": gateway_output or "OpenShell CLI available; gateway status returned no output"}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "version": None, "gateway_reachable": False, "message": f"OpenShell check failed: {exc}"}


def runtime_status(provider: str) -> Dict[str, Any]:
    shell = openshell_status()
    return {"provider": provider, "openshell": shell, "security_boundary": "OpenShell not active for RAMS" if not shell["available"] else "OpenShell detected; RAMS is not automatically sandboxed"}
