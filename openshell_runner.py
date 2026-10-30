"""Controlled OpenShell bridge for an already-created sandbox.

RAMS never installs OpenShell, creates a sandbox, passes host credentials, or
invokes a shell implicitly. The operator must create/select a policy-enforced
sandbox and then call this wrapper with a fixed executable/argument list.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from typing import Dict, List, Optional


HARMLESS_VERIFICATION_COMMAND = ["python3", "-c", "print('RAMS OpenShell verification OK')"]


def run_in_openshell(command: List[str], sandbox_name: str, timeout: int = 30, executable: Optional[str] = None) -> Dict[str, object]:
    if not command or any(not isinstance(part, str) or not part for part in command):
        raise ValueError("A non-empty argument list is required; shell strings are not accepted.")
    if not sandbox_name or any(ch.isspace() for ch in sandbox_name):
        raise ValueError("sandbox_name must be a single existing OpenShell sandbox name.")
    openshell = executable or shutil.which("openshell")
    if not openshell:
        return {"success": False, "error": "OpenShell CLI is not installed.", "sandboxed": False, "local_fallback_used": False}
    safe_env = {"PATH": os.environ.get("PATH", "")}
    try:
        result = subprocess.run([openshell, "sandbox", "exec", "-n", sandbox_name, "--", *command], capture_output=True, text=True, timeout=timeout, check=False, env=safe_env)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"success": False, "error": f"OpenShell execution failed: {exc}", "sandboxed": True, "local_fallback_used": False}
    return {"success": result.returncode == 0, "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr, "sandboxed": True, "local_fallback_used": False}


def verify_openshell_sandbox(sandbox_name: str, timeout: int = 30) -> Dict[str, object]:
    """Execute only a harmless, fixed RAMS marker in an existing sandbox."""
    result = run_in_openshell(HARMLESS_VERIFICATION_COMMAND, sandbox_name, timeout=timeout)
    result["command"] = list(HARMLESS_VERIFICATION_COMMAND)
    result["security_policy"] = {
        "host_credentials_passed": False,
        "docker_socket_exposed": False,
        "local_execution_fallback": False,
        # RAMS deliberately does not configure these policies.  They can only
        # be asserted after an operator has created and inspected a real
        # OpenShell sandbox, so do not represent them as verified here.
        "filesystem_policy": "not configured or verified by RAMS; set by the OpenShell sandbox operator",
        "network_policy": "not configured or verified by RAMS; set by the OpenShell sandbox operator",
    }
    return result
