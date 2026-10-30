# OpenShell demonstration boundary

The development Mac used for this submission does not have the `openshell` CLI
installed (`openshell --version` returned command not found), so no sandbox run is
claimed as completed.

After installing OpenShell from NVIDIA's documented installer, an operator can
create a retained sandbox and run the RAMS representative command:

```bash
openshell --version
openshell sandbox create --name rams-demo --detach -- python3 -c "print('RAMS controlled workflow OK')"
openshell sandbox exec -n rams-demo -- python3 -c "print('RAMS tool boundary OK')"
```

For the RAMS repository, the restricted bridge is:

```python
from openshell_runner import run_in_openshell
print(run_in_openshell(["python3", "-m", "py_compile", "opportunity_service.py"], "rams-demo"))
```

The bridge uses an argument list, never a shell, passes only `PATH`, and does not
pass Gemini/NVIDIA credentials or the Docker socket. Sandbox creation and the
operator's filesystem/network policy remain outside this application. Use the
OpenShell policy configuration and security guidance for the installed version;
RAMS reports the boundary honestly when the CLI or sandbox is unavailable.

References:

- [NVIDIA OpenShell installation](https://docs.nvidia.com/openshell/latest/about/installation/)
- [NVIDIA OpenShell sandbox execution](https://docs.nvidia.com/openshell/latest/sandboxes/manage-sandboxes)
