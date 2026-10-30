import os
import subprocess
import unittest
from unittest.mock import patch

from nvidia_client import generate_nvidia_response, nvidia_config
from runtime_status import openshell_status
from openshell_runner import run_in_openshell
from openshell_runner import verify_openshell_sandbox


class NvidiaIntegrationTests(unittest.TestCase):
    def test_missing_credentials_are_clear(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "NVIDIA_API_KEY"):
                nvidia_config()

    def test_invalid_endpoint_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "BASE_URL"):
            nvidia_config(api_key="test", base_url="not-a-url")

    def test_mocked_nemotron_response(self):
        with patch("nvidia_client._request", return_value={"choices": [{"message": {"content": "Hello from Nemotron"}}]}):
            result = generate_nvidia_response([{"role": "user", "content": "Hello"}], api_key="test")
        self.assertEqual("Hello from Nemotron", result["text"])
        self.assertEqual("nvidia", result["provider"])

    def test_mocked_nemotron_tool_call_uses_existing_dispatcher(self):
        responses = [
            {"choices": [{"message": {"content": None, "tool_calls": [{"id": "call-1", "function": {"name": "calculate", "arguments": '{"expression":"2 + 2"}'}}]}}]},
            {"choices": [{"message": {"content": "The answer is 4."}}]},
        ]
        with patch("nvidia_client._request", side_effect=responses):
            result = generate_nvidia_response([{"role": "user", "content": "Calculate 2 + 2"}], api_key="test")
        self.assertEqual("The answer is 4.", result["text"])

    def test_openshell_detection_is_honest(self):
        status = openshell_status()
        self.assertIn("available", status)
        self.assertIn("message", status)

    def test_openshell_runner_does_not_use_shell_and_reports_missing_cli(self):
        with patch("openshell_runner.shutil.which", return_value=None):
            result = run_in_openshell(["python", "-c", "print('ok')"], "rams-demo")
        self.assertFalse(result["success"])
        self.assertFalse(result["sandboxed"])

    def test_openshell_runner_invokes_only_the_sandbox_cli_with_a_minimal_environment(self):
        completed = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="inside sandbox\n", stderr=""
        )
        with patch("openshell_runner.subprocess.run", return_value=completed) as run:
            result = run_in_openshell(["python3", "-c", "print('ok')"], "rams-demo", executable="/fake/openshell")
        self.assertTrue(result["success"])
        self.assertEqual(
            ["/fake/openshell", "sandbox", "exec", "-n", "rams-demo", "--", "python3", "-c", "print('ok')"],
            run.call_args.args[0],
        )
        self.assertEqual({"PATH"}, set(run.call_args.kwargs["env"]))
        self.assertFalse(result["local_fallback_used"])

    def test_openshell_runtime_status_checks_cli_and_gateway(self):
        version = subprocess.CompletedProcess(args=[], returncode=0, stdout="OpenShell 1.2.3\n", stderr="")
        gateway = subprocess.CompletedProcess(args=[], returncode=0, stdout="gateway ready\n", stderr="")
        with patch("runtime_status.shutil.which", return_value="/fake/openshell"), patch("runtime_status.subprocess.run", side_effect=[version, gateway]) as run:
            status = openshell_status()
        self.assertTrue(status["available"])
        self.assertTrue(status["gateway_reachable"])
        self.assertEqual(["/fake/openshell", "status"], run.call_args_list[1].args[0])

    def test_openshell_verification_success_uses_fixed_harmless_command(self):
        with patch("openshell_runner.run_in_openshell", return_value={"success": True, "returncode": 0, "stdout": "RAMS OpenShell verification OK\n", "stderr": "", "sandboxed": True, "local_fallback_used": False}) as runner:
            result = verify_openshell_sandbox("rams-demo")
        self.assertTrue(result["success"])
        self.assertEqual(["python3", "-c", "print('RAMS OpenShell verification OK')"], runner.call_args.args[0])
        self.assertFalse(result["security_policy"]["local_execution_fallback"])

    def test_openshell_verification_failure_never_falls_back_locally(self):
        with patch("openshell_runner.run_in_openshell", return_value={"success": False, "error": "OpenShell CLI is not installed.", "sandboxed": False, "local_fallback_used": False}):
            result = verify_openshell_sandbox("rams-demo")
        self.assertFalse(result["success"])
        self.assertFalse(result["local_fallback_used"])


if __name__ == "__main__":
    unittest.main()
