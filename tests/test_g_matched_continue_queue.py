"""Read-only unit checks for the bounded continuation queue."""
import importlib.util
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/queue_g_matched_continue_20260914.py"
spec = importlib.util.spec_from_file_location("gqueue", SCRIPT)
queue = importlib.util.module_from_spec(spec)
spec.loader.exec_module(queue)


class QueueTest(unittest.TestCase):
    def test_two_unique_runs(self):
        self.assertEqual(len(queue.RUNS), 2)
        self.assertEqual(len({r[0] for r in queue.RUNS}), 2)

    def test_same_recipe_continuation(self):
        for run in queue.RUNS:
            cmd = queue.command(run)
            self.assertIn("--no-tc", cmd)
            self.assertNotIn("--local_lr", cmd)
            self.assertEqual(cmd[cmd.index("--max_steps") + 1], "5000")
            self.assertTrue(cmd[cmd.index("--parent") + 1].endswith("global_step_10000"))

    def test_arms(self):
        self.assertEqual([r[1] for r in queue.RUNS], ["F2", "F2RL"])

    def test_incumbent_alive(self):
        with patch.object(queue, "capture", return_value=""), patch.object(queue.os, "kill"):
            self.assertTrue(queue.busy(3007083))

    def test_other_cuda_client(self):
        with patch.object(queue, "capture", return_value="999"), \
             patch.object(queue.os, "kill", side_effect=ProcessLookupError):
            self.assertTrue(queue.busy(3007083))

    def test_really_idle(self):
        with patch.object(queue, "capture", return_value=""), \
             patch.object(queue.os, "kill", side_effect=ProcessLookupError):
            self.assertFalse(queue.busy(3007083))

    def test_probe_failure_is_not_idle(self):
        with patch.object(queue, "capture", side_effect=subprocess.CalledProcessError(1, "nvidia-smi")):
            with self.assertRaises(subprocess.CalledProcessError):
                queue.busy(3007083)

    def test_permission_failure_is_not_idle(self):
        with patch.object(queue, "capture", return_value=""), \
             patch.object(queue.os, "kill", side_effect=PermissionError):
            with self.assertRaises(PermissionError):
                queue.busy(3007083)

    def test_explicit_background_guard(self):
        with patch.object(queue, "capture", side_effect=["999", "0, 30000\n" * 8]), \
             patch.object(queue.os, "kill", side_effect=ProcessLookupError):
            self.assertFalse(queue.busy(3007083, (999,)))

    def test_background_becomes_active(self):
        with patch.object(queue, "capture", side_effect=["999", "20, 30000\n" * 8]), \
             patch.object(queue.os, "kill", side_effect=ProcessLookupError):
            self.assertTrue(queue.busy(3007083, (999,)))

    def test_background_low_memory(self):
        with patch.object(queue, "capture", side_effect=["999", "0, 12000\n" * 8]), \
             patch.object(queue.os, "kill", side_effect=ProcessLookupError):
            self.assertTrue(queue.busy(3007083, (999,)))

    def test_new_pid_blocks(self):
        with patch.object(queue, "capture", return_value="999\n1000"), \
             patch.object(queue.os, "kill", side_effect=ProcessLookupError):
            self.assertTrue(queue.busy(3007083, (999,)))


if __name__ == "__main__":
    unittest.main()
