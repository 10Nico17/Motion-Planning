import unittest
import subprocess
from pathlib import Path

class TestAssignment1_2(unittest.TestCase):

    def verify(self, file_in_env, file_in_plan):
        # visualize the output
        subprocess.run(
            ["python3",
            "arm_vis.py",
            Path("cfg") / file_in_env,
            Path("cfg") / file_in_plan,
            "--output", Path(file_in_plan).with_suffix(".html")],
            check=True)

    def test_1(self):
        self.verify("arm_env_0.yaml", "arm_plan_0.yaml")


if __name__ == '__main__':
    unittest.main()