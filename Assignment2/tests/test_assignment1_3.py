import unittest
import subprocess
import tempfile
from pathlib import Path
import yaml
import numpy as np

class TestAssignment1_3(unittest.TestCase):

    def verify(self, file_in_env, file_in_plan, file_out):
        with tempfile.TemporaryDirectory() as tmpdirname:
            p_out = Path(tmpdirname) / "output.yaml"

            subprocess.run(
                ["python3",
                "collisions.py",
                Path("cfg") / file_in_env,
                Path("cfg") / file_in_plan,
                p_out],
                check=True)
            
            with open(p_out) as f:
                output = yaml.safe_load(f)

        # now verify the output
        with open(Path("cfg") / file_out) as f:
            desired_output = yaml.safe_load(f)

        x_d = np.array(desired_output["collisions"])
        x = np.array(output["collisions"])
        self.assertTrue(np.allclose(x, x_d))

    def test_1(self):
        self.verify("car_env_0.yaml", "car_plan_0.yaml", "car_0_collision_sol.yaml")

    def test_2(self):
        self.verify("arm_env_0.yaml", "arm_plan_0.yaml", "arm_0_collision_sol.yaml")


if __name__ == '__main__':
    unittest.main()