import unittest
import subprocess
import tempfile
from pathlib import Path
import yaml
import numpy as np

class TestAssignment1_1(unittest.TestCase):

    def verify(self, file_in_actions, file_in_env, file_out):
        p = Path("cfg") / file_in_actions
        with tempfile.TemporaryDirectory() as tmpdirname:
            p_out = Path(tmpdirname) / "plan.yaml"

            subprocess.run(
                ["python3",
                "car_dynamics.py",
                p.absolute(),
                p_out.absolute()],
                check=True)
            
            with open(p_out) as f:
                output = yaml.safe_load(f)

            # visualize the output
            subprocess.run(
                ["python3",
                "car_vis.py",
                Path("cfg") / file_in_env,
                p_out,
                "--output", Path(file_out).with_suffix(".html")],
                check=True)

        # now verify the output
        with open(Path("cfg") / file_out) as f:
            desired_output = yaml.safe_load(f)

        x_d = np.array(desired_output["plan"]["states"])
        x = np.array(output["plan"]["states"])
        self.assertTrue(np.allclose(x, x_d))

    def test_1(self):
        self.verify("car_actions_0.yaml", "car_env_0.yaml", "car_plan_0.yaml")


if __name__ == '__main__':
    unittest.main()