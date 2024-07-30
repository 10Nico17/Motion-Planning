import unittest
import subprocess
import tempfile
from pathlib import Path
import yaml

class TestAssignment2_1(unittest.TestCase):

    def verify(self, file_in, file_out):
        with tempfile.TemporaryDirectory() as tmpdirname:
            p_out = Path(tmpdirname) / "output.yaml"

            subprocess.run(
                ["python3",
                "nearest_neighbor.py",
                Path("cfg") / file_in,
                p_out],
                check=True)
            
            with open(p_out) as f:
                output = yaml.safe_load(f)
                print(output)

        # now verify the output
        with open(Path("cfg") / file_out) as f:
            desired_output = yaml.safe_load(f)

        self.assertEqual(desired_output["results"], output["results"])

    def test_1(self):
        self.verify("nn_0.yaml", "nn_0_sol.yaml")


if __name__ == '__main__':
    unittest.main()