import unittest
import subprocess
import tempfile
from pathlib import Path
import yaml

class TestAssignment2_3(unittest.TestCase):

    def verify(self, file_in):
        with tempfile.TemporaryDirectory() as tmpdirname:
            p_out = Path(tmpdirname) / "output.yaml"

            subprocess.run(
                ["python3",
                "rrt.py",
                Path("cfg") / file_in,
                p_out],
                check=True)
            
            with open(p_out) as f:
                output = yaml.safe_load(f)
                print(output)

        # now verify the output
        with open(Path("cfg") / file_in) as f:
            instance = yaml.safe_load(f)

        # Very basic checks
        self.assertEqual(instance["motionplanning"]["type"], output["plan"]["type"])
        self.assertEqual(instance["motionplanning"]["L"], output["plan"]["L"])
        self.assertEqual(instance["motionplanning"]["W"], output["plan"]["W"])
        self.assertEqual(instance["motionplanning"]["H"], output["plan"]["H"])
        self.assertEqual(instance["motionplanning"]["dt"], output["plan"]["dt"])
        self.assertEqual(instance["motionplanning"]["start"], output["plan"]["states"][0])
        self.assertTrue(len(output["plan"]["actions"]) > 0)

    def test_1(self):
        self.verify("car_0.yaml")


if __name__ == '__main__':
    unittest.main()