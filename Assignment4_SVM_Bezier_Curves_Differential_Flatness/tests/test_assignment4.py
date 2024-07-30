import unittest
import subprocess
import tempfile
from pathlib import Path


class TestAssignment4(unittest.TestCase):

    def test_1(self):
        subprocess.run(
            ["python3",
            "opt_toy.py"], check=True)

        # make sure it actually created some output file
        self.assertTrue(Path("opt_toy.pdf").exists())

    def test_2(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            p_out = Path(tmpdirname) / "bezier_0.pdf"
            # check there is a file that we can execute
            subprocess.run(
                ["python3",
                "opt_safe.py",
                "cfg/bezier_0.yaml",
                p_out], check=True)
            
            # make sure it actually created some output file
            self.assertTrue(p_out.exists())

    def test_3(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            p_out = Path(tmpdirname) / "bezier_0.pdf"
            p_exp = Path(tmpdirname) / "bezier_car_plan_0.yaml"
            # run the export
            subprocess.run(
                ["python3",
                "opt_safe.py",
                "cfg/bezier_0.yaml",
                p_out,
                "--export-car", p_exp], check=True)
            
            # make sure a file was created
            self.assertTrue(p_out.exists())
            # make sure there is an output file
            self.assertTrue(p_exp.exists())

    def test_4(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            p_out = Path(tmpdirname) / "bezier_0.pdf"
            p_exp = Path(tmpdirname) / "bezier_arm_plan_0.yaml"
            # run the export
            subprocess.run(
                ["python3",
                "opt_safe.py",
                "cfg/bezier_0.yaml",
                p_out,
                "--export-arm", p_exp], check=True)
            
            # make sure a file was created
            self.assertTrue(p_out.exists())
            # make sure there is an output file
            self.assertTrue(p_exp.exists())


if __name__ == '__main__':
    unittest.main()

