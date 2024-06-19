import unittest
import subprocess
import tempfile
from pathlib import Path


class TestAssignment3(unittest.TestCase):

    def test_1(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            p_out = Path(tmpdirname) / "output.yaml"
            # check there is a file that we can execute
            subprocess.run(
                ["python3",
                "ompl_planner.py",
                "cfg/arm_0.yaml",
                p_out], check=True)
            
            # make sure it actually created some output file
            self.assertTrue(p_out.exists())

    '''
    def test_2(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            p_out = Path(tmpdirname) / "output.yaml"
            # check there is a file that we can execute
            subprocess.run(
                ["python3",
                "ompl_planner.py",
                "cfg/car_0.yaml",
                p_out], check=True)
            
            # make sure it actually created some output file
            self.assertTrue(p_out.exists())

    def test_3(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            p_out = Path(tmpdirname) / "output.yaml"
            p_exp = Path(tmpdirname) / "planner_data.out"
            # run the export
            subprocess.run(
                ["python3",
                "ompl_planner.py",
                "cfg/car_0.yaml",
                p_out,
                "--export-planner-data", p_exp], check=True)
            
            # make sure a file was created
            self.assertTrue(p_exp.exists())
            # make sure there is a file to visualize
            self.assertTrue(Path("ompl_tree_vis.py").exists())

    def test_4(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            p_log = Path(tmpdirname) / "car_1.log"
            # run the benchmark
            subprocess.run(
                ["python3",
                "ompl_benchmark.py",
                "cfg/car_1.yaml",
                p_log], check=True)
            
            # make sure a file was created
            self.assertTrue(p_log.exists())
            # make sure the resulting PDF is pushed
            self.assertTrue(Path("car_1.pdf").exists())
    '''


if __name__ == '__main__':
    unittest.main()

