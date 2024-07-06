# Motion Planning - Programming Assignment 3

Due: 7/3/24 (midnight)

The goal of this assignment is to use OMPL such that it works for the car and manipulator robot. 

## Tasks

### Setup OMPL (Not graded)

The easiest way to get started with OMPL is through its Python bindings. It's not on PyPI yet, but prebuilt wheels make the installations on Linux and Mac fairly easy. (Windows users: Try WSL, docker, or a VM.)

#### Installation (Recommended)

On Linux, you can just use requirements.txt

```
pip install -r requirements.txt
```

On other systems, either update requirements.txt or use the following steps:

```
pip install wheel
(Download the *.whl from https://github.com/ompl/ompl/releases, prerelease)
pip install *.whl
```

#### Docker (Alternative)

You can also use a docker image, although this is an older OMPL version and you won't get code completion.
After installing docker, you can get the image using:

```
docker pull kavrakilab/omplapp
```

Then, you can execute a Python script named `script.py` that uses OMPL via:

```
docker run --rm -v ${PWD}:/home/ompl -P kavrakilab/omplapp python3 script.py
```
Here, `-v` maps your current working directory to the home folder of the docker container. You can find many example scripts in the official repository, for example at https://github.com/ompl/ompl/blob/main/demos/RigidBodyPlanning.py.

### Manipulator case (4 pt)

Implement an OMPL-based planner for the geometric case for the manipulator robot. Your code should take a single yaml file as input (that describes the motion planning problem) and output the solution in the same format that we have used in the second assignment:

```
python3 ompl_planner.py cfg/arm_0.yaml arm_plan_0.yaml
```

Note that the input file has three parts as before. The `hyperparameters` key should support the same settings as before (timelimit, goal_bias, and goal_eps) as well as a planner setting (`planner`, which can be one of `rrt`, `rrt*`, `rrt-connect`).

### Car case (2 pt)

Extend your implementation to also support the kinodynamic case for the car-like robot. The same script (`ompl_planner.py`) should be able to work for both the manipulator and the car-like robot. For example, the following command should output a solution file similar to the one from the first assignment:

```
python3 ompl_planner.py cfg/car_0.yaml car_plan_0.yaml
```

As planners, you should support `rrt` and `sst` in this case.

### Visualize the constructed tree (2 pt)

Use meshcat to visualize the search tree that RRT constructs during the search. To this end, you can use `getPlannerData` from the `Planner` class. The `PlannerData` class has methods to access vertices and edges. You can find examples on how to plot points and lines at https://github.com/rdeits/meshcat-python/tree/master/examples.

Factor your code such that the visualization can be executed as:

```
python3 ompl_planner.py cfg/car_0.yaml car_plan_0.yaml --export-planner-data planner_data.out
python3 ompl_tree_vis.py planner_data.out
```

where `planner_data.out` is a file format of your choice. You only need to support the kinodynamic car case.

### Benchmarking sampling methods (2 pt)

Use the `car_1.yaml` scenario and compare sampling methods with different samplers (use `UniformValidStateSampler`, `BridgeTestValidStateSampler`, `ObstacleBasedValidStateSampler`, and `GaussianValidStateSampler`). You may adjust the hyperparameters.

Use the `ompl.tools.Benchmark` class of OMPL to compare those four samplers to each other.
Your code should be executable as 

```
python3 ompl_benchmark.py cfg/car_1.yaml car_1.log
```

As benchmark result, push a single pdf showing the runtime and cost named `car_1.pdf`. You can use [PlannerArena](https://plannerarena.org/) or [BenchmarkPlotter](https://github.com/aorthey/ompl_benchmark_plotter) to generate the pdf. To this end, you need to convert the log file using the official ompl script at https://github.com/ompl/ompl/blob/main/scripts/ompl_benchmark_statistics.py.

