# Motion Planning - Programming Assignment 2

Due: 6/12/24 (midnight)

The goal of this assignment is to implement the rapidly exploring random tree (RRT) algorithm such that it works for the car and manipulator robot.

## Tasks

### Nearest Neighbor (2 pt)

We need a nearest neighbor data structure that allows incremental updates.

Implement a naive nearest-neighbor datastructure that provides the following functions

- `addConfiguration(q)`: adds a given configuration to the datastructure
- `nearestK(q, k)`: returns up to k closest configurations to q that are in the datastructure (sorted by distance)
- `nearestR(q, r)`: returns all configurations in the datastructure that are within radius r of configuration q (sorted by distance)

Your datastructure should support difference distance metrics, at least `l2` (the Euclidean norm), `se2` (for the car, sum of l2-norm of position and angular metric for the orientation), and `angles` (for the manipulator, treat each component as an angle).

Write a script such that your datastructure can be tested independently.

```
python nearest_neighbor.py cfg/nn_0.yaml nn_sol.yaml
```

creates a file, `nn_sol.yaml` with the following content:

```yaml
results:
  - - [0.0, 1.5]
  - []
```

### Tree Visualization (Not graded)

Use your knowledge about meshcat (from the first assignment) to write a script that can visualize the nodes and edges of a tree. This part is optional (as in, it will not be graded). However, the visualization may help you with implementing, debugging, and verifying your RRT implementation in the next step.

Your visualization should take another yaml file as input:

```
python3 tree_vis.py cfg/tree.yaml
```

### RRT - Manipulator case (4 pt)

Implement RRT for the geometric case for the manipulator robot. Your code should take a single yaml file as input (that describes the motion planning problem) and output the solution in the same format that we have used in the first assignment:

```
python3 rrt.py cfg/arm_0.yaml arm_plan_0.yaml
```

Note that the input file has three parts:

1. The `environment` key, which is identical to the `*_env_*.yaml` files from the first assignment;
2. The `motionplanning` key, which identifies the robot type, start- and goal states; and
3. The `hyperparameters` key, which can be used to change settings of your algorithm.

For the hyperparameters, you should support timelimit, goal_bias, and goal_eps, as described in `cfg/arm_0.yaml`.

### RRT - Car case (4 pt)

Extend your implementation to also support the kinodynamic case for the car-like robot using the following control limits: s ∈ [−0.5, 2] m/s and φ ∈ [−π/6, π/6] rad.
The same script (`rrt.py`) should be able to work for both the manipulator and the car-like robot. For example, the following command should output a solution file similar to the one from the first assignment:

```
python3 rrt.py cfg/car_0.yaml car_plan_0.yaml
```
