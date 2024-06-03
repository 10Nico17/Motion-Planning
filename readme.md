# 1. Env:  
** Visu robot arm three joints in XY-plane**  

```
python3 arm_vis.py cfg/arm_env_0.yaml cfg/arm_plan_0.yaml
```

<div align="center">
  <img src="Assignment2/arm_visu.png" alt="License plate dataset"  width="640" height="480">
</div>

# Example:   
**2. RRT motion planning for robot arm from start to goal configuration in C-space with collision checking :**  

**Start RRT**
```
python3 rrt.py cfg/arm_0.yaml arm_plan_0.yaml
```

**Visualize tree and path**
```
 python3 solutionRRT.py tree_rrt.yaml cfg/arm_0.yaml

```

**Verify that all configs are collision free**
```
 python3 solutionRRT.py tree_rrt.yaml cfg/arm_0.yaml

```


<div align="center">
  <img src="Assignment2/Test1.png" alt="License plate dataset"  width="640" height="480">
</div>


<div align="center">
  <img src="Assignment2/Test2.png" alt="License plate dataset"  width="640" height="480">
</div>


** Visu car kinodynamic motion planning**  


<div align="center">
  <img src="Assignment2/car_planning.png" alt="License plate dataset"  width="640" height="480">
</div>


<div align="center">
  <img src="Assignment2/path_planning_car.png" alt="License plate dataset"  width="640" height="480">
</div>
