# Motion Planning - Programming Assignments

Repository with tasks and example cases for the motion planning class.

## Getting Started

* We will fork this repository for you and provide you with access
* Add this repository as a remote, so you can stay updated

```
git remote add upstream git@git.tu-berlin.de:imrc/teaching/motion-planning/2024_ss/assignments.git
```

* Keep an eye on the automated tests on gitlab, which tell you at the very least if your program structure matches our expectation. Grading will be done on more complicated test cases.
  
## Update Your Fork

If we add new assignments or example files, you have to update your fork with our latest changes. For that you can use the following.

```
git fetch upstream
git merge upstream/main
```

## Submission

Before the given deadline, add a tag to your code and push it

```
git tag assignment1 main
git push origin assignment1
```

(In case you need to change the tag, use the `-f` flag for both `git tag` and `git push`.)


## commands
# Run RRT algorithm for robot arm and robot env
 python3 rrt.py cfg/arm_0.yaml arm_plan_0.yaml
# Visualize results of RRT 
 python3 solutionRRT.py tree_rrt.yaml cfg/arm_0.yaml
 python3 tree_vis.py tree_rrt.yaml
# Visualize arm for different configurations
 python3 arm_vis.py cfg/arm_env_0.yaml cfg/arm_plan_0.yaml
# Check collision of arm for the states (joint angles) in different configs
 python collisions.py cfg/arm_env_0.yaml arm_plan_0.yaml collision_arm_sol.yaml