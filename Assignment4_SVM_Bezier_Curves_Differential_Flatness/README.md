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

 # Docker 
 docker run -it --rm -v $(pwd):/home/ompl -p 7000:7000 --name ompl_container my-omplapp-matplotlib


#  Benchmark
 python3 ompl_benchmark.py cfg/car_1.yaml car_1.log
 # Statistics
 python3 ompl_benchmark_statistics.py benchmark_car.log
 
 python3 ompl_benchmark_plotter/ompl_benchmark_plotter.py benchmark.db -s

# in Dockerfile folder
 docker build -t ompl .

# check with command docker images 

# start a container with in folder with your code from assignment
docker run -it --rm -v $(pwd):/home/ompl -p 7000:7000 --name ompl_container ompl:latest

# then you are in the the command line of the container and can execute python file or you connect via VS code to the runnung container 
