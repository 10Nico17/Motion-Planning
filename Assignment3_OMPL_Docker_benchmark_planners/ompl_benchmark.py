import yaml
import numpy as np
import sys
import argparse
import random
import time
import os
import subprocess
from functools import partial


from ompl import base as base
from ompl import geometric as geometric
import ompl.control as control
import ompl.tools as tools

import fcl
from nearest_neighbor import NearestNeighbor

def read_yaml(file_path):
    '''
    Reads a YAML file and returns the data as a dictionary.
    '''
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def write_yaml(data, file_path):
    '''
    Writes a dictionary to a YAML file.
    '''
    with open(file_path, 'w') as file:
        yaml.dump(data, file, default_flow_style=None, sort_keys=False)

def create_box_obstacle(obstacle):
    '''
    Creates a box obstacle for collision checking.
    '''
    size = obstacle['size']
    translation = obstacle['pos']
    box = fcl.Box(size[0], size[1], size[2])
    tf = fcl.Transform(np.eye(3), translation)
    return fcl.CollisionObject(box, tf)

def create_cylinder_obstacle(obstacle):
    '''
    Creates a cylinder obstacle for collision checking.
    '''
    radius = obstacle['r']
    length = obstacle['lz']
    translation = obstacle['pos']
    q = obstacle['q']  # Quaternion (w, x, y, z)
    cylinder = fcl.Cylinder(radius, length)
    tf = fcl.Transform(q, translation)
    return fcl.CollisionObject(cylinder, tf)

def create_sphere(radius, position):
    '''
    Creates a sphere for collision checking.
    '''
    sphere = fcl.Sphere(radius)
    tf = fcl.Transform(np.eye(3), position)
    return fcl.CollisionObject(sphere, tf)


def create_car_box(L, W, state):
    x, y, theta = state
    box = fcl.Box(L, W, 1.0)  # Assuming height of the car is 1.0
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0, 0, 1]
    ])
    translation = np.array([x, y, 0.5])  # Assuming the car is centered at z = 0.5
    tf = fcl.Transform(rotation_matrix, translation)
    return fcl.CollisionObject(box, tf)

def create_obstacle_objects(obstacles):
    '''
    Creates obstacle objects for collision checking.
    '''
    obstacle_objects = []
    for obstacle in obstacles:
        if obstacle["type"] == "box":
            obstacle_objects.append(create_box_obstacle(obstacle))
        elif obstacle["type"] == "cylinder":
            obstacle_objects.append(create_cylinder_obstacle(obstacle))
        else:
            raise ValueError(f"Unknown obstacle type: {obstacle['type']}")
    return obstacle_objects


def is_collision_free_car(config, obstacle_objects, L, W):
    '''
    Checks if a given configuration is collision-free for the car.
    '''
    collision = False
    car_box = create_car_box(L, W, config)
    request = fcl.CollisionRequest()
    result = fcl.CollisionResult()
    for obstacle in obstacle_objects:
        ret = fcl.collide(car_box, obstacle, request, result)
        if ret:
            collision = True
            break
    return not collision

def statePropagator(start, control, duration, target):
    global L
    u1 = control[0]
    u2 = control[1]
    x = start.getX()
    y = start.getY()
    theta = start.getYaw()

    x_n = u1 * np.cos(theta)
    y_n = u1 * np.sin(theta)
    theta_n = (u1 / L) * np.tan(u2)

    target.setX(x + duration * x_n)
    target.setY(y + duration * y_n)
    target.setYaw(theta + duration * theta_n)
    return target


def is_state_valid_car(state):
    '''
    Checks if a given configuration is collision-free for the car.
    '''
    global obstacles, L, W, actions
    collision = False
    x = state.getX()
    y = state.getY()
    yaw = state.getYaw()    
    car_box = create_car_box(L, W, [x, y, yaw])
    
    
    request = fcl.CollisionRequest()
    result = fcl.CollisionResult()
    for obstacle in obstacles:
        ret = fcl.collide(car_box, obstacle, request, result)
        if ret:
            collision = True
            break  
    
    return not collision

def getCost(si):
    return base.PathLengthOptimizationObjective(si)


def benchmark_car(environment, motionplanning, hyperparameters, output_file):

    timelimit = hyperparameters["timelimit"]
    goal_bias = hyperparameters["goal_bias"]
    goal_eps = hyperparameters["goal_eps"]


    min_border = environment["min"]
    max_border = environment["max"]

    space = base.SE2StateSpace()

    bounds = base.RealVectorBounds(2)
    bounds.setLow(0, min_border[0])
    bounds.setHigh(0, max_border[0])
    bounds.setLow(1, min_border[1])
    bounds.setHigh(1, max_border[1])
    space.setBounds(bounds)

    uspace = control.RealVectorControlSpace(space, 2)
    ubounds = base.RealVectorBounds(2)
    ubounds.setLow(0, -0.5)
    ubounds.setHigh(0, +2)
    ubounds.setLow(1, -np.pi/6)
    ubounds.setHigh(1, +np.pi/6)
    uspace.setBounds(ubounds)



    dt = motionplanning['dt']    
    si = control.SpaceInformation(space, uspace)
    si.setStateValidityChecker(base.StateValidityCheckerFn(is_state_valid_car))
    si.setStatePropagator(control.StatePropagatorFn(statePropagator))
    si.setPropagationStepSize(dt)
    si.setup()

    ########################################
    # Setup Start and Goal
    ########################################

    start = base.State(space)
    start[0] = motionplanning['start'][0]
    start[1] = motionplanning['start'][1]
    start[2] = motionplanning['start'][2]
    goal = base.State(space)
    goal[0] = motionplanning['goal'][0]
    goal[1] = motionplanning['goal'][1]
    goal[2] = motionplanning['goal'][2]



    goal_region = base.GoalRegion(si)
    goal_region.setThreshold(goal_eps) 

    goal_state = base.GoalState(si)
    goal_state.setState(goal)
    goal_state.setThreshold(goal_eps) 


    ########################################
    # (4) Setup problem definition
    ########################################

    si = control.SpaceInformation(space, uspace)
    si.setStateValidityChecker(base.StateValidityCheckerFn(partial(is_state_valid_car)))
    si.setStatePropagator(control.StatePropagatorFn(statePropagator))

    propagation_dt = motionplanning["dt"]
    si.setPropagationStepSize(propagation_dt)
    si.setup()

    start = motionplanning["start"]
    goal = motionplanning["goal"]

    start_config = base.State(space)
    start_config[0] = start[0]
    start_config[1] = start[1]
    start_config[2] = start[2]

    goal_config = base.State(space)
    goal_config[0] = goal[0]
    goal_config[1] = goal[1]
    goal_config[2] = goal[2]

    goal_region = base.GoalRegion(si)
    goal_region.setThreshold(goal_eps)  
    goal_state = base.GoalState(si)
    goal_state.setState(goal_config)
    goal_state.setThreshold(goal_eps) 


    ########################################
    # Setup Planner for different type of sampler
    ########################################


    ss = control.SimpleSetup(uspace)
    ss.setOptimizationObjective(getCost(si))
    ss.setStartAndGoalStates(start_config, goal_config)
    ss.setStatePropagator(control.StatePropagatorFn(statePropagator))
    si = ss.getSpaceInformation()
    si.setStateValidityChecker(base.StateValidityCheckerFn(partial(is_state_valid_car)))
    si.setPropagationStepSize(propagation_dt)


    benchmark = tools.Benchmark(ss)


    uniform_sampler_allocator = base.ValidStateSamplerAllocator(base.UniformValidStateSampler)
    si.setValidStateSamplerAllocator(uniform_sampler_allocator)
    if hyperparameters["planner"] == "sst":
        planner_0 = control.SST(si)
    elif hyperparameters["planner"] == "rrt":
        planner_0 = control.RRT(si)
    else:
        planner_0 = control.RRT(si)
    name = 'Uniform_SST'
    planner_0.setName(name)
    planner_0.setGoalBias(goal_bias)
    benchmark.addPlanner(planner_0)



    gaussian_sampler_allocator = base.ValidStateSamplerAllocator(base.GaussianValidStateSampler)
    si.setValidStateSamplerAllocator(gaussian_sampler_allocator)
    if hyperparameters["planner"] == "sst":
        planner_1 = control.SST(si)
    elif hyperparameters["planner"] == "rrt":
        planner_1 = control.RRT(si)
    else:
        planner_1 = control.RRT(si)
    name = 'Gaussian_SST'
    planner_1.setName(name)
    planner_1.setGoalBias(goal_bias)
    benchmark.addPlanner(planner_1)


    bridge_sampler_allocator = base.ValidStateSamplerAllocator(base.BridgeTestValidStateSampler)
    si.setValidStateSamplerAllocator(bridge_sampler_allocator)
    if hyperparameters["planner"] == "sst":
        planner_2 = control.SST(si)
    elif hyperparameters["planner"] == "rrt":
        planner_2 = control.RRT(si)
    else:
        planner_2 = control.RRT(si)
    name = 'Bridge_SST'
    planner_2.setName(name)
    planner_2.setGoalBias(goal_bias)
    benchmark.addPlanner(planner_2)
    

    obstacle_sampler_allocator = base.ValidStateSamplerAllocator(base.ObstacleBasedValidStateSampler)
    si.setValidStateSamplerAllocator(obstacle_sampler_allocator)
    if hyperparameters["planner"] == "sst":
        planner_3 = control.SST(si)
    elif hyperparameters["planner"] == "rrt":
        planner_3 = control.RRT(si)
    else:
        planner_3 = control.RRT(si)
    name = 'Obstacle_SST'
    planner_3.setName(name)
    planner_3.setGoalBias(goal_bias)
    benchmark.addPlanner(planner_3)



    ########################################
    # Start benchmark
    ########################################

    request = benchmark.Request()
    request.maxTime = timelimit
    request.maxMem = 2048
    request.runCount = 3
    request.displayProgress = True

    benchmark.benchmark(request)
    result = benchmark.getRecordedExperimentData()
    benchmark.saveResultsToFile(output_file)
    print(f"Results saved to {output_file}")
    benchmark.saveResultsToFile("benchmark.log")




def convert_log_to_db(log_file):
    '''
    Converts the benchmark log file to a SQLite database.
    '''
    cmd = ["python3", "ompl_benchmark_statistics.py", log_file]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"Error converting log to DB: {result.stderr}")
    else:
        print("Log file successfully converted to DB.")

def create_pdf_from_db(db_file, pdf_file):
    '''
    Creates a PDF from the SQLite database.

    '''

    cmd = ["python3", "ompl_benchmark_plotter/ompl_benchmark_plotter.py", db_file, "-o", pdf_file, "--title-name", "SST compare sampling"]

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"Error creating PDF from DB: {result.stderr}")
    else:
        print("PDF successfully created from DB.")



def main(input_file, output_file, export_planner_data=None):
    '''
    Main function to run the RRT algorithm and save the output to files.
    '''

    data = read_yaml(input_file)
    environment = data['environment']
    motionplanning = data['motionplanning']
    hyperparameters = data['hyperparameters']

    global obstacles, L, W
    obstacles = create_obstacle_objects(environment['obstacles'])
    L = motionplanning['L']
    planning_type = motionplanning['type']


    if planning_type == 'car':
        W = motionplanning['W']
        benchmark_car(environment, motionplanning, hyperparameters, output_file)
    else:
        raise ValueError(f"Unknown motion planning type: {planning_type}")

    convert_log_to_db(output_file)
    pdf_file = os.path.splitext(output_file)[0] + '.pdf'
    create_pdf_from_db('benchmark.db', pdf_file)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run OMPL planner")
    parser.add_argument("input_file", type=str, help="Input YAML file for the planner")
    parser.add_argument("output_file", type=str, help="Output YAML file for the planner")
    parser.add_argument("--export-planner-data", type=str, help="Export planner data to a YAML file")
 
    args = parser.parse_args()
    main(args.input_file, args.output_file, args.export_planner_data)
