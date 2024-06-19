import yaml
import numpy as np
import sys
import random
import time
from nearest_neighbor import NearestNeighbor
import fcl
from ompl import base as base
from ompl import geometric as geometric


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
    #print('Write yaml: ', file_path)
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

def forward_kinematics_arm(state, L):
    '''
    Computes the forward kinematics for the arm given the state and link lengths.
    '''
    #print('L: ', L)
    #print('L: ', type(L))
    theta1, theta2, theta3 = state
    p0 = np.array([0, 0, 0])
    p1 = p0 + L[0] * np.array([np.cos(theta1), np.sin(theta1), 0])
    p2 = p1 + L[1] * np.array([np.cos(theta1 + theta2), np.sin(theta1 + theta2), 0])
    p3 = p2 + L[2] * np.array([np.cos(theta1 + theta2 + theta3), np.sin(theta1 + theta2 + theta3), 0])
    joints = [p0, p1, p2, p3]
    return joints

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

def is_collision_free_arm(config, obstacle_objects, L):
    '''
    Checks if a given configuration is collision-free for the arm.
    '''
    joints = forward_kinematics_arm(config, L)
    collision = False
    request = fcl.CollisionRequest()
    result = fcl.CollisionResult()
    for joint in joints:
        joint_obj = create_sphere(0.05, joint)  # Create a sphere at each joint position
        for obstacle in obstacle_objects:
            ret = fcl.collide(joint_obj, obstacle, request, result)
            if ret:
                collision = True
                break
        if collision:
            break

    return not collision


def steer(xnear, xrand, mu):
    '''
    Steers from xnear towards xrand by a maximum step size mu.
    '''
    diff = np.array(xrand) - np.array(xnear)
    distance = np.linalg.norm(diff)
    if distance < mu:
        return xrand
    direction = diff / distance
    xnew = np.array(xnear) + mu * direction
    return xnew.tolist()



def is_state_valid(state):
    global obstacles, L
    #print('state: ', state)
    #print('type state: ', type(state))
    #print('state 0: ', state[0])
    #print('state 1: ', state[1])
    #print('state 2: ', state[2])
    state_list = [state[0], state[1], state[2]]
    joints = forward_kinematics_arm(state_list, L)
    #print('joints: ', joints)

    for joint in joints:
        joint_obj = create_sphere(0.05, joint)
        for obstacle in obstacles:
            if fcl.collide(joint_obj, obstacle, fcl.CollisionRequest(), fcl.CollisionResult()):
                #print('##### Collision #######')
                return False
    return True



def plan_with_ompl(environment, motionplanning, hyperparameters):
    space = base.RealVectorStateSpace(len(motionplanning['start']))
    bounds = base.RealVectorBounds(len(motionplanning['start']))
    for i in range(len(motionplanning['start'])):
        bounds.setLow(i, environment['min'][i])
        bounds.setHigh(i, environment['max'][i])
    space.setBounds(bounds)

    print(space.settings())    
    si = base.SpaceInformation(space)
    si.setStateValidityChecker(base.StateValidityCheckerFn(is_state_valid))    
    start = base.State(space)
    for i in range(len(motionplanning['start'])):
        start[i] = motionplanning['start'][i]

    goal = base.State(space)
    for i in range(len(motionplanning['goal'])):
        goal[i] = motionplanning['goal'][i]
    print('start: ', start)
    print('goal: ', goal)
    

    
    pdef = base.ProblemDefinition(si)
    pdef.setStartAndGoalStates(start, goal)    

    planner_type = hyperparameters.get('planner', 'rrt')
    if planner_type == 'rrt':
        planner = geometric.RRT(si)
        print('######### RRT ###########')
    elif planner_type == 'rrt*':
        planner = geometric.RRTstar(si)
        print('######### RRTstar ###########')
    elif planner_type == 'rrt-connect':
        planner = geometric.RRTConnect(si)
        print('######### RRTConnect ###########')
    else:
        planner = geometric.RRT(si)
        print('######### RRT ###########')

    planner.setProblemDefinition(pdef)
    planner.setup()

    time_limit = hyperparameters['timelimit']
    print('time_limit: ', time_limit)

    
    solved = planner.solve(time_limit)

    
    if solved:
        path = pdef.getSolutionPath()
        return path
    else:
        return None
    

def main(input_file, output_file):
    '''
    Main function to run the RRT algorithm and save the output to files.
    '''
    data = read_yaml(input_file)
    environment = data['environment']
    motionplanning = data['motionplanning']
    hyperparameters = data['hyperparameters']

    print("Environment:", environment)
    print("Motion Planning:", motionplanning)
    print("Hyperparameters:", hyperparameters)

    global obstacles, L
    obstacles = create_obstacle_objects(environment['obstacles'])
    L = motionplanning['L']
    #print('obstacles: ', obstacles)
    #print('L: ', L)

    path = plan_with_ompl(environment, motionplanning, hyperparameters)
    print('# path: ', path)

    states_number = path.getStateCount()
    states = path.getStates()
    '''
    state_q0 = list(map(lambda state: state[0], states))
    state_q1 = list(map(lambda state: state[1], states))
    state_q2 = list(map(lambda state: state[2], states))
    print('state_q0: ', state_q0)
    print('state_q1: ', state_q1)
    print('state_q2: ', state_q2)
    '''
    formatted_states = [[state[0], state[1], state[2]] for state in states]
    print('formatted_states: ',formatted_states)


    
    if motionplanning['type'] == 'arm':
        output_data = {
            'plan': {
                'type': 'arm',
                'L': motionplanning['L'],
                'states': formatted_states
            }
        }
    write_yaml(output_data, output_file)
    

    

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 rrt.py <input_file> <output_file>")
        sys.exit(1)
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    main(input_file, output_file)
