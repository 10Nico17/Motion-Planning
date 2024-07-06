import yaml
import numpy as np
import sys
import argparse
import random
import time
import meshcat.transformations as tf
import fcl


from ompl import base as base
from ompl import geometric as geometric
import ompl.control as control
import ompl.tools as tools
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

def create_cylinder(radius, length, transform):
    cylinder = fcl.Cylinder(radius, length)
    return fcl.CollisionObject(cylinder, transform)


def forward_kinematics_arm(state, L):
    '''
    Computes the forward kinematics for the arm given the state and link lengths.
    '''
    theta1 = state[0]
    theta2 = state[1]
    theta3 = state[2]
    p0 = np.array([0, 0, 0])
    p1 = p0 + L[0] * np.array([np.cos(theta1), np.sin(theta1), 0])
    p2 = p1 + L[1] * np.array([np.cos(theta1 + theta2), np.sin(theta1 + theta2), 0])
    p3 = p2 + L[2] * np.array([np.cos(theta1 + theta2 + theta3), np.sin(theta1 + theta2 + theta3), 0])
    joints = [p0, p1, p2, p3]
    #print('joints: ', joints)
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
            #print('create box')
            obstacle_objects.append(create_box_obstacle(obstacle))
        elif obstacle["type"] == "cylinder":
            #print('create cylinder')
            obstacle_objects.append(create_cylinder_obstacle(obstacle))
        else:
            raise ValueError(f"Unknown obstacle type: {obstacle['type']}")
    return obstacle_objects

def is_collision_free_arm(config, obstacle_objects, L):

    #print('#### config: ', config)
    joints = forward_kinematics_arm(config, L)
    #print('joints: ', joints)
    collision = False
    request = fcl.CollisionRequest()
    result = fcl.CollisionResult()
    for joint in joints:
        joint_obj = create_sphere(0.05, joint)  # Create a sphere at each joint position
        #print('joint: ', joint)
        for obstacle in obstacle_objects:
            ret = fcl.collide(joint_obj, obstacle, request, result)
            if ret:
                print('################## Colllision ###################')
                collision = True
                break
        if collision:
            break
        else:
            print('no collision')

    return not collision


def create_arm_cylinders(state, L, r=0.04, lz=1.0):
    '''
    Creates cylinder objects for each arm segment.
    '''
    print('######### state: ', state)
    theta_1 = state[0]
    theta_2 = state[1]
    theta_3 = state[2]

    x_1 = L[0]/2*np.cos(theta_1)
    y_1 = L[0]/2*np.sin(theta_1)
   
    x_2 = L[0]*np.cos(theta_1) + L[1]/2*np.cos(theta_1+theta_2)
    y_2 = L[0]*np.sin(theta_1) + L[1]/2*np.sin(theta_1+theta_2)    

    x_3 = L[0]*np.cos(theta_1) + L[1]*np.cos(theta_1+theta_2) + L[2]/2*np.cos(theta_1+theta_2+theta_3)
    y_3 = L[0]*np.sin(theta_1) + L[1]*np.sin(theta_1+theta_2) + L[2]/2*np.sin(theta_1+theta_2+theta_3)

    offset = np.pi/2
    T1 = tf.translation_matrix([x_1, y_1, 0]).dot(
            tf.euler_matrix(np.pi/2, 0, offset + theta_1))
    t1 = fcl.Transform(T1[0:3,0:3], T1[0:3,3]) # R, t
    
    T2 = tf.translation_matrix([x_2, y_2, 0]).dot(
            tf.euler_matrix(np.pi/2, 0, offset + theta_1+theta_2))
    t2 = fcl.Transform(T2[0:3,0:3], T2[0:3,3]) # R, t
    
    T3 = tf.translation_matrix([x_3, y_3, 0]).dot(
            tf.euler_matrix(np.pi/2, 0, offset + theta_1 + theta_2+ theta_3))
    t3 = fcl.Transform(T3[0:3,0:3], T3[0:3,3]) # R, t
    
    cylinders = [
        create_cylinder(r, L[0], t1),
        create_cylinder(r, L[1], t2),
        create_cylinder(r, L[2], t3)
    ]
    
    return cylinders



def is_collision_free_arm(config, obstacle_objects, L):

    cylinders = create_arm_cylinders(config, L)
    print('cylinders: ', cylinders)
    collision = False
    request = fcl.CollisionRequest()
    result = fcl.CollisionResult()
    
    for cylinder in cylinders:
        for obstacle in obstacle_objects:
            ret = fcl.collide(cylinder, obstacle, request, result)
            if ret:
                #print('### collision ####')
                collision = True
                break
        if collision:
            break

    return not collision


def is_collision_free_car(config, obstacle_objects, L, W):
    '''
    Checks if a given configuration is collision-free for the car.
    '''
    #print('config: ', config)
    collision = False
    car_box = create_car_box(L, W, config)
    request = fcl.CollisionRequest()
    result = fcl.CollisionResult()
    for obstacle in obstacle_objects:
        ret = fcl.collide(car_box, obstacle, request, result)
        #print('ret: ', ret)
        if ret:
            collision = True
            break
    return not collision

def statePropagator(start, control, duration, target, L=3):

    x = start.getX()
    y = start.getY()
    theta = start.getYaw() 
    s = control[0]
    phi = control[1]

    x_new = x + s * np.cos(theta) * duration
    y_new = y + s * np.sin(theta) * duration
    theta_new = theta + (s / L) * np.tan(phi) * duration
    target.setX(x_new)
    target.setY(y_new)
    target.setYaw(theta_new)
    #print('target: ', target.getX(), target.getY(), target.getYaw()) 
    return target


def is_state_valid(state):
    global obstacles, L
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

def is_state_valid_car(state):
    '''
    Checks if a given configuration is collision-free for the car.
    '''
    
    global obstacles, L, W, actions
    collision = False
    x = state.getX()
    y = state.getY()
    yaw = state.getYaw()    
    car_box = create_car_box(L, W, [x,y,yaw])
    
    request = fcl.CollisionRequest()
    result = fcl.CollisionResult()
    for obstacle in obstacles:
        ret = fcl.collide(car_box, obstacle, request, result)
        #print('ret: ', ret)
        if ret:
            collision = True
            break  
    return not collision


def get_planner_data(si, planner):

    planner_data = base.PlannerData(si)
    planner.getPlannerData(planner_data)
    vertices = [planner_data.getVertex(i).getState() for i in range(planner_data.numVertices())]
    edges = []

    for v1 in range(planner_data.numVertices()):
        edge_map = base.mapUintToPlannerDataEdge()  # Create an empty edge map
        planner_data.getEdges(v1, edge_map)  # Populate the edge map with edges for vertex v1

        # Iterate over the edge_map to access edges
        keys = edge_map.keys()  # Get all keys (v2 vertices)
        for v2 in keys:
            edge = edge_map[v2]  # Access edge using the key
            #print(f"Edge from vertex {v1} to vertex {v2} with edge data: {edge}")
            edges.append([v1, v2])  # Kante in der Form [v1, v2] in der Liste speichern

    nodes = [[vertex[0], vertex[1], vertex[2]] for vertex in vertices]

    return nodes, edges



def plan_with_ompl_arm(environment, motionplanning, hyperparameters):
    space = base.RealVectorStateSpace(len(motionplanning['start']))
    
    
    '''
    bounds = base.RealVectorBounds(len(motionplanning['start']))
    for i in range(len(motionplanning['start'])):
        bounds.setLow(i, environment['min'][i])
        bounds.setHigh(i, environment['max'][i])
    '''
    

    bounds = base.RealVectorBounds(3)
    #bounds.setLow(-3.14)  # Untere Grenze für alle Dimensionen
    #bounds.setHigh(3.14)  # Obere Grenze für alle Dimensionen
    bounds.setLow(-np.pi)  # Untere Grenze für alle Dimensionen
    bounds.setHigh(np.pi)  # Obere Grenze für alle Dimensionen
    

    space.setBounds(bounds)
    #print(space.settings())

    
    #print(space.settings())    
    si = base.SpaceInformation(space)
    si.setStateValidityChecker(base.StateValidityCheckerFn(is_state_valid))    
    start = base.State(space)
    for i in range(len(motionplanning['start'])):
        start[i] = motionplanning['start'][i]

    #print('start: ', start)
    collision_check = is_collision_free_arm(start, obstacles, L)
    #print('collision_check: ', collision_check)
    
    goal = base.State(space)
    for i in range(len(motionplanning['goal'])):
        goal[i] = motionplanning['goal'][i]
    
    if 'goal_eps' in hyperparameters:
        goal_eps = hyperparameters['goal_eps']
    else :
        goal_eps = 0.1

    pdef = base.ProblemDefinition(si)
    pdef.setStartAndGoalStates(start, goal, goal_eps)    

    planner_type = hyperparameters.get('planner', 'rrt')
    if planner_type == 'rrt':
        planner = geometric.RRT(si)
        if 'goal_bias' in hyperparameters:
            planner.setGoalBias(hyperparameters['goal_bias'])
        print('######### RRT ###########')
    elif planner_type == 'rrt*':
        planner = geometric.RRTstar(si)
        if 'goal_bias' in hyperparameters:
            planner.setGoalBias(hyperparameters['goal_bias'])
        print('######### RRTstar ###########')
    elif planner_type == 'rrt-connect':
        planner = geometric.RRTConnect(si)
        print('######### RRTConnect ###########')
    else:
        planner = geometric.RRT(si)
        if 'goal_bias' in hyperparameters:
            planner.setGoalBias(hyperparameters['goal_bias'])
        print('######### RRT ###########')


    planner.setProblemDefinition(pdef)
    planner.setup()
    time_limit = hyperparameters['timelimit']
    solved = planner.solve(time_limit)

    if solved:
        path = pdef.getSolutionPath()
        nodes, edges = get_planner_data(si, planner)       
        return path, nodes, edges
    else:
        return None, None, None
    
    



def getCost(si):
    return base.PathLengthOptimizationObjective(si)


def plan_with_ompl_car(environment, motionplanning, hyperparameters):
    space = base.SE2StateSpace()
    bounds = base.RealVectorBounds(2)
    for i in range(2):
        bounds.setLow(i, environment['min'][i])
        bounds.setHigh(i, environment['max'][i])
    space.setBounds(bounds)

    uspace = control.RealVectorControlSpace(space, 2)
    ubounds = base.RealVectorBounds(2)

    ubounds.setLow(0, -0.5)             # Geschwindigkeit
    ubounds.setHigh(0, 2.0)
    ubounds.setLow(1, -np.pi/6)         # Lenkwinkel
    ubounds.setHigh(1, np.pi/6)    
    uspace.setBounds(ubounds)
    
    si = control.SpaceInformation(space, uspace)
    si.setStateValidityChecker(base.StateValidityCheckerFn(is_state_valid_car))
    dt = motionplanning['dt']
    si.setStatePropagator(control.StatePropagatorFn(statePropagator))
    si.setPropagationStepSize(dt)

    start = base.State(space)
    start[0] = motionplanning['start'][0]
    start[1] = motionplanning['start'][1]
    start[2] = motionplanning['start'][2]
    goal = base.State(space)
    goal[0] = motionplanning['goal'][0]
    goal[1] = motionplanning['goal'][1]
    goal[2] = motionplanning['goal'][2]


    if 'goal_eps' in hyperparameters:
        goal_eps = hyperparameters['goal_eps']
    else :
        goal_eps = 0.1


    pdef = base.ProblemDefinition(si)
    pdef.setStartAndGoalStates(start, goal, goal_eps)
    pdef.setOptimizationObjective(getCost(si))

    planner_type = hyperparameters.get('planner', 'sst')
    print('planner_type: ', planner_type)
    if planner_type == 'sst':
        planner = control.SST(si)
        if 'goal_bias' in hyperparameters:
            planner.setGoalBias(hyperparameters['goal_bias'])
        print('######### SST ###########')
    elif planner_type == 'rrt':
        planner = control.RRT(si)
        if 'goal_bias' in hyperparameters:
            planner.setGoalBias(hyperparameters['goal_bias'])
        print('######### RRT ###########')

    planner.setProblemDefinition(pdef)
    planner.setup()

    time_limit = hyperparameters['timelimit']
    solved = planner.solve(time_limit)

    nodes, edges = None, None


    if solved:
        path = pdef.getSolutionPath()   
        states = path.getStates()
        planner_data = base.PlannerData(si)
        planner.getPlannerData(planner_data)
        vertices = [planner_data.getVertex(i).getState() for i in range(planner_data.numVertices())]
        nodes = [[vertex.getX(), vertex.getY(), vertex.getYaw()] for vertex in vertices]

        edges = []

        for v1 in range(planner_data.numVertices()):
            edge_map = base.mapUintToPlannerDataEdge()  
            planner_data.getEdges(v1, edge_map)  

            keys = edge_map.keys()  
            for v2 in keys:
                edge = edge_map[v2]  
                edges.append([v1, v2])  

        return path, nodes, edges
    else:
        return None, None, None
    

    

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

    if planning_type == 'arm':
        #print('call arm')
        path, nodes, edges = plan_with_ompl_arm(environment, motionplanning, hyperparameters)
        #print('#### path: ', path)
        #path.interpolate()
    elif planning_type == 'car':
        W = motionplanning['W']
        path, nodes, edges = plan_with_ompl_car(environment, motionplanning, hyperparameters)
        #print('#### path: ', path)
        #print('######################################')
        path.interpolate()
        #print('#### path2: ', path)
    else:
        raise ValueError(f"Unknown motion planning type: {planning_type}")
    
   
    
    if export_planner_data:
        tree_data = {
         'nodes': [list(node) for node in nodes],
         'edges': edges
        }
        write_yaml(tree_data, export_planner_data)    

    
    states_number = path.getStateCount()
    states = path.getStates()
       
    if motionplanning['type'] == 'arm':
        formatted_states = [[state[0], state[1], state[2]] for state in states]
        print('########## formatted_states: ', formatted_states)     
        output_data = {
            'plan': {
                'type': 'arm',
                'L': motionplanning['L'],
                'states': formatted_states
            }
        }
    elif motionplanning['type'] == 'car':
        formatted_states = [[state.getX(), state.getY(), state.getYaw()] for state in states]
        controls = path.getControls()
        formatted_actions = [[control[0], control[1]] for control in controls]
        print('########## formatted_states: ', formatted_states)     
        print('########## formatted_actions: ', formatted_actions)        
        output_data = {
            'plan': {
                'type': 'car',
                'dt': motionplanning['dt'],
                'L': motionplanning['L'],
                'W': motionplanning['W'],
                'H': 1,  # Adding height for the car
                'states': formatted_states,
                'actions': formatted_actions
            }
        }  
             

    #print('output_file: ', output_file)
    write_yaml(output_data, output_file)
    
 
    

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run OMPL planner")
    parser.add_argument("input_file", type=str, help="Input YAML file for the planner")
    parser.add_argument("output_file", type=str, help="Output YAML file for the planner")
    parser.add_argument("--export-planner-data", type=str, help="Export planner data to a YAML file")
    
    args = parser.parse_args()
    main(args.input_file, args.output_file, args.export_planner_data)
