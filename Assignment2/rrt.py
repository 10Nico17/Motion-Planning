import yaml
import numpy as np
import sys
import random
import time
from nearest_neighbor import NearestNeighbor
import fcl

class Node:
    def __init__(self, config, parent=None):
        self.config = config
        self.parent = parent

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

def forward_kinematics_arm(state, L):
    '''
    Computes the forward kinematics for the arm given the state and link lengths.
    '''
    theta1, theta2, theta3 = state
    p0 = np.array([0, 0, 0])
    p1 = p0 + L[0] * np.array([np.cos(theta1), np.sin(theta1), 0])
    p2 = p1 + L[1] * np.array([np.cos(theta1 + theta2), np.sin(theta1 + theta2), 0])
    p3 = p2 + L[2] * np.array([np.cos(theta1 + theta2 + theta3), np.sin(theta1 + theta2 + theta3), 0])
    joints = [p0, p1, p2, p3]
    return joints

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

def is_collision_free(config, obstacle_objects, L):
    '''
    Checks if a given configuration is collision-free.
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

def rrt(environment, motionplanning, hyperparameters):
    '''
    RRT implementation based on the provided pseudocode.

    This function attempts to find a path for a robotic arm from a start configuration
    to a goal configuration using the Rapidly-exploring Random Tree (RRT) algorithm.
    The function takes in the environment description, motion planning parameters,
    and algorithm hyperparameters, and returns a path from the start to the goal if
    one is found within the time limit.

    Parameters:
    - environment: A dictionary containing environment settings including boundaries.
    - motionplanning: A dictionary containing the start and goal configurations and link lengths.
    - hyperparameters: A dictionary containing hyperparameters such as mu (step size), goal_eps, goal_bias, and time limit.

    Returns:
    - A list of configurations representing the path from the start to the goal.
    '''
    random.seed()  # Initialize the random number generator
    xstart = Node(motionplanning['start'])
    xgoal = Node(motionplanning['goal'])
    L = motionplanning['L']
    goal_eps = hyperparameters['goal_eps']
    goal_bias = hyperparameters['goal_bias']
    timelimit = hyperparameters['timelimit']
    mu = 0.1

    obstacle_objects = create_obstacle_objects(environment['obstacles'])

    nn = NearestNeighbor('l2')
    nn.addConfiguration(xstart.config)
    nodes = [xstart]
    edges = []
    start_time = time.time()
    last_print_time = start_time

    sample_count = 0
    node_count = 1
    finished = False
    succes = False

    best_node = None
    best_distance = float('inf')

    while not finished or (int(time.time()) - int(start_time) < timelimit):   
        
        current_time = time.time()
        if current_time - last_print_time >= 9.9:
            print(f"Elapsed time: {current_time - start_time:.2f} seconds")
            last_print_time = current_time         
        if random.random() < goal_bias:
            #print('choose goal')
            xrand = xgoal.config
        else:
            #print('random')
            xrand = [random.uniform(environment['min'][i], environment['max'][i]) for i in range(len(xstart.config))]

        #print('xrand: ', xrand)

        current_time = time.time() - start_time
        sample_count += 1
        
        # find k=1 nearest neighbour 
        nearest_config = nn.nearestK(xrand, 1)[0]
        
        xnear = next(node for node in nodes if np.array_equal(node.config, nearest_config))
        xnew = steer(xnear.config, xrand, mu)
        
        if is_collision_free(xnew, obstacle_objects, L):
            new_node = Node(xnew, xnear)
            nodes.append(new_node)
            nn.addConfiguration(xnew)
            edges.append([nodes.index(xnear), len(nodes) - 1])
            node_count += 1

            distance_to_goal = np.linalg.norm(np.array(xnew) - np.array(xgoal.config))
            if distance_to_goal < goal_eps:
                #print('xnew: ', xnew)
                #print('##### distance_to_goal: ', distance_to_goal)
                goal_node = Node(xgoal.config, new_node)
                nodes.append(goal_node)
                edges.append([nodes.index(new_node), len(nodes) - 1])
                #path = extract_path(goal_node)
                finished = True
                succes = True
                first_solution = find_shortest_path(nodes, edges, xstart.config, xgoal.config)
                len_first_solution = len(first_solution)
                print(f'first_solution: {first_solution}, len_first_solution: {len_first_solution}')

                if timelimit <= 0:
                    #print('direct solution')
                    #path1 = extract_path(goal_node)
                    #print('path1: ', path1)                    
                    path = find_shortest_path(nodes, edges, xstart.config, xgoal.config)
                    len_path = len(path)    
                    print(f'solution: {path}, len_solution: {len_path}')
                    return path, nodes, edges
                
            if distance_to_goal < best_distance:
                best_node = new_node
                best_distance = distance_to_goal

    if succes:  
        print('Outisde while')
        print('time: ', int(time.time()) - int(start_time))
        #path = extract_path(goal_node)
        path = find_shortest_path(nodes, edges, xstart.config, xgoal.config)
        len_path = len(path)    
        print(f'solution: {path}, len_solution: {len_path}')

    else:
        print("Returning the best solution found within the time limit.")
        #path = extract_path(best_node)
        path = find_shortest_path(nodes, edges, xstart.config, xgoal.config)
        len_path = len(path)    
        print(f'solution: {path}, len_solution: {len_path}')
        if np.linalg.norm(np.array(path[-1]) - np.array(xgoal.config)) < goal_eps:
            path.append(xgoal.config)
    
    return path, nodes, edges
        


def extract_path(node):

    path = []
    while node:
        path.append(node.config)
        node = node.parent
    return path[::-1]



def find_shortest_path(nodes, edges, start_config, goal_config):
    """
    Finds the shortest path from any goal node to the start node in the given tree.
    """
    config_to_node = {tuple(node.config): node for node in nodes}
    for edge in edges:
        parent_index, child_index = edge
        parent_node = nodes[parent_index]
        child_node = nodes[child_index]
        child_node.parent = parent_node
    start_node = config_to_node[tuple(start_config)]
    goal_nodes = [node for node in nodes if np.array_equal(node.config, goal_config)]
    if not goal_nodes:
        print("Goal node not found in the tree.")
        return []
    shortest_path = None
    shortest_path_length = float('inf')
    for goal_node in goal_nodes:
        path = extract_path(goal_node)
        path_length = len(path)
        if path_length < shortest_path_length:
            shortest_path = path
            shortest_path_length = path_length
    return shortest_path



def main(input_file, output_file):
    '''
    Main function to run the RRT algorithm and save the output to files.
    '''
    data = read_yaml(input_file)
    environment = data['environment']
    motionplanning = data['motionplanning']
    hyperparameters = data['hyperparameters']

    path, nodes, edges = rrt(environment, motionplanning, hyperparameters)
    print('path: ', path)

    output_data = {
        'plan': {
            'type': 'arm',
            'L': motionplanning['L'],
            'states': [list(state) for state in path]
        }
    }
    
    #print('output_data: ', output_data)
    write_yaml(output_data, output_file)

    tree_output_file = "tree_rrt.yaml"
    tree_data = {
        'nodes': [list(node.config) for node in nodes],
        'edges': edges
    }

    write_yaml(tree_data, tree_output_file)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 rrt.py <input_file> <output_file>")
        sys.exit(1)
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    main(input_file, output_file)
