import yaml
import numpy as np
import sys
import random
import time
from nearest_neighbor import NearestNeighbor
import fcl

class Node:
    def __init__(self, config, parent=None, control=None):
        self.config = config
        self.parent = parent
        self.control = control  # Control inputs (only for car-like robot)

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

def forward_dynamics_car(state, action, dt):
    x, y, theta = state
    v, omega = action

    # Simple bicycle model for forward dynamics
    x_new = x + v * np.cos(theta) * dt
    y_new = y + v * np.sin(theta) * dt
    theta_new = theta + omega * dt

    return np.array([x_new, y_new, theta_new])

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


def steer_car(xnear, xrand, dt, min_speed, max_speed, min_steering, max_steering, k):
    """
    Steers the car from xnear towards xrand under the car's dynamic constraints.
    Propagates the system k times with random controls and returns the closest state to xrand along with the control used.
    """
    best_state = None
    best_control = None
    best_distance = float('inf')
    
    for _ in range(k):
        v = random.uniform(min_speed, max_speed)
        omega = random.uniform(min_steering, max_steering)
        xnew = xnear
        for _ in range(k):
            xnew = forward_dynamics_car(xnew, [v, omega], dt)
        
        current_distance = np.linalg.norm(np.array(xnew) - np.array(xrand))
        
        if current_distance < best_distance:
            best_state = xnew.tolist()
            best_control = [v, omega]
            best_distance = current_distance
    
    return best_state, best_control




def rrt(environment, motionplanning, hyperparameters):
    '''
    RRT implementation based on the provided pseudocode.

    This function attempts to find a path for a robotic arm or a car-like robot from a start configuration
    to a goal configuration using the Rapidly-exploring Random Tree (RRT) algorithm.
    The function takes in the environment description, motion planning parameters,
    and algorithm hyperparameters, and returns a path from the start to the goal if
    one is found within the time limit.

    Parameters:
    - environment: A dictionary containing environment settings including boundaries.
    - motionplanning: A dictionary containing the start and goal configurations and link lengths or car dimensions.
    - hyperparameters: A dictionary containing hyperparameters such as mu (step size), goal_eps, goal_bias, and time limit.

    Returns:
    - A list of configurations representing the path from the start to the goal.
    '''
    random.seed()  # Initialize the random number generator
    xstart = Node(motionplanning['start'])
    xgoal = Node(motionplanning['goal'])
    goal_eps = hyperparameters['goal_eps']
    goal_bias = hyperparameters['goal_bias']
    timelimit = hyperparameters['timelimit']
    mu = 0.2

    obstacle_objects = create_obstacle_objects(environment['obstacles'])
    #print('obstacle_objects: ', obstacle_objects)

    nn = NearestNeighbor('l2')
    nn.addConfiguration(xstart.config)
    nodes = [xstart]
    edges = []
    start_time = time.time()
    last_print_time = start_time

    sample_count = 0
    node_count = 1
    finished = False
    success = False

    best_node = None
    best_distance = float('inf')

    first_solution_length = None

    #while (timelimit > 0 and (time.time() - start_time) < timelimit) or (timelimit == 0 and not finished):           
    while (timelimit > 0 and (time.time() - start_time) < timelimit and not finished) or (timelimit == 0 and not finished):
    
        current_time = time.time()
        if current_time - last_print_time >= 10:
            print(f"Elapsed time: {current_time - start_time:.2f} seconds")
            last_print_time = current_time         
        if random.random() < goal_bias:
            xrand = xgoal.config
        else:
            xrand = [random.uniform(environment['min'][i], environment['max'][i]) for i in range(len(xstart.config))]

        #print('xrand: ', xrand)    
        nearest_config = nn.nearestK(xrand, 1)[0]
        
        xnear = next(node for node in nodes if np.array_equal(node.config, nearest_config))
        
        if motionplanning['type'] == 'arm':
            xnew = steer(xnear.config, xrand, mu)
            if is_collision_free_arm(xnew, obstacle_objects, motionplanning['L']):
                new_node = Node(xnew, xnear)
                nodes.append(new_node)
                nn.addConfiguration(xnew)
                edges.append([nodes.index(xnear), len(nodes) - 1])
                node_count += 1

                distance_to_goal = np.linalg.norm(np.array(xnew) - np.array(xgoal.config))
                if distance_to_goal < goal_eps:
                    goal_node = Node(xgoal.config, new_node)
                    nodes.append(goal_node)
                    edges.append([nodes.index(new_node), len(nodes) - 1])
                    finished = True
                    success = True
                    first_solution = find_shortest_path(nodes, edges, xstart.config, xgoal.config)
                    first_solution_length = len(first_solution)
                    #print(f'First solution length: {first_solution_length}')

                    if timelimit <= 0:
                        path = find_shortest_path(nodes, edges, xstart.config, xgoal.config)
                        final_solution_length = len(path)
                        print(f'Final solution length: {final_solution_length}')
                        return path, nodes, edges
                
                if distance_to_goal < best_distance:
                    best_node = new_node
                    best_distance = distance_to_goal

        elif motionplanning['type'] == 'car':
            dt = motionplanning['dt']
            L = motionplanning['L']
            W = motionplanning['W']
            k = 5
            min_speed = -0.5
            max_speed = 2
            min_steering = - np.pi / 6
            max_steering = np.pi / 6
            xnew, control = steer_car(xnear.config, xrand, dt, min_speed, max_speed, min_steering, max_steering, k)

            if is_collision_free_car(xnew, obstacle_objects, L, W):
                #print('##### no collision ##### ')
                new_node = Node(xnew, xnear, control)
                nodes.append(new_node)
                nn.addConfiguration(xnew)
                edges.append([nodes.index(xnear), len(nodes) - 1])
                node_count += 1

                distance_to_goal = np.linalg.norm(np.array(xnew) - np.array(xgoal.config))
                if distance_to_goal < goal_eps:
                    print('######### Reached ############')
                    goal_node = Node(xgoal.config, new_node)
                    nodes.append(goal_node)
                    edges.append([nodes.index(new_node), len(nodes) - 1])
                    finished = True
                    success = True
                    first_solution = find_shortest_path(nodes, edges, xstart.config, xgoal.config)
                    first_solution_length = len(first_solution)
                    #print(f'First solution length: {first_solution_length}')

                    if timelimit <= 0:
                        path = find_shortest_path(nodes, edges, xstart.config, xgoal.config)
                        final_solution_length = len(path)
                        #print(f'Final solution length: {final_solution_length}')
                        return path, nodes, edges

                if distance_to_goal < best_distance:
                    best_node = new_node
                    best_distance = distance_to_goal
            else:
                pass
                #print('collision')
            

    if success:  
        path = find_shortest_path(nodes, edges, xstart.config, xgoal.config)
        final_solution_length = len(path)
        return path, nodes, edges
    else:
        print('Goal Node not reached: no path and tree returned')   
        return None, None, None

def extract_path(node):
    """
    Extracts the path from the given node to the start node using the parent references.
    """
    path = []
    while node:
        path.append(node.config)
        node = node.parent
    return path[::-1]


def extract_path_with_controls(node):
    """
    Extracts the path and corresponding controls from the given node to the start node using the parent references.
    """
    path = []
    controls = []
    while node:
        path.append(node.config)
        #print('node.config: ', node.config)
        if node.control is not None:
            controls.append(node.control)
            #print('node.control: ', node.control)

        node = node.parent
    return path[::-1], controls[::-1]

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
        #print("Goal node not found in the tree.")
        return []
    shortest_path = None
    shortest_path_length = float('inf')
    for goal_node in goal_nodes:
        path = extract_path(goal_node)
        #path = extract_path_with_controls(goal_node)
        path_length = len(path)
        if path_length < shortest_path_length:
            shortest_path = path
            shortest_path_length = path_length
    return shortest_path

def convert_numpy_to_list(data):
    """
    Recursively convert numpy arrays and scalars to native Python lists and floats.
    """
    if isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, np.generic):
        return data.item()
    elif isinstance(data, list):
        return [convert_numpy_to_list(item) for item in data]
    elif isinstance(data, dict):
        return {key: convert_numpy_to_list(value) for key, value in data.items()}
    else:
        return data


def main(input_file, output_file):
    '''
    Main function to run the RRT algorithm and save the output to files.
    '''
    data = read_yaml(input_file)
    environment = data['environment']
    motionplanning = data['motionplanning']
    hyperparameters = data['hyperparameters']

    path, nodes, edges = rrt(environment, motionplanning, hyperparameters)
    print('final path: ', path)

    if motionplanning['type'] == 'arm':
        states = [list(motionplanning['start'])] if path is None else [list(state) for state in path]
        output_data = {
            'plan': {
                'type': 'arm',
                'L': motionplanning['L'],
                'states': states
            }
        }
    elif motionplanning['type'] == 'car':
        
        if path is None:
            states = [list(motionplanning['start'])]
            actions = [[]]
        
        else:
            final_path_node = next(node for node in nodes if np.array_equal(node.config, path[-1]))
            states, actions = extract_path_with_controls(final_path_node)
            states = [list(map(float, state)) for state in states]
            actions = [list(map(float, action)) for action in actions]

        output_data = {
            'plan': {
                'type': 'car',
                'dt': motionplanning['dt'],
                'L': motionplanning['L'],
                'W': motionplanning['W'],
                'H': 1,  # Adding height for the car
                'states': states,
                'actions': actions
            }
        }        

    
    
    output_data = convert_numpy_to_list(output_data)
    print('output_data: ', output_data)
    write_yaml(output_data, output_file)

    if path is not None:  
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
