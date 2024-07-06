import yaml
import numpy as np
import fcl
import sys

def read_yaml(file_path):
    '''Reads a YAML file and returns the data as a dictionary.'''
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def create_box_obstacle(obstacle):
    '''Creates a box obstacle for collision checking.'''
    size = obstacle['size']
    translation = obstacle['pos']
    box = fcl.Box(size[0], size[1], size[2])
    tf = fcl.Transform(np.eye(3), translation)
    return fcl.CollisionObject(box, tf)

def create_cylinder_obstacle(obstacle):
    '''Creates a cylinder obstacle for collision checking.'''
    radius = obstacle['r']
    length = obstacle['lz']
    translation = obstacle['pos']
    q = obstacle['q']  # Quaternion (w, x, y, z)
    cylinder = fcl.Cylinder(radius, length)
    tf = fcl.Transform(q, translation)
    return fcl.CollisionObject(cylinder, tf)

def create_sphere(radius, position):
    '''Creates a sphere for collision checking.'''
    sphere = fcl.Sphere(radius)
    tf = fcl.Transform(np.eye(3), position)
    return fcl.CollisionObject(sphere, tf)

def forward_kinematics_arm(state, L):
    '''Computes the forward kinematics for the arm given the state and link lengths.'''
    print('state: ', state)
    print('state: ', type(state))
    theta1, theta2, theta3 = state
    p0 = np.array([0, 0, 0])
    p1 = p0 + L[0] * np.array([np.cos(theta1), np.sin(theta1), 0])
    p2 = p1 + L[1] * np.array([np.cos(theta1 + theta2), np.sin(theta1 + theta2), 0])
    p3 = p2 + L[2] * np.array([np.cos(theta1 + theta2 + theta3), np.sin(theta1 + theta2 + theta3), 0])
    joints = [p0, p1, p2, p3]
    print('joints: ', joints)
    return joints

def create_obstacle_objects(obstacles):
    '''Creates obstacle objects for collision checking.'''
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
    '''Checks if a given configuration is collision-free for the arm.'''
    joints = forward_kinematics_arm(config, L)
    collision = False
    request = fcl.CollisionRequest()
    result = fcl.CollisionResult()
    for joint in joints:
        joint_obj = create_sphere(0.05, joint)  # Create a sphere at each joint position
        for obstacle in obstacle_objects:
            ret = fcl.collide(joint_obj, obstacle, request, result)
            if ret:
                print('Collision detected at joint position:', joint, 'with obstacle:', obstacle)
                collision = True
                break
        if collision:
            break
    return not collision

def main(configs_file, obstacles_file):
    '''Main function to read configurations and obstacles, and check for collisions.'''
    #data = read_yaml(obstacles_file)
    data = read_yaml('cfg/arm_0.yaml')


    environment = data['environment']
    motionplanning = data['motionplanning']
    L = motionplanning['L']
    print('environment: ', environment)

    configs = read_yaml('configs_file.yaml')['configs']
    print('configs: ', configs)    
    print('configs: ', type(configs))    


    obstacle_objects = create_obstacle_objects(environment['obstacles'])

    for i, config in enumerate(configs):
        print(f'Checking configuration {i+1}: {config}')
        if is_collision_free_arm(config, obstacle_objects, L):
            print('No collision detected.')
        else:
            print('Collision detected.')

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 check_collisions.py <configs_file> <obstacles_file>")
        sys.exit(1)
    configs_file = sys.argv[1]
    obstacles_file = sys.argv[2]
    main(configs_file, obstacles_file)
