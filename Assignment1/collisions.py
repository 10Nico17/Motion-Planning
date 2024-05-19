import argparse
import yaml
import numpy as np
import fcl

def read_yaml(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def create_box_obstacle(obstacle):
    size = obstacle['size']
    translation = obstacle['pos']
    box = fcl.Box(size[0], size[1], size[2])
    tf = fcl.Transform(np.eye(3), translation)
    return fcl.CollisionObject(box, tf)

def create_cylinder_obstacle(obstacle):
    radius = obstacle['r']
    length = obstacle['lz']
    translation = obstacle['pos']
    q = obstacle['q']  # Quaternion (w, x, y, z)
    #print('q: ', q)
    cylinder = fcl.Cylinder(radius, length)
    tf = fcl.Transform(q, translation)
    return fcl.CollisionObject(cylinder, tf)

def create_sphere(radius, position):
    sphere = fcl.Sphere(radius)
    tf = fcl.Transform(np.eye(3), position)
    return fcl.CollisionObject(sphere, tf)

def forward_kinematics_arm(state, L):
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



def check_collision(env_file, plan_file, output_file):
    env = read_yaml(env_file)
    plan = read_yaml(plan_file)

    # Create environment obstacles
    obstacles = []
    for obstacle in env["environment"]["obstacles"]:
        if obstacle["type"] == "box":
            obstacles.append(create_box_obstacle(obstacle))
        elif obstacle["type"] == "cylinder":
            obstacles.append(create_cylinder_obstacle(obstacle))           
        else:
            raise ValueError(f"Unknown obstacle type: {obstacle['type']}")

    
    
    
    manager = fcl.DynamicAABBTreeCollisionManager()
    manager.registerObjects(obstacles)
    manager.setup()

    collision_results = []
    if plan["plan"]["type"] == "arm":
        L = plan["plan"]["L"]
        for state in plan["plan"]["states"]:
            joints = forward_kinematics_arm(state, L)
            #print('Joint positions:', joints)

            collision = False
            request = fcl.CollisionRequest()
            result = fcl.CollisionResult()
            for joint in joints:
                #print('joint xyz coordinates: ', joint)
                joint_obj = create_sphere(0.05, joint)
                for obstacle in obstacles:
                    ret = fcl.collide(joint_obj, obstacle, request, result)
                    if ret:
                        collision = True
                        break
            collision_results.append(collision)
            #print('collision_results: ', collision_results)
    
    elif plan["plan"]["type"] == "car":
        L = plan["plan"]["L"]
        W = plan["plan"].get("W", None) 
        dt = plan["plan"].get("dt", None)  
        initial_state = plan["plan"]["states"][0]
        actions = plan["plan"]["actions"]
        car_states = [initial_state]  # Initialize with the first state

        # Apply actions to get subsequent states
        for action in actions:
            next_state = forward_dynamics_car(car_states[-1], action, dt)
            car_states.append(next_state)

        #print('car_states: ', car_states)


        for state in car_states:
            car_box = create_car_box(L, W, state)
            collision = False
            request = fcl.CollisionRequest()
            result = fcl.CollisionResult()
            for obstacle in obstacles:
                ret = fcl.collide(car_box, obstacle, request, result)
                if ret:
                    collision = True
                    break
            collision_results.append(collision)     
            #print('collision_results: ', collision_results)



    print('Writing results to file...')
    try:
        with open(output_file, 'w') as file:
            yaml.dump({'collisions': collision_results}, file)
        print(f"Results written to {output_file}")
    except Exception as e:
        print(f"Error writing to file {output_file}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('env', help='input YAML file with environment')
    parser.add_argument('plan', help='input YAML file with plan')
    parser.add_argument('output', help='output file with collision results')
    args = parser.parse_args()

    check_collision(args.env, args.plan, args.output)
