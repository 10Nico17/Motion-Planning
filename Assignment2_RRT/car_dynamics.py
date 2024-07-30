import yaml
import numpy as np
import argparse
import subprocess
from pathlib import Path

def integrate_dynamics(actions, initial_state, L, dt):
    states = [initial_state]
    x, y, theta = initial_state
    #print(f"x: {x}, y: {y}, theta: {theta}")

    for action in actions:
        s, phi = action

        # Calculate derivatives
        dx = s * np.cos(theta)
        dy = s * np.sin(theta)
        dtheta = (s / L) * np.tan(phi)

        # Update states using Euler integration
        x += dx * dt
        y += dy * dt
        theta += dtheta * dt

        # Append the new state
        states.append([x, y, theta])

    return states

def main(input_file_path, output_file_path):
    # Load car actions from the YAML file
    with open(input_file_path, 'r') as file:
        car_actions_content = yaml.safe_load(file)

    # Extract parameters from the YAML content
    dt = car_actions_content['dt']
    L = car_actions_content['L']
    W = car_actions_content['W']
    H = car_actions_content['H']
    initial_state = car_actions_content['start']
    actions = car_actions_content['actions']

    # Integrate dynamics using the given actions
    car_plan = integrate_dynamics(actions, initial_state, L, dt)
    #print('states: ', car_plan)

    # Convert Numpy arrays to lists
    car_plan = [[float(x), float(y), float(theta)] for x, y, theta in car_plan]

    # Prepare the output data
    car_plan_yaml = {
        'plan': {
            'type': 'car',
            'dt': dt,
            'L': L,
            'W': W,
            'H': H,
            'states': car_plan,
            'actions': actions
        }
    }

    # Save the resulting plan to a new YAML file with the correct formatting
    with open(output_file_path, 'w') as file:
        yaml.dump(car_plan_yaml, file, default_flow_style=None, sort_keys=False)

    print(f"Generated car plan saved to: {output_file_path}")


    '''
    # Visualize the resulting plan using car_vis.py
    env_file_path = Path(input_file_path).parent / "car_env_0.yaml"
    
    subprocess.run(
        ["python3", "car_vis.py", str(env_file_path), str(output_file_path)],
        check=True
    )
    '''


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process car dynamics.')
    parser.add_argument('input_file', type=str, help='Path to the input YAML file with car actions')
    parser.add_argument('output_file', type=str, help='Path to the output YAML file for car plan')

    args = parser.parse_args()
    main(args.input_file, args.output_file)
