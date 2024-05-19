import argparse
import yaml
import time
import numpy as np
import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf
from meshcat.animation import Animation

def read_yaml(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def forward_kinematics(theta1, theta2, theta3, L):
    """Compute the (x, y) coordinates of each joint given the angles and segment lengths."""
    l1, l2, l3 = L
    p0 = np.array([0, 0])
    p1 = p0 + l1 * np.array([np.cos(theta1), np.sin(theta1)])
    p2 = p1 + l2 * np.array([np.cos(theta1 + theta2), np.sin(theta1 + theta2)])
    p3 = p2 + l3 * np.array([np.cos(theta1 + theta2 + theta3), np.sin(theta1 + theta2 + theta3)])
    return p0, p1, p2, p3

def add_cylinder_between_joints(vis, start, end, name):
    """Add a cylinder between two joints."""
    direction = end - start
    length = np.linalg.norm(direction)
    midpoint = (start + end) / 2
    angle = np.arctan2(direction[1], direction[0])
    angle += 1.57  # Adjust the angle for correct orientation

    # Create a transform matrix that translates to the midpoint and then rotates
    translation = tf.translation_matrix([midpoint[0], midpoint[1], 0.0])
    rotation = tf.rotation_matrix(angle, [0, 0, 1])
    transform = translation @ rotation

    # Set the object and transform
    vis[name].set_object(g.Cylinder(length, 0.02))
    return transform

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('env', help='input YAML file with environment')
    parser.add_argument('plan', help='input YAML file with plan')
    parser.add_argument('--dt', type=float, default=0.1, help='sleeping time between frames')
    parser.add_argument('--output', default=None, help='output file with animation')
    args = parser.parse_args()

    with open(args.env, "r") as stream:
        env = yaml.safe_load(stream)

    with open(args.plan, "r") as stream:
        plan = yaml.safe_load(stream)

    vis = meshcat.Visualizer()

    for k, o in enumerate(env["environment"]["obstacles"]):
        if o["type"] == "box":
            p = o["pos"]
            s = o["size"]
            vis["obstacles"][str(k)].set_object(g.Box(s))
            vis["obstacles"][str(k)].set_transform(tf.translation_matrix(p))
        elif o["type"] == "cylinder":
            p = o["pos"]
            q = o["q"]
            r = o["r"]
            lz = o["lz"]
            vis["obstacles"][str(k)].set_object(g.Cylinder(lz, r))
            # NOTE: Additional transformation to match fcl
            vis["obstacles"][str(k)].set_transform(
                tf.translation_matrix(p).dot(
                    tf.quaternion_matrix(q)).dot(
                        tf.euler_matrix(np.pi/2,0,0)))
        else:
            raise RuntimeError("Unknown obstacle type " + o["type"])

    L = plan["plan"]["L"]
    
    # Initialize the joints and segments
    vis['joint0'].set_object(g.Sphere(0.05))
    vis['joint1'].set_object(g.Sphere(0.05))
    vis['joint2'].set_object(g.Sphere(0.05))
    vis['joint3'].set_object(g.Sphere(0.05))

    seg1_transform = add_cylinder_between_joints(vis, np.array([0, 0]), np.array([1, 0]), 'segment1')
    seg2_transform = add_cylinder_between_joints(vis, np.array([1, 0]), np.array([2, 0]), 'segment2')
    seg3_transform = add_cylinder_between_joints(vis, np.array([2, 0]), np.array([3, 0]), 'segment3')

    anim = Animation()

    for frame, angles in enumerate(plan["plan"]["states"]):
        theta1, theta2, theta3 = angles
        p0, p1, p2, p3 = forward_kinematics(theta1, theta2, theta3, L)

        #print(f"Frame {frame}:")
        #print(f"  Joint 0: {p0}")
        #print(f"  Joint 1: {p1}")
        #print(f"  Joint 2: {p2}")
        #print(f"  Joint 3: {p3}")

        with anim.at_frame(vis, frame / args.dt) as frame_vis:
            # Set the positions for each joint
            frame_vis['joint0'].set_transform(tf.translation_matrix([p0[0], p0[1], 0.0]))
            frame_vis['joint1'].set_transform(tf.translation_matrix([p1[0], p1[1], 0.0]))
            frame_vis['joint2'].set_transform(tf.translation_matrix([p2[0], p2[1], 0.0]))
            frame_vis['joint3'].set_transform(tf.translation_matrix([p3[0], p3[1], 0.0]))

            # Add transformations for cylinders between joints
            frame_vis['segment1'].set_transform(add_cylinder_between_joints(vis, p0, p1, 'segment1'))
            frame_vis['segment2'].set_transform(add_cylinder_between_joints(vis, p1, p2, 'segment2'))
            frame_vis['segment3'].set_transform(add_cylinder_between_joints(vis, p2, p3, 'segment3'))

    vis.set_animation(anim)

    if args.output is None:
        vis.open()
        time.sleep(1e9)
    else:
        res = vis.static_html()
        with open(args.output, "w") as f:
            f.write(res)

if __name__ == "__main__":
    main()
