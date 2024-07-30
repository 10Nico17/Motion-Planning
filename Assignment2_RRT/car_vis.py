"""
Example script to visualize a motion plan of a car using Meshcat

Usage: car_vis.py <env.yaml> <plan.yaml>
"""

import argparse
import yaml
import time
import numpy as np

# visualization related
import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf
from meshcat.animation import Animation


def calculate_car_corners(x, y, theta, L, W):
    # Compute car corners
    car_corners = [
        np.array([x - L/2 * np.cos(theta) - W/2 * np.sin(theta), y - L/2 * np.sin(theta) + W/2 * np.cos(theta), 0]),
        np.array([x + L/2 * np.cos(theta) - W/2 * np.sin(theta), y + L/2 * np.sin(theta) + W/2 * np.cos(theta), 0]),
        np.array([x + L/2 * np.cos(theta) + W/2 * np.sin(theta), y + L/2 * np.sin(theta) - W/2 * np.cos(theta), 0]),
        np.array([x - L/2 * np.cos(theta) + W/2 * np.sin(theta), y - L/2 * np.sin(theta) - W/2 * np.cos(theta), 0]),
    ]
    return car_corners


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
    W = plan["plan"]["W"]
    H = plan["plan"]["H"]

    # For the car, show both a box collision shape and a 3D model
    # The 3D model is from https://www.thingiverse.com/thing:5148987
    vis["car"]["collision"].set_object(g.Box([L, W, H]), g.MeshLambertMaterial(
                             color=0x0000ff,
                             opacity=0.4))
    vis["car"]["3d"].set_object(g.Mesh(g.StlMeshGeometry.from_file("car.stl")))
    vis["car"]["3d"].set_transform(
        tf.scale_matrix(0.7).dot(
            tf.translation_matrix([0,0,-0.7])).dot(tf.euler_matrix(0,0,np.pi)))


    anim = Animation()

    for k, (x, y, theta) in enumerate(plan["plan"]["states"]):
        car_corners = calculate_car_corners(x, y, theta, L, W)
        print(f"Frame {k}: Car corners: {car_corners}")

        with anim.at_frame(vis, k/args.dt) as frame:
            # NOTE: technically the center of mass for the car is between the back wheels;
            #       For simplicity, we use the center here
            frame["car"].set_transform(
                tf.translation_matrix([x, y, 0.5]).dot(
                    tf.euler_matrix(0, 0, theta)))
            
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
