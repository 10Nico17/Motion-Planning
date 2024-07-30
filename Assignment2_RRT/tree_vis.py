import yaml
import numpy as np
import sys
import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf

def read_yaml(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)
    
def visualize_tree(tree_file):
    tree_data = read_yaml(tree_file)

    vis = meshcat.Visualizer().open()

    # Extract nodes and edges from the tree data
    nodes = tree_data['nodes']
    edges = tree_data['edges']

    # Visualize nodes
    for i, node in enumerate(nodes):
        vis[f'node_{i}'].set_object(g.Sphere(0.05))
        vis[f'node_{i}'].set_transform(tf.translation_matrix(node))

    # Visualize edges
    for i, (start, end) in enumerate(edges):
        start_pos = np.array(nodes[start])
        end_pos = np.array(nodes[end])
        midpoint = (start_pos + end_pos) / 2
        length = np.linalg.norm(end_pos - start_pos)

        direction = (end_pos - start_pos) / length
        angle = np.arctan2(direction[1], direction[0])
        angle += 1.57  # Adjust the angle for correct orientation
        
        '''
        print(f"Edge {i}:")
        print(f"  Start position: {start_pos}")
        print(f"  End position: {end_pos}")
        print(f"  Midpoint: {midpoint}")
        print(f"  Length: {length}")
        '''
        
        # Create a transform matrix that translates to the midpoint and then rotates
        translation = tf.translation_matrix([midpoint[0], midpoint[1], 0.0])
        rotation = tf.rotation_matrix(angle, [0, 0, 1])
        transform = translation @ rotation

        
        # Scale the cylinder to the length of the edge
        vis[f'edge_{i}'].set_object(g.Cylinder(length, 0.0001))
        vis[f'edge_{i}'].set_transform(transform)

    print("Press Enter to exit...")
    input()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 tree_vis.py <tree_file>")
        sys.exit(1)
    tree_file = sys.argv[1]
    visualize_tree(tree_file)
