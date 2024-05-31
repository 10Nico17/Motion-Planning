import yaml
import numpy as np
import sys
import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf

class Node:
    def __init__(self, config, parent=None):
        self.config = config
        self.parent = parent

def read_yaml(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def extract_path(node):
    """
    Extracts the path from the given node to the start node using the parent references.
    """
    path = []
    while node is not None:
        path.append(node.config)
        node = node.parent
    return path[::-1]

def find_shortest_path(nodes, edges, start_config, goal_config):
    """
    Finds the shortest path from any goal node to the start node in the given tree.
    """
    # Create a dictionary to map configurations to Node objects
    config_to_node = {tuple(node.config): node for node in nodes}

    # Set up the parent relationships based on the edges
    for edge in edges:
        parent_index, child_index = edge
        parent_node = nodes[parent_index]
        child_node = nodes[child_index]
        child_node.parent = parent_node

    # Find the start node
    start_node = config_to_node[tuple(start_config)]

    # Find all goal nodes
    goal_nodes = [node for node in nodes if np.array_equal(node.config, goal_config)]

    if not goal_nodes:
        print("Goal node not found in the tree.")
        return []

    # Extract paths for all goal nodes and find the shortest path
    shortest_path = None
    shortest_path_length = float('inf')

    for goal_node in goal_nodes:
        path = extract_path(goal_node)
        path_length = len(path)
        if path_length < shortest_path_length:
            shortest_path = path
            shortest_path_length = path_length

    return shortest_path

def visualize_tree(tree_file, shortest_path, start_config, goal_config):
    tree_data = read_yaml(tree_file)

    vis = meshcat.Visualizer().open()

    # Extract nodes and edges from the tree data
    nodes = tree_data['nodes']
    edges = tree_data['edges']

    # Visualize nodes
    for i, node in enumerate(nodes):
        if node in shortest_path:
            if node == start_config or node == goal_config:
                vis[f'node_{i}'].set_object(g.Sphere(0.025), g.MeshLambertMaterial(color=0x00ff00))
            else:
                vis[f'node_{i}'].set_object(g.Sphere(0.025), g.MeshLambertMaterial(color=0xff0000))
        else:
            vis[f'node_{i}'].set_object(g.Sphere(0.025))
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
        
        # Create a transform matrix that translates to the midpoint and then rotates
        translation = tf.translation_matrix([midpoint[0], midpoint[1], 0.0])
        rotation = tf.rotation_matrix(angle, [0, 0, 1])
        transform = translation @ rotation

        # Scale the cylinder to the length of the edge
        vis[f'edge_{i}'].set_object(g.Cylinder(length, 0.0001))
        vis[f'edge_{i}'].set_transform(transform)

    print("Press Enter to exit...")
    input()

def main(tree_file, config_file):
    # Load the tree from the YAML file
    with open(tree_file, 'r') as file:
        tree_data = yaml.safe_load(file)
    
    # Load the configuration from the config file
    config_data = read_yaml(config_file)
    start_config = config_data['motionplanning']['start']
    goal_config = config_data['motionplanning']['goal']
    
    # Initialize nodes and edges
    nodes_list = [Node(config) for config in tree_data['nodes']]
    edges_list = tree_data['edges']
    
    # Find the shortest path
    shortest_path = find_shortest_path(nodes_list, edges_list, start_config, goal_config)
    print("Shortest path:", shortest_path)

    # Visualize the tree and highlight the shortest path
    visualize_tree(tree_file, shortest_path, start_config, goal_config)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 tree_vis.py <tree_file> <config_file>")
        sys.exit(1)
    tree_file = sys.argv[1]
    config_file = sys.argv[2]
    main(tree_file, config_file)
