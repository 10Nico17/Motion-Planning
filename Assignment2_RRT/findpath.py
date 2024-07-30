import yaml
import numpy as np

class Node:
    def __init__(self, config, parent=None):
        self.config = config
        self.parent = parent

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

def main(yaml_file):
    # Load the tree from the YAML file
    with open(yaml_file, 'r') as file:
        tree_data = yaml.safe_load(file)
    
    # Initialize nodes and edges
    nodes_list = [Node(config) for config in tree_data['nodes']]
    edges_list = tree_data['edges']
    
    # Define start and goal configurations
    start_config = [1.57, 1.57, 0]
    goal_config = [1.5, 1.3, 0.1]
    
    # Find the shortest path
    shortest_path = find_shortest_path(nodes_list, edges_list, start_config, goal_config)
    print("Shortest path:", shortest_path)

# Example usage
yaml_file = 'tree_rrt.yaml'
main(yaml_file)
