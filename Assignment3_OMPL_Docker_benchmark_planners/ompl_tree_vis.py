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

    # Extrahiere Knoten und Kanten aus den Baumdaten
    nodes = tree_data['nodes']
    edges = tree_data['edges']

    # Visualisiere Knoten und Kanten
    for i, node in enumerate(nodes):
        vis[f'node_{i}'].set_object(g.Sphere(0.002))
        vis[f'node_{i}'].set_transform(tf.translation_matrix(node))

        # Finde alle Kanten, die von diesem Knoten ausgehen
        for j, (start, end) in enumerate(edges):
            if start == i or end == i:
                start_pos = np.array(nodes[start])
                end_pos = np.array(nodes[end])
                midpoint = (start_pos + end_pos) / 2
                length = np.linalg.norm(end_pos - start_pos)
                direction = (end_pos - start_pos) / length

                # Berechne die Rotationsachse und den Winkel
                y_axis = np.array([0, 1, 0])
                axis = np.cross(y_axis, direction)
                if np.linalg.norm(axis) != 0:
                    axis = axis / np.linalg.norm(axis)
                angle = np.arccos(np.dot(y_axis, direction))

                # Erstelle die Rotationsmatrix
                if np.linalg.norm(axis) < 1e-6:
                    rotation = tf.identity_matrix()
                else:
                    rotation = tf.rotation_matrix(angle, axis)

                # Erstelle eine Transformationsmatrix, die zum Mittelpunkt übersetzt und dann rotiert
                translation = tf.translation_matrix(midpoint)
                transform = translation @ rotation

                # Skaliere den Zylinder auf die Länge der Kante
                vis[f'edge_{j}'].set_object(g.Cylinder(length, 0.001))
                vis[f'edge_{j}'].set_transform(transform)

    print("Druecken Eingabetaste, um zu beenden...")
    input()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Verwendung: python3 tree_vis.py <tree_file>")
        sys.exit(1)
    tree_file = sys.argv[1]
    visualize_tree(tree_file)
