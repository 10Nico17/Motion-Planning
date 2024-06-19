import yaml
import numpy as np
import sys

class NearestNeighbor:
    def __init__(self, distance_metric):
        '''
        Initialize with an empty list of configurations and a specified distance metric.
        '''
        self.configurations = []
        self.distance_metric = distance_metric

    def addConfiguration(self, q):
        '''
        Add a new configuration to the list.
        '''
        self.configurations.append(np.array(q))

    def nearestK(self, q, k):
        '''
        Find the k nearest configurations to q.
        '''
        q = np.array(q)
        dists = [self.compute_distance(q, conf) for conf in self.configurations]
        nearest_indices = np.argsort(dists)[:k]
        nearest_configs = [self.configurations[i].tolist() for i in nearest_indices]
        return nearest_configs

    def nearestR(self, q, r):
        '''
        Find all configurations within radius r of q.
        '''
        q = np.array(q)
        dists = [self.compute_distance(q, conf) for conf in self.configurations]
        within_r_indices = [i for i, dist in enumerate(dists) if dist <= r]
        nearest_configs = [self.configurations[i].tolist() for i in within_r_indices]
        return nearest_configs

    def compute_distance(self, q1, q2):
        '''
        Compute the distance between two configurations based on the specified metric.
        '''
        if self.distance_metric == 'l2':
            return np.linalg.norm(q1 - q2)  
        elif self.distance_metric == 'se2':
            pos_dist = np.linalg.norm(q1[:2] - q2[:2])
            ang_dist = min(abs(q1[2] - q2[2]), 2 * np.pi - abs(q1[2] - q2[2]))
            return pos_dist + ang_dist
        elif self.distance_metric == 'angles':
            return np.sum([min(abs(a - b), 2 * np.pi - abs(a - b)) for a, b in zip(q1, q2)])
        else:
            raise ValueError(f"Unsupported distance metric: {self.distance_metric}")

def main(config_file, output_file):
    '''
    Read configurations and queries from the input YAML file, process the queries,
    and write the results to the output YAML file.
    '''
    with open(config_file, 'r') as file:
        config_data = yaml.safe_load(file)

    nn = NearestNeighbor(config_data['distance'])

    for conf in config_data['configurations']:
        nn.addConfiguration(conf)

    results = []
    for query in config_data['queries']:
        if query['type'] == 'nearestK':
            result = nn.nearestK(query['q'], query['k'])
        elif query['type'] == 'nearestR':
            result = nn.nearestR(query['q'], query['r'])
        else:
            raise ValueError(f"Unsupported query type: {query['type']}")
        results.append(result)

    output_data = {'results': results}
    with open(output_file, 'w') as file:
        yaml.dump(output_data, file, default_flow_style=None)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 nearest_neighbor.py <config_file> <output_file>")
        sys.exit(1)
    config_file = sys.argv[1]
    output_file = sys.argv[2]
    main(config_file, output_file)
