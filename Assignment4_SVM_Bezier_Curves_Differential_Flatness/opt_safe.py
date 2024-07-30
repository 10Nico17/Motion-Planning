import yaml
import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
import sys
import argparse

def load_config(filename):
    with open(filename, 'r') as file:
        config = yaml.safe_load(file)
    return config

def get_corners(pos, size):
    half_size = size / 2.0
    bottom_left = pos - half_size
    bottom_right = pos + [half_size[0], -half_size[1]]
    top_right = pos + half_size
    top_left = pos + [-half_size[0], half_size[1]]
    return [bottom_left, bottom_right, top_right, top_left]

def generate_hyperplanes(solution_path, obstacles):
    hyperplanes = []

    for i in range(len(solution_path) - 1):
        segment_hyperplanes = []
        segment_start = np.array(solution_path[i][:2])
        segment_end = np.array(solution_path[i + 1][:2])        
        segment_dir = segment_end - segment_start        
        segment_dir_perp = np.array([-segment_dir[1], segment_dir[0]])

        for obstacle in obstacles:
            pos = np.array(obstacle['pos'][:2])
            size = np.array(obstacle['size'][:2])
            corners = get_corners(pos, size)
            

            w = cp.Variable(2)
            b = cp.Variable()
            
            constraints = []

            # all points that should be on one side (i.e., corners of an obstacle)
            for corner in corners:
                constraints.append(w.T @ corner - b >= 1)
            
            # all points that should be on the other side (i.e., endpoints of a line segment)
            constraints.append(w.T @ segment_start - b <= -1)
            constraints.append(w.T @ segment_end - b <= -1)
            
            # Define the objective
            objective = cp.Minimize(cp.norm(w, 2)**2)
            
            # Form and solve the problem
            prob = cp.Problem(objective, constraints)
            prob.solve()

            if prob.status == cp.OPTIMAL:
                segment_hyperplanes.append((w.value, b.value))
            else:
                print(f"Warning: Problem status {prob.status} for segment {i} and obstacle {obstacle}")

        hyperplanes.append(segment_hyperplanes)

    return hyperplanes


def compute_intersection_points(hyperplanes, min_env, max_env, obstacles):
    points = []
    for i, (w1, b1) in enumerate(hyperplanes):
        for j, (w2, b2) in enumerate(hyperplanes):
            if i >= j:
                continue
            A = np.array([w1, w2])
            if np.linalg.matrix_rank(A) < 2:
                continue
            b = np.array([b1, b2])
            intersection = np.linalg.solve(A, b)
            if (min_env[0] <= intersection[0] <= max_env[0]) and (min_env[1] <= intersection[1] <= max_env[1]):
                # Prüfen, ob der Schnittpunkt nicht auf einem Hindernis liegt
                on_obstacle = False
                for obstacle in obstacles:
                    pos = np.array(obstacle['pos'][:2])
                    size = np.array(obstacle['size'][:2])
                    half_size = size / 2.0
                    if all(pos - half_size <= intersection) and all(intersection <= pos + half_size):
                        on_obstacle = True
                        break
                if not on_obstacle:
                    points.append(intersection)

    return np.array(points)



def convex_hull(points):
    points = sorted(points, key=lambda x: (x[0], x[1]))
    lower = []
    for p in points:
        while len(lower) >= 2 and np.cross(lower[-1] - lower[-2], p - lower[-1]) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and np.cross(upper[-1] - upper[-2], p - upper[-1]) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])

def distance_to_hyperplane(point, w, b):
    return abs(w @ point - b) / np.linalg.norm(w)


# Funktion, um den linken und rechten Nachbarn zu finden
def find_closest_neighbors(segment_point, point_list, coord_index):
    fixed_coord = segment_point[coord_index]
    variable_coord = segment_point[1 - coord_index]
    
    left_neighbor = None
    right_neighbor = None
    min_left_diff = float('inf')
    min_right_diff = float('inf')
    
    for point in point_list:
        fixed, variable = point[coord_index], point[1 - coord_index]
        if variable == variable_coord:
            diff = fixed_coord - fixed
            if diff > 0 and diff < min_left_diff:
                min_left_diff = diff
                left_neighbor = point
            elif diff < 0 and -diff < min_right_diff:
                min_right_diff = -diff
                right_neighbor = point
    
    return left_neighbor, right_neighbor

def is_line_segment_collision_free(p1, p2, obstacles):
    for obstacle in obstacles:
        pos = np.array(obstacle['pos'][:2])
        size = np.array(obstacle['size'][:2])
        half_size = size / 2.0
        corners = get_corners(pos, size)

        for i in range(len(corners)):
            corner1 = corners[i]
            corner2 = corners[(i + 1) % len(corners)]
            if do_lines_intersect(p1, p2, corner1, corner2):
                return False
    return True

def do_lines_intersect(p1, p2, q1, q2):
    def ccw(A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
    return ccw(p1, q1, q2) != ccw(p2, q1, q2) and ccw(p1, p2, q1) != ccw(p1, p2, q2)

def compute_boundary_intersections(hyperplane, min_env, max_env):
    w, b = hyperplane
    intersections = []

    # Intersection with the vertical boundaries (min_env[0] and max_env[0])
    for x in [min_env[0], max_env[0]]:
        if abs(w[1]) > 1e-12:  # To avoid division by zero or near-zero
            y = (b - w[0] * x) / w[1]
            if min_env[1] <= y <= max_env[1]:
                intersections.append([x, y])
    
    # Intersection with the horizontal boundaries (min_env[1] and max_env[1])
    for y in [min_env[1], max_env[1]]:
        if abs(w[0]) > 1e-12:  # To avoid division by zero or near-zero
            x = (b - w[1] * y) / w[0]
            if min_env[0] <= x <= max_env[0]:
                intersections.append([x, y])
    
    return intersections

def compute_safe_regions(solution_path, hyperplanes, min_env, max_env, obstacles):
    regions = []

    for i in range(len(solution_path) - 1):
        segment_start = np.array(solution_path[i][:2])
        segment_end = np.array(solution_path[i + 1][:2])

        segment_hyperplanes = hyperplanes[i]
        if len(segment_hyperplanes) >= 2:
            points = compute_intersection_points(segment_hyperplanes, min_env, max_env, obstacles)           
            for hyperplane in segment_hyperplanes:
                boundary_points = compute_boundary_intersections(hyperplane, min_env, max_env)
            
        point_y_start_list = []
        point_x_start_list = []
        point_y_end_list = []
        point_x_end_list = []

        # Add additional points along the height of start and end points
        for w, b in segment_hyperplanes:
            print('w, b: ', w, b)
            point_y_start = [segment_start[0], (-w[0] * segment_start[0] + b) / w[1]]    
            if min_env[1] <= point_y_start[1] <= max_env[1]:
                point_y_start_list.append(point_y_start)
                #points = np.vstack([points, point_y_start])

            point_x_start = [(b - w[1] * segment_start[1]) / w[0], segment_start[1]]
            if min_env[0] <= point_x_start[0] <= max_env[0]:
                point_x_start_list.append(point_x_start)
                #points = np.vstack([points, point_x_start])
            
            point_y_end = [segment_end[0], (-w[0] * segment_end[0] + b) / w[1]]
            if min_env[1] <= point_y_end[1] <= max_env[1]:
                point_y_end_list.append(point_y_end)
                #points = np.vstack([points, point_y_end])
 
            point_x_end = [(b - w[1] * segment_end[1]) / w[0], segment_end[1]]
            if min_env[0] <= point_x_end[0] <= max_env[0]:
                point_x_end_list.append(point_x_end)
                #points = np.vstack([points, point_x_end])

        # Finde Nachbarn für Startpunkt in X-Richtung
        left_neighbor_x, right_neighbor_x = find_closest_neighbors(segment_start, point_x_start_list, coord_index=0)
        if left_neighbor_x is not None:
            points = np.vstack([points, left_neighbor_x])
        else:
            print('Noneeeeeeeee segment_start left_neighbor_x: ', left_neighbor_x)

        if right_neighbor_x is not None:
            points = np.vstack([points, right_neighbor_x])
        else:
            manual_point = np.array([max_env[0], segment_start[1]])
            points = np.vstack([points, manual_point])
      
        
        # Finde Nachbarn für Endpunkt in X-Richtung
        left_neighbor_x, right_neighbor_x = find_closest_neighbors(segment_end, point_x_end_list, coord_index=0)
        if left_neighbor_x is not None:
            points = np.vstack([points, left_neighbor_x])
        else:
            print('Noneeeeeeeee segment_end left_neighbor_x: ', left_neighbor_x)

        if right_neighbor_x is not None:
            points = np.vstack([points, right_neighbor_x])
        else: 
            print('Noneeeeeeeee  segment_end right_neighbor_x: ', right_neighbor_x)


        # Finde Nachbarn für Startpunkt in Y-Richtung
        left_neighbor_y, right_neighbor_y = find_closest_neighbors(segment_start, point_y_start_list, coord_index=1)
        if left_neighbor_y is not None:
            points = np.vstack([points, left_neighbor_y])
        else:
            print('Noneeeeeeeee: ', left_neighbor_y)
        if right_neighbor_y is not None:
            points = np.vstack([points, right_neighbor_y])
        else:
            print('Noneeeeeeeee: ', left_neighbor_x)

        # Finde Nachbarn für Endpunkt in Y-Richtung
        left_neighbor_y, right_neighbor_y = find_closest_neighbors(segment_end, point_y_end_list, coord_index=1)
        if left_neighbor_y is not None:
            points = np.vstack([points, left_neighbor_y])
        if right_neighbor_y is not None:
            points = np.vstack([points, right_neighbor_y])   

        # Filter points to keep only those that can connect without collision
        collision_free_points = []
        for point in points:
            collision_free = True
            for other_point in points:
                if not np.array_equal(point, other_point) and not is_line_segment_collision_free(point, other_point, obstacles):
                    collision_free = False
                    break
            if collision_free:
                collision_free_points.append(point)

        collision_free_points = np.array(collision_free_points)
        # Finde Nachbarn für Startpunkt in X-Richtung
        left_neighbor_x, right_neighbor_x = find_closest_neighbors(segment_start, point_x_start_list, coord_index=0)
        if left_neighbor_x is not None:
            collision_free_points = np.vstack([collision_free_points, left_neighbor_x])
        else:
            manual_point = np.array([min_env[0], segment_start[1]])
            collision_free_points = np.vstack([collision_free_points, manual_point])

        if right_neighbor_x is not None:
            collision_free_points = np.vstack([collision_free_points, right_neighbor_x])
        else:
            manual_point = np.array([max_env[0], segment_start[1]])
            collision_free_points = np.vstack([collision_free_points, manual_point])


        
        # Finde Nachbarn für Endpunkt in X-Richtung
        left_neighbor_x, right_neighbor_x = find_closest_neighbors(segment_end, point_x_end_list, coord_index=0)
        if left_neighbor_x is not None:
            collision_free_points = np.vstack([collision_free_points, left_neighbor_x])
        else:
            manual_point = np.array([min_env[0], segment_end[1]])
            collision_free_points = np.vstack([collision_free_points, manual_point])
        if right_neighbor_x is not None:
            collision_free_points = np.vstack([collision_free_points, right_neighbor_x])
        else:
            manual_point = np.array([max_env[0], segment_end[1]])
            collision_free_points = np.vstack([collision_free_points, manual_point])




        # Finde Nachbarn für Startpunkt in Y-Richtung
        left_neighbor_y, right_neighbor_y = find_closest_neighbors(segment_start, point_y_start_list, coord_index=1)
        if left_neighbor_y is not None:
            collision_free_points = np.vstack([collision_free_points, left_neighbor_y])
        else:
            manual_point = np.array([segment_start[0], min_env[1]])
            collision_free_points = np.vstack([collision_free_points, manual_point])

        if right_neighbor_y is not None:
            collision_free_points = np.vstack([collision_free_points, right_neighbor_y])
        else:
            manual_point = np.array([segment_start[0], max_env[1]])
            collision_free_points = np.vstack([collision_free_points, manual_point])


        # Finde Nachbarn für Endpunkt in Y-Richtung
        left_neighbor_y, right_neighbor_y = find_closest_neighbors(segment_end, point_y_end_list, coord_index=1)
        if left_neighbor_y is not None:
            collision_free_points = np.vstack([collision_free_points, left_neighbor_y])
        else:
            manual_point = np.array([segment_end[0], min_env[1]])
            collision_free_points = np.vstack([collision_free_points, manual_point])
        if right_neighbor_y is not None:
            collision_free_points = np.vstack([collision_free_points, right_neighbor_y])
        else:
            manual_point = np.array([segment_end[0], max_env[1]])
            collision_free_points = np.vstack([collision_free_points, manual_point])  



        if len(collision_free_points) > 0:
            hull_points = convex_hull(collision_free_points)
            regions.append(hull_points)       

    return regions



def compute_bezier_curves(solution_path, regions):
    bezier_curves = []

    q_start = np.array(solution_path[0][:2])
    q_goal = np.array(solution_path[-1][:2])

    region1_x = [np.min(regions[0][:, 0]), np.max(regions[0][:, 0])]
    region1_y = [np.min(regions[0][:, 1]), np.max(regions[0][:, 1])]    
    region2_x = [np.min(regions[1][:, 0]), np.max(regions[1][:, 0])]
    region2_y = [np.min(regions[1][:, 1]), np.max(regions[1][:, 1])]    
    region3_x = [np.min(regions[2][:, 0]), np.max(regions[2][:, 0])]
    region3_y = [np.min(regions[2][:, 1]), np.max(regions[2][:, 1])]

    P0 = cp.Variable(2)
    P1 = cp.Variable(2)
    P2 = cp.Variable(2)
    P3 = cp.Variable(2)
    P4 = cp.Variable(2)
    P5 = cp.Variable(2)
    P6 = cp.Variable(2)
    P7 = cp.Variable(2)
    P8 = cp.Variable(2)
    P9 = cp.Variable(2)
    P10 = cp.Variable(2)
    P11 = cp.Variable(2)

    def obstacle_avoidance_cost(P, bottom_left, top_right):
        return cp.norm(cp.maximum(0, bottom_left - P), 2) + cp.norm(cp.maximum(0, P - top_right), 2)

    # Objective: Minimize the squared distances between consecutive control points + obstacle avoidance
    objective = cp.Minimize(
        cp.sum_squares(P1 - P0) + cp.sum_squares(P2 - P1) + cp.sum_squares(P3 - P2) +
        cp.sum_squares(P4 - P3) + cp.sum_squares(P5 - P4) + cp.sum_squares(P6 - P5) +
        cp.sum_squares(P7 - P6) + cp.sum_squares(P8 - P7) + cp.sum_squares(P9 - P8) +
        obstacle_avoidance_cost(P1, np.array([0, 0]), np.array([4, 2])) +
        obstacle_avoidance_cost(P2, np.array([1, 3]), np.array([4, 5])) +
        obstacle_avoidance_cost(P4, np.array([6, 1]), np.array([7, 5])) +
        obstacle_avoidance_cost(P5, np.array([6, 1]), np.array([7, 5]))
    )

    constraints = [
        P0 == q_start,  
        P1[0] >= region1_x[0], P1[0] <= region1_x[1], P1[1] >= region1_y[0], P1[1] <= region1_y[1],
        P2[0] >= region1_x[0], P2[0] <= region1_x[1], P2[1] >= region1_y[0], P2[1] <= region1_y[1],
        P3[0] >= max(region1_x[0], region2_x[0]), P3[0] <= min(region1_x[1], region2_x[1]), P3[1] >= max(region1_y[0], 
                                                                region2_y[0]), P3[1] <= min(region1_y[1], region2_y[1]),
        P4 == P3,
        P5[0] >= region2_x[0], P5[0] <= region2_x[1], P5[1] >= region2_y[0], P5[1] <= region2_y[1],
        P6[0] >= region2_x[0], P6[0] <= region2_x[1], P6[1] >= region2_y[0], P6[1] <= region2_y[1],
        P7[0] >= max(region2_x[0], region3_x[0]), P7[0] <= min(region2_x[1], region3_x[1]), P7[1] >= max(region2_y[0], 
                                                                region3_y[0]), P7[1] <= min(region2_y[1], region3_y[1]),
        P8 == P7,
        P9[0] >= region3_x[0], P9[0] <= region3_x[1], P9[1] >= region3_y[0], P9[1] <= region3_y[1],
        P10[0] >= region3_x[0], P10[0] <= region3_x[1], P10[1] >= region3_y[0], P10[1] <= region3_y[1],
        P11 == q_goal
    ]


    def first_derivative(P0, P1, P2, P3, t):
        return 3 * (1 - t)**2 * (P1 - P0) + 6 * (1 - t) * t * (P2 - P1) + 3 * t**2 * (P3 - P2)

    def second_derivative(P0, P1, P2, P3, t):
        return 6 * (1 - t) * (P2 - 2 * P1 + P0) + 6 * t * (P3 - 2 * P2 + P1)

    constraints += [
        first_derivative(P0, P1, P2, P3, 1) == first_derivative(P4, P5, P6, P7, 0),
        first_derivative(P4, P5, P6, P7, 1) == first_derivative(P8, P9, P10, P11, 0)
        ]

    constraints += [
        second_derivative(P0, P1, P2, P3, 1) == second_derivative(P4, P5, P6, P7, 0),
        second_derivative(P4, P5, P6, P7, 1) == second_derivative(P8, P9, P10, P11, 0)
    ]


    problem = cp.Problem(objective, constraints)
    problem.solve()
    P0_opt = P0.value
    P1_opt = P1.value
    P2_opt = P2.value
    P3_opt = P3.value
    P4_opt = P4.value
    P5_opt = P5.value
    P6_opt = P6.value
    P7_opt = P7.value
    P8_opt = P8.value
    P9_opt = P9.value
    P10_opt = P10.value
    P11_opt = P11.value

    bezier_curves.append((P0_opt, P1.value, P2.value, P3.value))
    bezier_curves.append((P4.value, P5.value, P6.value, P7.value))
    bezier_curves.append((P8.value, P9.value, P10.value, P11_opt))  

    return bezier_curves


def bezier_curve(P0, P1, P2, P3):
    t = np.linspace(0, 1, 100)
    bezier_curve_x = (1-t)**3 * P0[0] + 3*(1-t)**2 * t * P1[0] + 3*(1-t)*t**2 * P2[0] + t**3 * P3[0]
    bezier_curve_y = (1-t)**3 * P0[1] + 3*(1-t)**2 * t * P1[1] + 3*(1-t)*t**2 * P2[1] + t**3 * P3[1]
    return bezier_curve_x, bezier_curve_y


def plot_results(config, regions, hyperplanes, output_file):
    print('############# PLOT ###########################')
    fig, ax = plt.subplots(figsize=(16, 11))
    env = config['environment']
    min_env = env['min'][:2]
    max_env = env['max'][:2]
    
    plt.xlim(min_env[0], max_env[0])
    plt.ylim(min_env[1], max_env[1])

    for obstacle in env['obstacles']:
        pos = np.array(obstacle['pos'][:2]) - np.array(obstacle['size'][:2]) / 2.0
        size = obstacle['size'][:2]
        rect = Rectangle(pos, size[0], size[1], linewidth=1, edgecolor='r', facecolor='grey', alpha=0.5)
        ax.add_patch(rect)

    solution_path = np.array(config['motionplanning']['solutionpath'])
    plt.plot(solution_path[:, 0], solution_path[:, 1], 'b-o', label='Solution Path')

    colors = ['g', 'r', 'b', 'm', 'c', 'y', 'k']  # Liste der Farben für die Hyperplanes
    x = np.linspace(min_env[0], max_env[0], 200)
    for segment_index, segment_hyperplanes in enumerate(hyperplanes):
        for index, (w, b) in enumerate(segment_hyperplanes):
            if abs(w[1]) > 1e-12:  # Avoid division by near zero
                y = (-w[0] * x + b) / w[1]
                color = colors[segment_index % len(colors)]  # Wählen Sie die Farbe zyklisch aus
                plt.plot(x, y, color=color, linestyle='--', label=f'Segment {segment_index + 1} Hyperplane {index + 1}')
            else:
                print("Warning: Hyperplane is vertical, w[1] is very close to zero")

    

    '''
    for region in regions:
        polygon = Polygon(region, closed=True, edgecolor='orange', facecolor='lightblue', alpha=0.5, linestyle='--', linewidth=1.5)        
        ax.add_patch(polygon)
    ''' 

    legend_polygon = Polygon([[0, 0], [1, 1], [1, 0]], closed=True, edgecolor='orange', facecolor='lightblue', alpha=0.5, linestyle='--', linewidth=1.5)
    plt.plot([], [], color='lightblue', alpha=0.5, label='Safe Region', linestyle='--', linewidth=1.5)

    bezier_curves = compute_bezier_curves(solution_path, regions)

    for i, (P0, P1, P2, P3) in enumerate(bezier_curves):
        curve_x, curve_y = bezier_curve(P0, P1, P2, P3)
        plt.plot(curve_x, curve_y, label=f'Bezier Segment {i + 1}')
        
        control_points = [P0, P1, P2, P3]
        control_points_labels = ['P0', 'P1', 'P2', 'P3']
        for point, label in zip(control_points, control_points_labels):
            plt.scatter(point[0], point[1], color='black')
            plt.text(point[0], point[1], f'{label}_{i + 1}', fontsize=12, ha='right')


    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Hyperplanes and Safe Regions')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout(rect=[0, 0, 0.75, 1])  # Adjust the right side of the plot to make room for the legend

    plt.savefig(output_file)
    plt.close()

def convert_numpy(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.generic):
        return float(obj)
    return obj

def bezier_first_derivative(t, P0, P1, P2, P3):
    return (3*(1-t)**2*(P1-P0)+6*(1-t)*t*(P2-P1)+3*t**2*(P3-P2))

def bezier_second_derivative(t, P0, P1, P2, P3):
    return (6*(1-t)*(P2-2*P1+P0)+6*t*(P3-2*P2+P1))


def compute_car_plan(bezier_curves, L, start_state):
    print('############################################ car ######################################################')

    lower_bound = 0.5  
    upper_bound = 2.0  

    states = []
    actions = []
    #states.append(start_state)   
          
    for P0, P1, P2, P3 in bezier_curves:

        print('# points bezier curve: ', P0, P1, P2, P3)        
        steps = 10
        duration = 1/steps
        T=10
        step = T/duration
        t = np.linspace(0, T, int(step)+1)
        bezier_curve_x = (1-t/T)**3 * P0[0] + 3*(1-t/T)**2 * t/T * P1[0] + 3*(1-t/T)*(t/T)**2 * P2[0] + (t/T)**3 * P3[0]
        bezier_curve_y = (1-t/T)**3 * P0[1] + 3*(1-t/T)**2 * t/T * P1[1] + 3*(1-t/T)*(t/T)**2 * P2[1] + (t/T)**3 * P3[1]
        dx_dt = 1/T*(3*(1-t/T)**2*(P1[0]-P0[0])+6*(1-t/T)*t/T*(P2[0]-P1[0])+3*(t/T)**2*(P3[0]-P2[0]))
        dy_dt = 1/T*(3*(1-t/T)**2*(P1[1]-P0[1])+6*(1-t/T)*t/T*(P2[1]-P1[1])+3*(t/T)**2*(P3[1]-P2[1]))       
        
        ddx_dt = np.gradient(dx_dt, t)
        ddy_dt = np.gradient(dy_dt, t)     

   
        for x, y, dx, dy, ddx, ddy in zip(bezier_curve_x, bezier_curve_y, dx_dt, dy_dt, ddx_dt, ddy_dt):
            print('x and y: ', x, y)
            s = np.sqrt(dx**2 + dy**2)  
            phi = np.arctan(L * (dx * ddy - dy * ddx) / (dx**2 + dy**2)**1.5)     
            theta = np.arctan(dy/dx)
            states.append([float(x), float(y), float(theta)])
            actions.append([float(s), float(phi)])

           

    return {'states': states, 'actions': actions}, duration



def save_plan(plan, filename):
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.generic):
            return obj.item()
        return obj

    plan_converted = {k: [convert_numpy(v) for v in values] for k, values in plan.items()}
    with open(filename, 'w') as file:
        yaml.dump({'plan': plan_converted}, file)




def inverse_kinematics_arm(end_effector_pos, L):
    phi = 1.57   
    xe, ye = end_effector_pos
    l1,l2, l3 = L

    # P2 
    x2 = xe-l3*np.cos(phi)
    y2 = ye-l3*np.sin(phi)

    d = np.sqrt(x2**2 + y2**2)

    #if d > l1 + l2:
    #    raise ValueError("########################## Position außerhalb des erreichbaren Bereichs")

    cos_theta2 = (d**2 - l1**2 - l2**2) / (2 * l1 * l2)
    theta2 = np.arccos(cos_theta2)

    alpha = np.arctan2(y2, x2)
    beta = np.arccos((l1**2 + d**2 - l2**2) / (2 * l1 * d))
    theta1 = alpha - beta

    theta3 = phi - theta1 - theta2

    return np.array([theta1, theta2, theta3])


def forward_kinematics_arm(state, L):
    '''
    Computes the forward kinematics for the arm given the state and link lengths.
    '''
    theta1 = state[0]
    theta2 = state[1]
    theta3 = state[2]
    p0 = np.array([0, 0, 0])
    p1 = p0 + L[0] * np.array([np.cos(theta1), np.sin(theta1), 0])
    p2 = p1 + L[1] * np.array([np.cos(theta1 + theta2), np.sin(theta1 + theta2), 0])
    p3 = p2 + L[2] * np.array([np.cos(theta1 + theta2 + theta3), np.sin(theta1 + theta2 + theta3), 0])
    joints = [p0, p1, p2, p3]
    return joints


def compute_arm_plan(bezier_curves, L, goal):
    arm_plan = []
    forward_positions = []

    for P0, P1, P2, P3 in bezier_curves:    
        t = np.linspace(0, 1, 100)
        bezier_curve_x = (1-t)**3 * P0[0] + 3*(1-t)**2 * t * P1[0] + 3*(1-t)*t**2 * P2[0] + t**3 * P3[0]
        bezier_curve_y = (1-t)**3 * P0[1] + 3*(1-t)**2 * t * P1[1] + 3*(1-t)*t**2 * P2[1] + t**3 * P3[1]  
        for x, y in zip(bezier_curve_x, bezier_curve_y):
            joint_angles = inverse_kinematics_arm([x, y], L)
            arm_plan.append(joint_angles)
            forward_position = forward_kinematics_arm(joint_angles, L)
            print('goal: ', goal)
            print('bezier curve x and y: ', x, y)            
            print('end effector position: ', forward_position[3])
            if np.allclose([x, y], goal, atol=0.1):
                print('Goooooooooooooal reached')
                joint_angles = inverse_kinematics_arm([goal[0], goal[1]], L)
                arm_plan.append(joint_angles)
                forward_position = forward_kinematics_arm(joint_angles, L)
                print('end effector position: ', forward_position[3])
                return {'configurations': arm_plan}
    return {'configurations': arm_plan}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute safe regions and Bézier curves for a given path.')
    parser.add_argument('input_yaml', type=str, help='Input YAML file with the environment and solution path.')
    parser.add_argument('output_pdf', type=str, help='Output PDF file to save the plot.')
    parser.add_argument('--export-car', type=str, help='Output YAML file to save the car plan.')
    parser.add_argument('--export-arm', type=str, help='Output YAML file to save the arm plan.')

    args = parser.parse_args()

    config = load_config(args.input_yaml)
    solution_path = config['motionplanning']['solutionpath']
    obstacles = config['environment']['obstacles']
    min_env = config['environment']['min'][:2]
    max_env = config['environment']['max'][:2]

    hyperplanes = generate_hyperplanes(solution_path, obstacles)
    regions = compute_safe_regions(solution_path, hyperplanes, min_env, max_env, obstacles)
    
    bezier_curves = compute_bezier_curves(solution_path, regions)
    
    plot_results(config, regions, hyperplanes, args.output_pdf)


    if args.export_car:
        L = 3.0
        start_state= config['motionplanning']['start']

        car_plan, duration = compute_car_plan(bezier_curves, L, start_state)
        
        # assemble result
        result = {
            'plan': {
                'type': 'car',
                'dt': duration,
                'L': 3,
                'states': car_plan['states'],
                'actions': car_plan['actions']
            }
        }

        # write results
        with open(args.export_car, "w") as stream:
            yaml.dump(result, stream, default_flow_style=None, sort_keys=False)
    
    if args.export_arm:
        link_lengths = [1.0, 1.0, 1.0]  
        goal_position = config['motionplanning']['goal'][:2]       
        arm_plan = compute_arm_plan(bezier_curves, link_lengths, goal_position)
        configurations_as_lists = [array.tolist() for array in arm_plan['configurations']]
        result = {
            'plan': {
                'type': 'arm',
                'L': [1,1,1],
                'states': configurations_as_lists
            }
        }
        print('result: ', result)   
        
        # write results
        with open(args.export_arm, "w") as stream:
            yaml.dump(result, stream, default_flow_style=None, sort_keys=False)
        
        


