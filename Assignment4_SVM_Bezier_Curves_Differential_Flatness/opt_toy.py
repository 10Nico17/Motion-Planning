import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

# Define start and goal points
q_start = np.array([0.5, 2.5])
q_goal = np.array([9.5, 0.5])

# Define the regions
region1_x = [0, 6]
region1_y = [2, 3]
region2_x = [4, 6]
region2_y = [0, 5]
region3_x = [4, 10]
region3_y = [0, 1]

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

def obstacle_avoidance_cost(point, bottom_left, top_right):
    #return cp.norm(cp.maximum(0, bottom_left - P), 2) + cp.norm(cp.maximum(0, P - top_right), 2)
    w=2.5
    return w*(cp.norm(cp.maximum(0, bottom_left - point), 2) + cp.norm(cp.maximum(0, point - top_right), 2))

objective = cp.Minimize(
    cp.sum_squares(P1 - P0) + cp.sum_squares(P2 - P1) + cp.sum_squares(P3 - P2) +
    cp.sum_squares(P4 - P3) + cp.sum_squares(P5 - P4) + cp.sum_squares(P6 - P5) +
    cp.sum_squares(P7 - P6) + cp.sum_squares(P8 - P7) + cp.sum_squares(P9 - P8) +
    cp.sum_squares(P10 - P9) + cp.sum_squares(P11 - P10)
     +
    obstacle_avoidance_cost(P1, np.array([0, 0]), np.array([4, 2])) +
    obstacle_avoidance_cost(P1, np.array([1, 3]), np.array([4, 5])) +
    obstacle_avoidance_cost(P1, np.array([6, 1]), np.array([7, 5])) + 

    obstacle_avoidance_cost(P2, np.array([0, 0]), np.array([4, 2])) +
    obstacle_avoidance_cost(P2, np.array([1, 3]), np.array([4, 5])) +
    obstacle_avoidance_cost(P2, np.array([6, 1]), np.array([7, 5])) + 

    obstacle_avoidance_cost(P3, np.array([0, 0]), np.array([4, 2])) +
    obstacle_avoidance_cost(P3, np.array([1, 3]), np.array([4, 5])) +
    obstacle_avoidance_cost(P3, np.array([6, 1]), np.array([7, 5])) + 

    obstacle_avoidance_cost(P4, np.array([0, 0]), np.array([4, 2])) +
    obstacle_avoidance_cost(P4, np.array([1, 3]), np.array([4, 5])) +
    obstacle_avoidance_cost(P4, np.array([6, 1]), np.array([7, 5])) + 

    obstacle_avoidance_cost(P5, np.array([0, 0]), np.array([4, 2])) +
    obstacle_avoidance_cost(P5, np.array([1, 3]), np.array([4, 5])) +
    obstacle_avoidance_cost(P5, np.array([6, 1]), np.array([7, 5])) + 

    obstacle_avoidance_cost(P6, np.array([0, 0]), np.array([4, 2])) +
    obstacle_avoidance_cost(P6, np.array([1, 3]), np.array([4, 5])) +
    obstacle_avoidance_cost(P6, np.array([6, 1]), np.array([7, 5])) + 

    obstacle_avoidance_cost(P7, np.array([0, 0]), np.array([4, 2])) +
    obstacle_avoidance_cost(P7, np.array([1, 3]), np.array([4, 5])) +
    obstacle_avoidance_cost(P7, np.array([6, 1]), np.array([7, 5])) + 

    obstacle_avoidance_cost(P8, np.array([0, 0]), np.array([4, 2])) +
    obstacle_avoidance_cost(P8, np.array([1, 3]), np.array([4, 5])) +
    obstacle_avoidance_cost(P8, np.array([6, 1]), np.array([7, 5])) + 

    obstacle_avoidance_cost(P9, np.array([0, 0]), np.array([4, 2])) +
    obstacle_avoidance_cost(P9, np.array([1, 3]), np.array([4, 5])) +
    obstacle_avoidance_cost(P9, np.array([6, 1]), np.array([7, 5])) + 

    obstacle_avoidance_cost(P10, np.array([0, 0]), np.array([4, 2])) +
    obstacle_avoidance_cost(P10, np.array([1, 3]), np.array([4, 5])) +
    obstacle_avoidance_cost(P10, np.array([6, 1]), np.array([7, 5])) 


)

d_min = 0.1

# Constraints
constraints = [
    P0 == q_start,  # Ensure P0 is the start point
    P1[0] >= region1_x[0], P1[0] <= region1_x[1], P1[1] >= region1_y[0], P1[1] <= region1_y[1],
    P2[0] >= region1_x[0], P2[0] <= region1_x[1], P2[1] >= region1_y[0], P2[1] <= region1_y[1],
    P3[0] >= max(region1_x[0], region2_x[0]), P3[0] <= min(region1_x[1], region2_x[1]), P3[1] >= max(region1_y[0], region2_y[0]), P3[1] <= min(region1_y[1], region2_y[1]),
    P4 == P3,
    P5[0] >= region2_x[0], P5[0] <= region2_x[1], P5[1] >= region2_y[0], P5[1] <= region2_y[1],
    P6[0] >= region2_x[0], P6[0] <= region2_x[1], P6[1] >= region2_y[0], P6[1] <= region2_y[1],
    P7[0] >= max(region2_x[0], region3_x[0]), P7[0] <= min(region2_x[1], region3_x[1]), P7[1] >= max(region2_y[0], region3_y[0]), P7[1] <= min(region2_y[1], region3_y[1]),
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

print('Segment 1 control points: ', P0_opt, P1_opt, P2_opt, P3_opt)
print('Segment 2 control points: ', P4_opt, P5_opt, P6_opt, P7_opt)
print('Segment 3 control points: ', P8_opt, P9_opt, P10_opt, P11_opt)

# Plot the optimized path
plt.figure()

# Define the obstacles as rectangles (bottom-left corner and top-right corner)
obstacles = [
    {'bottom_left': np.array([0, 0]), 'top_right': np.array([4, 2]), 'label': 'Obstacle 1'},
    {'bottom_left': np.array([1, 3]), 'top_right': np.array([4, 5]), 'label': 'Obstacle 2'},
    {'bottom_left': np.array([6, 1]), 'top_right': np.array([7, 5]), 'label': 'Obstacle 3'},
]

# Plot obstacles
for obs in obstacles:
    rect = plt.Rectangle(obs['bottom_left'], obs['top_right'][0] - obs['bottom_left'][0], obs['top_right'][1] - obs['bottom_left'][1], color='r', alpha=0.5, label=obs['label'])
    plt.gca().add_artist(rect)

# Plot regions
region1 = plt.Rectangle((0, 2), 6, 1, color='b', alpha=0.2, label='Region 1')
region2 = plt.Rectangle((4, 0), 2, 5, color='g', alpha=0.2, label='Region 2')
region3 = plt.Rectangle((4, 0), 6, 1, color='y', alpha=0.2, label='Region 3')

plt.gca().add_artist(region1)
plt.gca().add_artist(region2)
plt.gca().add_artist(region3)

plt.scatter(q_start[0], q_start[1], color='g', s=100, label='Start')
plt.scatter(q_goal[0], q_goal[1], color='purple', s=100, label='Goal')

# Plot the control points
control_points = [P1_opt, P2_opt, P3_opt, P4_opt, P5_opt, P6_opt, P7_opt, P8_opt, P9_opt, P10_opt]
control_points_labels = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10']
for point, label in zip(control_points, control_points_labels):
    plt.scatter(point[0], point[1], color='black')
    plt.text(point[0], point[1], label, fontsize=12, ha='right')

# Plot the Bézier curve segments
def bezier_curve(P0, P1, P2, P3):
    t = np.linspace(0, 1, 100)
    bezier_curve_x = (1-t)**3 * P0[0] + 3*(1-t)**2 * t * P1[0] + 3*(1-t)*t**2 * P2[0] + t**3 * P3[0]
    bezier_curve_y = (1-t)**3 * P0[1] + 3*(1-t)**2 * t * P1[1] + 3*(1-t)*t**2 * P2[1] + t**3 * P3[1]
    return bezier_curve_x, bezier_curve_y

curve1_x, curve1_y = bezier_curve(P0_opt, P1_opt, P2_opt, P3_opt)
curve2_x, curve2_y = bezier_curve(P4_opt, P5_opt, P6_opt, P7_opt)
curve3_x, curve3_y = bezier_curve(P8_opt, P9_opt, P10_opt, P11_opt)

plt.plot(curve1_x, curve1_y, 'b', label='Segment 1')
plt.plot(curve2_x, curve2_y, 'g', label='Segment 2')
plt.plot(curve3_x, curve3_y, 'y', label='Segment 3')

plt.legend()
plt.grid(True)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Optimized Bézier Curve Path with Control Points')
plt.axis([0, 10, 0, 6])
plt.xticks(np.arange(0, 10.5, 1))
plt.yticks(np.arange(0, 6.5, 1))
plt.gca().set_aspect('equal', adjustable='box')

# Save the plot
plt.savefig('opt_toy.pdf')
plt.show()
