from ompl import base as base
from ompl import geometric as geometric

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

rect = Rectangle((-0.5, -0.5), 1, 0.5)

def isStateValid(state):
    x = state[0]
    y = state[1]
    xl = rect.get_x()
    yl = rect.get_y()
    yh = yl + rect.get_height()
    xh = xl + rect.get_width()

    if x >= xl and x <= xh:
        if y >= yl and y <= yh:
            return False
    return True

def getCost(si):
    return base.PathLengthOptimizationObjective(si)

##########################################
### State Space 
##########################################
space = base.RealVectorStateSpace(2)

### Declare bounds
bounds = base.RealVectorBounds(2)
bounds.setLow(-1)
bounds.setHigh(+1)
space.setBounds(bounds)
print(space.settings())

##########################################
### Space Information
##########################################
si = base.SpaceInformation(space)

### Set state validity checker
si.setStateValidityChecker(base.StateValidityCheckerFn(isStateValid))

##########################################
### Problem definition
##########################################
pdef = base.ProblemDefinition(si)

start = base.State(space)
start[0] = -1
start[1] = -1

goal = base.State(space)
goal[0] = +1
goal[1] = +1

pdef.setStartAndGoalStates(start, goal)
pdef.setOptimizationObjective(getCost(si))

##########################################
### Planner
##########################################
print(si.settings())
print("Planning from %s to %s" % (start, goal))
planner = geometric.RRTstar(si)
planner.setProblemDefinition(pdef)

planner.setup()
planner.solve(0.3) ##time to solve
path = pdef.getSolutionPath()
print("Solution found: %s (length: %s)" % (path, path.length()))

states = path.getStates()
x = list(map(lambda state: state[0], states))
y = list(map(lambda state: state[1], states))

currentAxis = plt.gca()
currentAxis.add_patch(rect)
plt.plot(x,y)
plt.savefig('2.png')
plt.show()

