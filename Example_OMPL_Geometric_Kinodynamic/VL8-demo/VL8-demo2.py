import ompl.base as base
import ompl.geometric as geometric
import ompl.control as control
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from functools import partial

## Implement a car-like robot with (x,y,theta) state space and dynamics

def isStateValid(si, state):
    if not si.satisfiesBounds(state):
        return False
    x = state.getX()
    y = state.getY()
    if x > 0 and x < 1:
        if y > -1 and y < 0:
            return False
    return True

def getCost(si):
    return base.PathLengthOptimizationObjective(si)
    
def statePropagator(start, control, duration, target):
    u1 = control[0]
    u2 = control[1]
    x = start.getX()
    y = start.getY()
    theta = start.getYaw()

    xn = u1*np.cos(theta)
    yn = u1*np.sin(theta)
    thetan = u1*np.tan(u2)

    target.setX(x + duration * xn)
    target.setY(y + duration * yn)
    target.setYaw(theta + duration * thetan)

    return target

########################################
# (1) Setup state space
########################################
space = base.SE2StateSpace()

bounds = base.RealVectorBounds(2)
bounds.setLow(-1)
bounds.setHigh(+1)

space.setBounds(bounds)

########################################
# (2) Setup control space
########################################

uspace = control.RealVectorControlSpace(space, 2)

ubounds = base.RealVectorBounds(2)
ubounds.setLow(0, 0.5)
ubounds.setHigh(0, +1)
ubounds.setLow(1, -1.5)
ubounds.setHigh(1, +1.5)
uspace.setBounds(ubounds)

########################################
# (3) Setup space information
########################################

si = control.SpaceInformation(space, uspace)

si.setStateValidityChecker(base.StateValidityCheckerFn(partial(isStateValid, si)))
si.setStatePropagator(control.StatePropagatorFn(statePropagator))

si.setPropagationStepSize(0.05)

########################################
# (4) Setup problem definition
########################################
pdef = base.ProblemDefinition(si)

start = base.State(space)
start[0]=-1
start[1]=-1
start[2]=0.0
goal = base.State(space)
goal[0]=+1
goal[1]=+1
goal[2]=+0

pdef.setOptimizationObjective(getCost(si))
pdef.setStartAndGoalStates(start, goal, 0.1)
print(si.settings())

########################################
# (5) Setup planner
########################################

planner = control.SST(si)
planner.setProblemDefinition(pdef)
planner.setup()

planner.solve(1.0)

########################################
# (6) Plot solution
########################################

path = pdef.getSolutionPath()
print(path)

path.interpolate()
path.interpolate()
states = path.getStates()

x = list(map(lambda state: state.getX(), states))
y = list(map(lambda state: state.getY(), states))

gca = plt.gca()
gca.add_patch(Rectangle((0, -1), 1, 1))
gca.set_xlim((-1,1))
gca.set_ylim((-1,1))
plt.plot(x,y)
plt.savefig("4.png")
plt.show()

