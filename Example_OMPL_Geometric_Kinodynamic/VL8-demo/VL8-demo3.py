import ompl.base as base
import ompl.geometric as geometric
import ompl.tools as tools
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

## 2d point robot in the plane with one obstacle (Benchmark Edition)
def isValid(state):
    x=state[0]
    y=state[1]
    if x > -0.5 and x < 0.5:
        if y > -0.5 and y < 0.5:
            return False
    return True

def getCost(si):
    return base.PathLengthOptimizationObjective(si)

#####################################
# (1) Setup state space
#####################################
space = base.RealVectorStateSpace(2)

bounds = base.RealVectorBounds(2)
bounds.setLow(-1)
bounds.setHigh(+1)

space.setBounds(bounds)

ss = geometric.SimpleSetup(space)
ss.setStateValidityChecker(base.StateValidityCheckerFn(isValid))

#####################################
# (2) Setup problem definition
#####################################
start = base.State(space)
start[0] = -1
start[1] = -1

goal = base.State(space)
goal[0] = +1
goal[1] = +1

si = ss.getSpaceInformation()
ss.setOptimizationObjective(getCost(si))
ss.setStartAndGoalStates(start, goal)

#####################################
# (3) Setup benchmarking
#####################################
benchmark = tools.Benchmark(ss)
query = benchmark.Request()
query.maxTime = 2.0
query.runCount = 5

benchmark.addPlanner(geometric.RRT(si))
benchmark.addPlanner(geometric.RRTConnect(si))
benchmark.addPlanner(geometric.EST(si))

benchmark.benchmark(query)
result = benchmark.getRecordedExperimentData()
print(result)
benchmark.saveResultsToFile("benchmark.log")
