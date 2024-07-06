import argparse
import yaml

# collision checking
import numpy as np
import fcl
import meshcat.transformations as tf


def build_collision_manager_env(env):
    manager = fcl.DynamicAABBTreeCollisionManager()
    objs = []
    for o in env["environment"]["obstacles"]:
        if o["type"] == "box":
            p = o["pos"]
            s = o["size"]
            objs.append(fcl.CollisionObject(fcl.Box(s[0], s[1], s[2]), fcl.Transform(p)))
        elif o["type"] == "cylinder":
            p = o["pos"]
            q = o["q"]
            r = o["r"]
            lz = o["lz"]
            objs.append(fcl.CollisionObject(fcl.Cylinder(r, lz), fcl.Transform(q, p)))
        else:
            raise RuntimeError("Unknown obstacle type " + o["type"])


    manager.registerObjects(objs)
    manager.setup()
    return manager

def build_collision_manager_plan_car():
    manager = fcl.DynamicAABBTreeCollisionManager()
    # TODO: this either needs to be specified in the task, or part of the config file
    L = 3.0
    W = 1.5
    H = 1.0
    objs = [fcl.CollisionObject(fcl.Box(L, W, H))]
    manager.registerObjects(objs)
    manager.setup()

    def update(manager, state):
        objs = manager.getObjects()
        obj = objs[0]
        obj.setTranslation(state[0:3])
        # obj.setQuatRotation(q2)
        manager.update(obj)

    return manager, update

def build_collision_manager_plan_arm():
    manager = fcl.DynamicAABBTreeCollisionManager()
    # TODO: this either needs to be specified in the task, or part of the config file
    r = 0.04
    lz = 1.0
    objs = [
        fcl.CollisionObject(fcl.Cylinder(r, lz)),
        fcl.CollisionObject(fcl.Cylinder(r, lz)),
        fcl.CollisionObject(fcl.Cylinder(r, lz))]
    manager.registerObjects(objs)
    manager.setup()

    def update(manager, state):
        objs = manager.getObjects()

        theta_1, theta_2, theta_3 = state
        L = [1,1,1]

        x_1 = L[0]/2*np.cos(theta_1)
        y_1 = L[0]/2*np.sin(theta_1) 
        
        x_2 = L[0]*np.cos(theta_1) + L[1]/2*np.cos(theta_1+theta_2)
        y_2 = L[0]*np.sin(theta_1) + L[1]/2*np.sin(theta_1+theta_2)    

        x_3 = L[0]*np.cos(theta_1) + L[1]*np.cos(theta_1+theta_2) + L[2]/2*np.cos(theta_1+theta_2+theta_3)
        y_3 = L[0]*np.sin(theta_1) + L[1]*np.sin(theta_1+theta_2) + L[2]/2*np.sin(theta_1+theta_2+theta_3)

        offset = np.pi/2
        T1 = tf.translation_matrix([x_1, y_1, 0]).dot(
                tf.euler_matrix(np.pi/2, 0, offset + theta_1))
        t1 = fcl.Transform(T1[0:3,0:3], T1[0:3,3]) # R, t
        objs[0].setTransform(t1)
        
        T2 = tf.translation_matrix([x_2, y_2, 0]).dot(
                tf.euler_matrix(np.pi/2, 0, offset + theta_1+theta_2))
        t2 = fcl.Transform(T2[0:3,0:3], T2[0:3,3]) # R, t
        objs[1].setTransform(t2)
        
        T3 = tf.translation_matrix([x_3, y_3, 0]).dot(
                tf.euler_matrix(np.pi/2, 0, offset + theta_1 + theta_2+ theta_3))
        t3 = fcl.Transform(T3[0:3,0:3], T3[0:3,3]) # R, t
        objs[2].setTransform(t3)
        print(T3)
        manager.update()

    return manager, update

def build_collision_manager_plan(plan):
    if plan["plan"]["type"] == "car":
        return build_collision_manager_plan_car()
    if plan["plan"]["type"] == "arm":
        return build_collision_manager_plan_arm()
    else:
        raise RuntimeError("Unknown type " + plan["plan"]["type"])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('env', help='input YAML file with environment')
    parser.add_argument('plan', help='input YAML file with plan')
    parser.add_argument('output', help='output YAML file with collisions')
    args = parser.parse_args()

    # load input file
    with open(args.env, "r") as stream:
        env = yaml.safe_load(stream)

    with open(args.plan, "r") as stream:
        plan = yaml.safe_load(stream)

    # create collision checking managers
    manager_env = build_collision_manager_env(env)
    manager_plan, update = build_collision_manager_plan(plan)

    # check collisions for each state
    collisions = []
    for state in plan["plan"]["states"]:
        req = fcl.CollisionRequest()
        rdata = fcl.CollisionData(request = req)
        update(manager_plan, np.asarray(state))
        manager_env.collide(manager_plan, rdata, fcl.defaultCollisionCallback)
        collisions.append(rdata.result.is_collision)

    # output result
    result = {
        'collisions': collisions
    }

    # write results
    with open(args.output, "w") as stream:
        yaml.dump(result, stream)

if __name__ == "__main__":
    main()
