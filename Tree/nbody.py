import numpy as np
import matplotlib.pyplot as plt
import time
from constants import G

class Node:
    def __init__(self, position, mass, width, topRight):
        self.totalMass = mass
        self.centerOfMass = position
        self.isInternal = False
        self.isExternal = False
        self.width = width
        self.nw = None
        self.ne = None
        self.sw = None
        self.se = None
        self.topRight = topRight

    def markInternal(self):
        self.isInternal = True

    def markExternal(self):
        self.isExternal = True

    def eraseExternal(self):
        self.isExternal = False

    def addNodeToInternal(self, position, mass):
        self.centerOfMass[0] = (self.centerOfMass[0] * self.totalMass + position[0] * mass) / (self.totalMass + mass)
        self.centerOfMass[1] = (self.centerOfMass[1] * self.totalMass + position[1] * mass) / (self.totalMass + mass)
        self.totalMass += mass

def addNodeInSpecifiedDirection(root, targetNode, curPos, curMass, width, curTopRight):
    if root.isInternal:
        root.addNodeToInternal(curPos, curMass)
        if targetNode is not None:
            if targetNode.isExternal:
                return addNodeInExternal(targetNode, curPos, curMass, width / 2, curTopRight)
            else:
                targetNode.addNodeToInternal(curPos, curMass)
                return constructBHTree(targetNode, None, curPos, None, curMass, width / 2, curTopRight)
        else:
            pos = np.array([curPos[0], curPos[1]])
            targetNode = Node(pos, curMass, width / 2, curTopRight)
            targetNode.markExternal()
            return targetNode
    else:
        return addNodeInExternal(root, curPos, curMass, width, curTopRight)

def addNodeInExternal(node, pos, mass, width, topRight):
    node.markInternal()
    node.eraseExternal()
    x1 = node.centerOfMass[0]
    y1 = node.centerOfMass[1]
    mass1 = node.totalMass
    pos1 = np.array([x1, y1])
    cur = constructBHTree(node, None, pos1, None, mass1, width, topRight)
    cur.addNodeToInternal(pos, mass)
    return constructBHTree(node, None, pos, None, mass, width, topRight)

# use Barnes-Hut Tree
def constructBHTree(root, allPos, curPos, allMass, curMass, width, topRight):
    if root is None:
        position = np.array([0, 0])
        mass = 0
        root = Node(position, mass, width, topRight)
        root.markInternal()
        for i in range(len(allPos)):
            constructBHTree(root, None, allPos[i], None, allMass, width, topRight)
    else:
        x = curPos[0]
        y = curPos[1]
        topRightX = topRight[0]
        topRightY = topRight[1]
        if (x >= topRightX - width / 2) & (x <= topRightX) & (y >= topRightY - width / 2) & (y <= topRightY):
            # northeast part
            curTopRightX = topRightX
            curTopRightY = topRightY
            curTopRight = np.array([curTopRightX, curTopRightY])
            root.ne = addNodeInSpecifiedDirection(root, root.ne, curPos, curMass, width, curTopRight)
        elif (x >= topRightX - width) & (x < topRightX - width/2) & (y >= topRightY - width/2) & (y <= topRightY):
            # northwest part
            curTopRightX = topRightX - width / 2
            curTopRightY = topRightY
            curTopRight = np.array([curTopRightX, curTopRightY])
            root.nw = addNodeInSpecifiedDirection(root, root.nw, curPos, curMass, width, curTopRight)
        elif (x >= topRightX - width) & (x < topRightX - width/2) & (y >= topRightY - width) & (y < topRightY - width/2):
            # southwest part
            curTopRightX = topRightX - width / 2
            curTopRightY = topRightY - width / 2
            curTopRight = np.array([curTopRightX, curTopRightY])
            root.sw = addNodeInSpecifiedDirection(root, root.sw, curPos, curMass, width, curTopRight)
        elif (x >= topRightX - width/2) & (x <= topRightX) & (y >= topRightY - width) & (y < topRightY - width/2):
            # southeast part
            curTopRightX = topRightX
            curTopRightY = topRightY - width / 2
            curTopRight = np.array([curTopRightX, curTopRightY])
            root.se = addNodeInSpecifiedDirection(root, root.se, curPos, curMass, width, curTopRight)
        else:
            pass

    return root


def calAccByBHTree(root, pos, theta, G):
    if root is None: return np.array([0,0])
    if root.centerOfMass[0] == pos[0] and root.centerOfMass[1] == pos[1]: return np.array([0,0])
    distx = root.centerOfMass[0] - pos[0]
    disty = root.centerOfMass[1] - pos[1]
    if root.isExternal:
        accx = G * root.totalMass * distx / (distx**2 + disty**2)**1.5
        accy = G * root.totalMass * disty / (distx**2 + disty**2)**1.5
        return np.array([accx, accy])
    accx = 0
    accy = 0
    dis = ((pos[0] - root.centerOfMass[0])**2 + (pos[1] - root.centerOfMass[1])**2)**0.5
    if root.width / dis >= theta:
        packAcc = calAccByBHTree(root.nw, pos, theta, G)
        accx += packAcc[0]
        accy += packAcc[1]
        packAcc = calAccByBHTree(root.ne, pos, theta, G)
        accx += packAcc[0]
        accy += packAcc[1]
        packAcc = calAccByBHTree(root.sw, pos, theta, G)
        accx += packAcc[0]
        accy += packAcc[1]
        packAcc = calAccByBHTree(root.se, pos, theta, G)
        accx += packAcc[0]
        accy += packAcc[1]
    else:
        accx = G * root.totalMass * distx / (distx ** 2 + disty ** 2) ** 1.5
        accy = G * root.totalMass * disty / (distx ** 2 + disty ** 2) ** 1.5
    return np.array([accx, accy])

def getAccFromBHTree(pos, mass, width, theta, G):
    topRight = np.array([width / 2, width / 2])
    # cmass = mass.copy()
    root = constructBHTree(None, pos, None, 1, None, width, topRight)
    size = len(pos)
    accs = np.zeros((size, 2))
    for i in range(size):
        accs[i] = calAccByBHTree(root, pos[i], theta, G)
    return accs

def getAcc(pos, mass, G, softening):

    # positions r = [x,y] for all particles
    x = pos[:, 0:1]
    y = pos[:, 1:2]

    # matrix that stores all pairwise particle separations: r_j - r_i
    dx = x.T - x
    dy = y.T - y

    # matrix that stores 1/r^3 for all particle pairwise particle separations
    inv_r3 = (dx ** 2 + dy ** 2 + softening ** 2)
    inv_r3[inv_r3 > 0] = inv_r3[inv_r3 > 0] ** (-1.5)

    ax = G * (dx * inv_r3) @ mass
    ay = G * (dy * inv_r3) @ mass

    # pack together the acceleration components
    a = np.hstack((ax, ay))

    return a

def accOption(pos, mass, width, theta, G, softening, chooseBHTree):
    if chooseBHTree:
        return getAccFromBHTree(pos, mass, width, theta, G)
    else:
        return getAcc(pos, mass, G, softening)


def checkDefaultDt(new_dt, dt):
    """
    new_dt: new size of timestep
    dt: default timestep
    """
    for i in range(len(new_dt)):
        for j in range(len(new_dt[0])):
            if new_dt[i][j] > 2.0 * dt:
                new_dt[i][j] = 2.0 * dt


def checkDt(new_dt, dt):
    """
    new_dt: new size of timestep
    dt: updated timestep
    """
    for i in range(len(new_dt)):
        for j in range(len(new_dt[0])):
            if new_dt[i][j] > 2 * dt[i][j]:
                new_dt[i][j] = 2 * dt[i][j]


def getDt(acc, dt, defaultT):
    eta = 1.4  # constant associated with timestep
    a_prime = acc / dt
    a_double_prime = a_prime / dt
    new_dt = eta * ((a_prime / acc) ** 2 + a_double_prime / acc) ** (-0.5)
    if defaultT:
        checkDefaultDt(new_dt, dt)
    else:
        checkDt(new_dt, dt)
    return new_dt


def main():

    # parameters
    N = 10000  # Number of particles
    t = 0  # current time of the simulation
    tEnd = 20.0  # time at which simulation ends
    dt = 20  # timestep
    softening = 0  # softening length
    #G = 1.0  # Newton's Gravitational Constant
    plotRealTime = True  # switch on for plotting as the simulation goes along
    theta = 0.5 # threshold value, i.e. theta = s/d, of Barnes-Hut Tree
    width = 1200 # biggest width of figure

    # Generate Initial Conditions
    np.random.seed(133)  # set the random number generator seed

    mass = 20.0 * np.ones((N, 1)) / N  # total mass of particles is 20
    # mass = 20.0 * np.array([[0.4],[0.6]])
    # mass = 20.0 * np.array([[0.4], [0.3], [0.3]])
    # mass = 20.0 * np.array([[0.25], [0.25], [0.25], [0.25]])

    pos = 200 * np.random.randn(N, 2)  # randomly selected positions and velocities
    # pos = np.array([[-1.0,0.1,0.2],[1.0,0.2,0.3]])
    # print(pos)

    vel = np.random.randn(N, 2)
    # vel = np.zeros((N,3))

    # Convert to Center-of-Mass frame
    vel -= np.mean(mass * vel, 0) / np.mean(mass)

    # calculate initial gravitational accelerations
    # acc = getAcc(pos, mass, G, softening)
    # acc = getAccFromBHTree(pos, mass, width, theta, G)
    acc = accOption(pos, mass, width, theta, G, softening, True)
    # print(acc)

    new_dt_init = getDt(acc, dt, 1)

    # calculate initial energy of system
    # KE, PE  = getEnergy( pos, vel, mass, G )

    # number of timesteps
    # Nt = int(np.ceil(tEnd/dt))
    Nt = 100

    # save energies, particle orbits for plotting trails
    pos_save = np.zeros((N, 2, Nt + 1))
    pos_save[:, :, 0] = pos

    # prep figure
    fig = plt.figure(figsize=(15, 15), dpi=80)
    grid = plt.GridSpec(1, 1, wspace=0.0, hspace=0.3)
    ax1 = plt.subplot(grid[0:10, 0])
    # ax2 = plt.subplot(grid[2,0])]
    # 3D
    # fig = plt.figure()
    # ax1 = fig.add_subplot(projection='3d')

    new_dt_pre = new_dt_init

    for i in range(Nt):
        # (1/2) kick
        vel += acc * new_dt_pre / 2.0

        # drift
        pos += vel * new_dt_pre

        # print("i:" + str(i))
        # print(pos)
        # print("vel:")
        # print(vel)
        # print("new_dt:")

        # update accelerations
        # acc = getAcc(pos, mass, G, softening)
        # acc = getAccFromBHTree(pos, mass, width, theta, G)
        acc = accOption(pos, mass, width, theta, G, softening, True)

        new_dt = getDt(acc, new_dt_pre, 0)

        # (1/2) kick
        vel += acc * new_dt / 2.0

        # update time
        t += (new_dt_pre / 2.0 + new_dt / 2.0)
        new_dt_pre = new_dt

        pos_save[:, :, i + 1] = pos

        # plot in real time
        if plotRealTime or (i == Nt - 1):
            plt.sca(ax1)
            plt.cla()
            xx = pos_save[:, 0, max(i - 50, 0):i + 1]
            yy = pos_save[:, 1, max(i - 50, 0):i + 1]
            # colors = ['#ff474c', '#75bbfd', '#10a674', '#8e82fe', '#c20078', '#1e488f']
            # colors = ['#ff474c', '#75bbfd', '#10a674']
            # colors = ['#ff474c', '#75bbfd']
            plt.scatter(xx, yy, s=1, color=[0.8, 0.8, 1])
            plt.scatter(pos[:, 0], pos[:, 1], s=10, c='#ff474c')
            # plt.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c='#ff474c')
            ax1.set(xlim=(-width/2, width/2), ylim=(-width/2, width/2))
            # ax1.set(xlim=(-600, 600), ylim=(-600, 600), zlim=(-600, 600))
            ax1.set_aspect('equal', 'box')
            ax1.set_xticks([-width/2, -width/4, 0, width/4, width/2])
            ax1.set_yticks([-width/2, -width/4, 0, width/4, width/2])
            # ax1.set_zticks([-600, -300, 0, 300, 600])
            plt.xlabel('x', fontsize=14, fontweight='bold')
            plt.ylabel('y', fontsize=14, fontweight='bold')

            plt.pause(0.001)

    # add labels/legend
    plt.sca(ax1)
    plt.xlabel('x', fontsize=14, fontweight='bold')
    plt.ylabel('y', fontsize=14, fontweight='bold')
    # ax1.legend(loc='upper right')

    # Save figure
    plt.savefig('nbody.png', dpi=240)
    plt.show()

    return 0


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")
