import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time
import random
import math


class Node:
    def __init__(self, position, mass, velocity, width, topRight):
        self.totalMass = mass
        self.centerOfMass = position
        self.velocity = velocity
        self.isInternal = False
        self.isExternal = False
        self.width = width
        self.pnw = None  # positive northwest
        self.nnw = None  # negative northwest
        self.pne = None  # ...
        self.nne = None
        self.psw = None
        self.nsw = None
        self.pse = None
        self.nse = None
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
        self.centerOfMass[2] = (self.centerOfMass[2] * self.totalMass + position[2] * mass) / (self.totalMass + mass)
        self.totalMass += mass


def addNodeInSpecifiedDirection(root, targetNode, curPos, curMass, curVel, width, curTopRight):
    if root.isInternal:
        root.addNodeToInternal(curPos, curMass)
        if targetNode is not None:
            if targetNode.isExternal:
                return addNodeInExternal(targetNode, curPos, curMass, curVel, width / 2, curTopRight)
            else:
                targetNode.addNodeToInternal(curPos, curMass)
                return constructBHTree(targetNode, None, curPos, None, curMass, None, curVel, width / 2, curTopRight)
        else:
            pos = np.array([curPos[0], curPos[1], curPos[2]])
            vel = np.array([curVel[0], curVel[1], curVel[2]])
            targetNode = Node(pos, curMass, vel, width / 2, curTopRight)
            targetNode.markExternal()
            return targetNode
    else:
        return addNodeInExternal(root, curPos, curMass, curVel, width, curTopRight)


def addNodeInExternal(node, pos, mass, vel, width, topRight):
    node.markInternal()
    node.eraseExternal()
    x1 = node.centerOfMass[0]
    y1 = node.centerOfMass[1]
    z1 = node.centerOfMass[2]
    mass1 = node.totalMass
    pos1 = np.array([x1, y1, z1])
    vel1 = node.velocity[0]
    vel2 = node.velocity[1]
    vel3 = node.velocity[2]
    velocity1 = np.array([vel1, vel2, vel3])
    cur = constructBHTree(node, None, pos1, None, mass1, None, velocity1, width, topRight)
    cur.addNodeToInternal(pos, mass)
    return constructBHTree(node, None, pos, None, mass, None, vel, width, topRight)


# use Barnes-Hut Tree
def constructBHTree(root, allPos, curPos, allMass, curMass, allVel, curVel, width, topRight):
    if root is None:
        position = np.array([0, 0, 0])
        velcity = np.array([0, 0, 0])
        mass = 0
        root = Node(position, mass, velcity, width, topRight)
        root.markInternal()
        for i in range(len(allPos)):
            constructBHTree(root, None, allPos[i], None, allMass[i][0], None, allVel[i], width, topRight)
    else:
        x = curPos[0]
        y = curPos[1]
        z = curPos[2]
        topRightX = topRight[0]
        topRightY = topRight[1]
        topRightZ = topRight[2]
        if (x >= topRightX - width / 2) & (x <= topRightX) & (y >= topRightY - width / 2) & (y <= topRightY) & (
                z >= topRightZ - width / 2) & (z <= topRightZ):
            # + northeast part
            curTopRightX = topRightX
            curTopRightY = topRightY
            curTopRightZ = topRightZ
            curTopRight = np.array([curTopRightX, curTopRightY, curTopRightZ])
            root.pne = addNodeInSpecifiedDirection(root, root.pne, curPos, curMass, curVel, width, curTopRight)
        elif (x >= topRightX - width / 2) & (x <= topRightX) & (y >= topRightY - width / 2) & (y <= topRightY) & (
                z >= topRightZ - width) & (z < topRightZ - width / 2):
            # - northeast part
            curTopRightX = topRightX
            curTopRightY = topRightY
            curTopRightZ = topRightZ - width / 2
            curTopRight = np.array([curTopRightX, curTopRightY, curTopRightZ])
            root.nne = addNodeInSpecifiedDirection(root, root.nne, curPos, curMass, curVel, width, curTopRight)
        elif (x >= topRightX - width) & (x < topRightX - width / 2) & (y >= topRightY - width / 2) & (
                y <= topRightY) & (z >= topRightZ - width / 2) & (z <= topRightZ):
            # + northwest part
            curTopRightX = topRightX - width / 2
            curTopRightY = topRightY
            curTopRightZ = topRightZ
            curTopRight = np.array([curTopRightX, curTopRightY, curTopRightZ])
            root.pnw = addNodeInSpecifiedDirection(root, root.pnw, curPos, curMass, curVel, width, curTopRight)
        elif (x >= topRightX - width) & (x < topRightX - width / 2) & (y >= topRightY - width / 2) & (
                y <= topRightY) & (z >= topRightZ - width) & (z < topRightZ - width / 2):
            # - northwest part
            curTopRightX = topRightX - width / 2
            curTopRightY = topRightY
            curTopRightZ = topRightZ - width / 2
            curTopRight = np.array([curTopRightX, curTopRightY, curTopRightZ])
            root.nnw = addNodeInSpecifiedDirection(root, root.nnw, curPos, curMass, curVel, width, curTopRight)
        elif (x >= topRightX - width) & (x < topRightX - width / 2) & (y >= topRightY - width) & (
                y < topRightY - width / 2) & (z >= topRightZ - width / 2) & (z <= topRightZ):
            # + southwest part
            curTopRightX = topRightX - width / 2
            curTopRightY = topRightY - width / 2
            curTopRightZ = topRightZ
            curTopRight = np.array([curTopRightX, curTopRightY, curTopRightZ])
            root.psw = addNodeInSpecifiedDirection(root, root.psw, curPos, curMass, curVel, width, curTopRight)
        elif (x >= topRightX - width) & (x < topRightX - width / 2) & (y >= topRightY - width) & (
                y < topRightY - width / 2) & (z >= topRightZ - width) & (z < topRightZ - width / 2):
            # - southwest part
            curTopRightX = topRightX - width / 2
            curTopRightY = topRightY - width / 2
            curTopRightZ = topRightZ - width / 2
            curTopRight = np.array([curTopRightX, curTopRightY, curTopRightZ])
            root.nsw = addNodeInSpecifiedDirection(root, root.nsw, curPos, curMass, curVel, width, curTopRight)
        elif (x >= topRightX - width / 2) & (x <= topRightX) & (y >= topRightY - width) & (
                y < topRightY - width / 2) & (z >= topRightZ - width / 2) & (z <= topRightZ):
            # + southeast part
            curTopRightX = topRightX
            curTopRightY = topRightY - width / 2
            curTopRightZ = topRightZ
            curTopRight = np.array([curTopRightX, curTopRightY, curTopRightZ])
            root.pse = addNodeInSpecifiedDirection(root, root.pse, curPos, curMass, curVel, width, curTopRight)
        elif (x >= topRightX - width / 2) & (x <= topRightX) & (y >= topRightY - width) & (
                y < topRightY - width / 2) & (z >= topRightZ - width) & (z < topRightZ - width / 2):
            # - southeast part
            curTopRightX = topRightX
            curTopRightY = topRightY - width / 2
            curTopRightZ = topRightZ - width / 2
            curTopRight = np.array([curTopRightX, curTopRightY, curTopRightZ])
            root.nse = addNodeInSpecifiedDirection(root, root.nse, curPos, curMass, curVel, width, curTopRight)
        else:
            pass

    return root


def calAccByBHTree(root, pos, theta, G):
    if root is None: return np.array([0, 0, 0])
    if root.centerOfMass[0] == pos[0] and root.centerOfMass[1] == pos[1] and root.centerOfMass[2] == pos[
        2]: return np.array([0, 0, 0])
    distx = root.centerOfMass[0] - pos[0]
    disty = root.centerOfMass[1] - pos[1]
    distz = root.centerOfMass[2] - pos[2]
    if root.isExternal:
        accx = G * root.totalMass * distx / (distx ** 2 + disty ** 2 + distz ** 2) ** 1.5
        accy = G * root.totalMass * disty / (distx ** 2 + disty ** 2 + distz ** 2) ** 1.5
        accz = G * root.totalMass * distz / (distx ** 2 + disty ** 2 + distz ** 2) ** 1.5
        return np.array([accx, accy, accz])
    accx = 0
    accy = 0
    accz = 0
    dis = ((pos[0] - root.centerOfMass[0]) ** 2 + (pos[1] - root.centerOfMass[1]) ** 2 + (
            pos[2] - root.centerOfMass[2]) ** 2) ** 0.5
    if root.width / dis >= theta:
        packAcc = calAccByBHTree(root.pnw, pos, theta, G)
        accx += packAcc[0]
        accy += packAcc[1]
        accz += packAcc[2]
        packAcc = calAccByBHTree(root.nnw, pos, theta, G)
        accx += packAcc[0]
        accy += packAcc[1]
        accz += packAcc[2]
        packAcc = calAccByBHTree(root.pne, pos, theta, G)
        accx += packAcc[0]
        accy += packAcc[1]
        accz += packAcc[2]
        packAcc = calAccByBHTree(root.nne, pos, theta, G)
        accx += packAcc[0]
        accy += packAcc[1]
        accz += packAcc[2]
        packAcc = calAccByBHTree(root.psw, pos, theta, G)
        accx += packAcc[0]
        accy += packAcc[1]
        accz += packAcc[2]
        packAcc = calAccByBHTree(root.nsw, pos, theta, G)
        accx += packAcc[0]
        accy += packAcc[1]
        accz += packAcc[2]
        packAcc = calAccByBHTree(root.pse, pos, theta, G)
        accx += packAcc[0]
        accy += packAcc[1]
        accz += packAcc[2]
        packAcc = calAccByBHTree(root.nse, pos, theta, G)
        accx += packAcc[0]
        accy += packAcc[1]
        accz += packAcc[2]
    else:
        accx = G * root.totalMass * distx / (distx ** 2 + disty ** 2 + distz ** 2) ** 1.5
        accy = G * root.totalMass * disty / (distx ** 2 + disty ** 2 + distz ** 2) ** 1.5
        accz = G * root.totalMass * distz / (distx ** 2 + disty ** 2 + distz ** 2) ** 1.5
    return np.array([accx, accy, accz])


def getAccFromBHTree(root, pos, theta, G):
    size = len(pos)
    accs = np.zeros((size, 3))
    for i in range(size):
        accs[i] = calAccByBHTree(root, pos[i], theta, G)
    return accs


def getAcc(pos, mass, G, softening):
    """
    :param pos: size: N*3
    :param mass: size: N*1
    :param G: Newton's Gravitational Constant
    :param softening: softening length
    :return: acceleration and size: N*3
    """
    # positions r = [x,y,z] for all objects
    x = pos[:, 0:1]
    y = pos[:, 1:2]
    z = pos[:, 2:3]

    # matrix that stores all pairwise particle separations: r_j - r_i
    dx = x.T - x
    dy = y.T - y
    dz = z.T - z

    # matrix that stores 1/r^3 for all particle pairwise particle separations
    inv_r3 = (dx ** 2 + dy ** 2 + dz ** 2 + softening ** 2)
    inv_r3[inv_r3 > 0] = inv_r3[inv_r3 > 0] ** (-1.5)

    ax = G * (dx * inv_r3) @ mass
    ay = G * (dy * inv_r3) @ mass
    az = G * (dz * inv_r3) @ mass

    # pack together the acceleration components
    a = np.hstack((ax, ay, az))

    return a


def accOption(root, pos, mass, theta, G, softening, chooseBHTree):
    if chooseBHTree:
        return getAccFromBHTree(root, pos, theta, G)
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


def getDt(acc, eta, dt, defaultT):
    a_prime = acc / dt
    a_double_prime = a_prime / dt
    new_dt = eta * ((a_prime / acc) ** 2 + a_double_prime / acc) ** (-0.5)
    if defaultT:
        checkDefaultDt(new_dt, dt)
    else:
        checkDt(new_dt, dt)
    return new_dt


def generate_random_number():
    """
    Generates a normalized random number x, with a uniform probability distribution between 0 and 1.

    Returns:
        float: A random number uniformly distributed between 0 and 1.
    """
    return random.random()


def plummerInitForSingle(G, M, R):
    x1 = generate_random_number()
    r = (x1 ** (-2 / 3) - 1) ** (-1 / 2)
    x2 = generate_random_number()
    x3 = generate_random_number()
    pos_z = (1 - 2 * x2) * r
    pos_x = ((r ** 2 - pos_z ** 2) ** (1 / 2)) * math.cos(2 * math.pi * x3)
    pos_y = ((r ** 2 - pos_z ** 2) ** (1 / 2)) * math.sin(2 * math.pi * x3)
    x4 = generate_random_number()
    x5 = generate_random_number()
    while 0.1 * x5 >= (x4 ** 2) * ((1 - x4 ** 2) ** (7 / 2)):
        x4 = generate_random_number()
        x5 = generate_random_number()
    v_e = (2 ** (1 / 2)) * ((1 + r ** 2) ** (-1 / 4))
    v = x4 * v_e
    x6 = generate_random_number()
    x7 = generate_random_number()
    vel_z = (1 - 2 * x6) * v
    vel_x = (v ** 2 - vel_z ** 2) ** (1 / 2) * math.cos(2 * math.pi * x7)
    vel_y = (v ** 2 - vel_z ** 2) ** (1 / 2) * math.sin(2 * math.pi * x7)
    mass = (r ** 3) * ((1 + r ** 2) ** (-3 / 2))

    energy = -(3 * math.pi / 64) * G * (M ** 2) * (R ** (-1))
    pos_z *= (3 * math.pi / 64) * (M ** 2) * (abs(energy) ** (-1))
    pos_x *= (3 * math.pi / 64) * (M ** 2) * (abs(energy) ** (-1))
    pos_y *= (3 * math.pi / 64) * (M ** 2) * (abs(energy) ** (-1))
    vel_z *= (64 / (3 * math.pi)) * (abs(energy) ** (1 / 2)) * (M ** (-1 / 2))
    vel_x *= (64 / (3 * math.pi)) * (abs(energy) ** (1 / 2)) * (M ** (-1 / 2))
    vel_y *= (64 / (3 * math.pi)) * (abs(energy) ** (1 / 2)) * (M ** (-1 / 2))
    mass *= M
    return np.array([pos_x, pos_y, pos_z]), np.array([vel_x, vel_y, vel_z]), np.array([mass])


def plummerInit(N, G, M, R):
    pos = np.zeros((N, 3))
    vel = np.zeros((N, 3))
    mass = np.zeros((N, 1))
    for i in range(N):
        posTemp, velTemp, massTemp = plummerInitForSingle(G, M, R)
        pos[i] = posTemp
        vel[i] = velTemp
        mass[i] = massTemp
    # Convert to Center-of-Mass frame
    vel -= np.mean(mass * vel, 0) / np.mean(mass)
    # use the same masses
    mass2 = M * np.ones((N, 1)) / N
    # vel -= np.mean(mass_test * vel, 0) / np.mean(mass_test)
    return pos, vel, mass2


def generalInit(N, M, scaleInitPos):
    np.random.seed(133)  # set the random number generator seed
    pos = scaleInitPos * np.random.randn(N, 3)
    vel = np.random.randn(N, 3)
    mass = M * np.ones((N, 1)) / N
    # Convert to Center-of-Mass frame
    vel -= np.mean(mass * vel, 0) / np.mean(mass)
    return pos, vel, mass


def chooseInitCondition(N, G, M, R, scaleInitPos, choosePlummerInit):
    if choosePlummerInit:
        return plummerInit(N, G, M, R)
    else:
        return generalInit(N, M, scaleInitPos)


def determineMerger(pos1, vel1, pos2, vel2, mass, C, G, maxb):
    Rs = 2 * G * mass / (C ** 2)  # Schwarzschild radius
    pos_rel = pos2 - pos1  # relative position
    vel_rel = vel2 - vel1  # relative velocity
    sigma = (math.pi * (85 * math.pi / 3) ** (2 / 7) * (Rs ** 2) * (np.linalg.norm(vel_rel) / C) ** (-18 / 7)) ** (
        0.5)  # cross-section
    b = np.linalg.norm(np.cross(pos_rel, vel_rel)) / np.linalg.norm(vel_rel)  # impact parameter
    if b < sigma and b < maxb:
        print("b: " + str(b) + ", sigma: " + str(sigma))
        return True
    return False


def getInfoMergerByBrute(mergeStats, pos, vel, mass, N, C, G, maxb):
    for i in range(N):
        row = []
        for j in range(N):
            if i == j: continue
            res = determineMerger(pos[i], vel[i], pos[j], vel[j], mass[i], C, G, maxb)
            if res:
                row.append(np.array([pos[j][0], pos[j][1], pos[j][2]]))
        mergeStats.append(row)


def hasMergerByTree(root, pos, vel, mass, row, theta, C, G, maxb):
    if root is None: return
    if root.centerOfMass[0] == pos[0] and root.centerOfMass[1] == pos[1] and root.centerOfMass[2] == pos[2]: return
    if root.isExternal:
        res = determineMerger(pos, vel, root.centerOfMass, root.velocity, mass, C, G, maxb)
        if res:
            row.append(np.array([root.centerOfMass[0], root.centerOfMass[1], root.centerOfMass[2]]))
        return
    dis = ((pos[0] - root.centerOfMass[0]) ** 2 + (pos[1] - root.centerOfMass[1]) ** 2 + (
            pos[2] - root.centerOfMass[2]) ** 2) ** 0.5
    if root.width / dis >= theta:
        hasMergerByTree(root.pnw, pos, vel, mass, row, theta, C, G, maxb)
        hasMergerByTree(root.nnw, pos, vel, mass, row, theta, C, G, maxb)
        hasMergerByTree(root.pne, pos, vel, mass, row, theta, C, G, maxb)
        hasMergerByTree(root.nne, pos, vel, mass, row, theta, C, G, maxb)
        hasMergerByTree(root.psw, pos, vel, mass, row, theta, C, G, maxb)
        hasMergerByTree(root.nsw, pos, vel, mass, row, theta, C, G, maxb)
        hasMergerByTree(root.pse, pos, vel, mass, row, theta, C, G, maxb)
        hasMergerByTree(root.nse, pos, vel, mass, row, theta, C, G, maxb)
    else:
        # treat as no merger
        return


def getInfoMergerByTree(mergeStats, root, pos, vel, mass, N, C, G, maxb, theta):
    for i in range(N):
        row = []
        hasMergerByTree(root, pos[i], vel[i], mass[i], row, theta, C, G, maxb)
        mergeStats.append(row)


def getInfoMerger(mergeStatsByBrute, mergeStatsByTree, root, pos, vel, mass, N, C, G, maxb, theta):
    getInfoMergerByBrute(mergeStatsByBrute, pos, vel, mass, N, C, G, maxb)
    getInfoMergerByTree(mergeStatsByTree, root, pos, vel, mass, N, C, G, maxb, theta)


def drawBoxForMerger(point1, point2, ax, color):
    # Extract coordinates of the two points
    x1, y1, z1 = point1
    x2, y2, z2 = point2

    # Calculate the corners of the box
    x_min, x_max = min(x1, x2), max(x1, x2)
    y_min, y_max = min(y1, y2), max(y1, y2)
    z_min, z_max = min(z1, z2), max(z1, z2)

    # Create the vertices of the box
    vertices = np.array([
        [x_min, y_min, z_min],
        [x_max, y_min, z_min],
        [x_max, y_max, z_min],
        [x_min, y_max, z_min],
        [x_min, y_min, z_max],
        [x_max, y_min, z_max],
        [x_max, y_max, z_max],
        [x_min, y_max, z_max],
    ])
    # print("vertices: ", vertices)

    # Define the edges of the box
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7]
    ]

    # Plot the box
    for edge in edges:
        ax.plot(vertices[edge, 0], vertices[edge, 1], vertices[edge, 2], color=color, linewidth=3)

    # # Plot the two points
    # ax.scatter([x1, x2], [y1, y2], [z1, z2], color='black', s=100)


def getCandidateMerger(mergeStats, pos, N):
    candidateMerger = []
    for i in range(N):
        leni = len(mergeStats[i])
        # only choose two objects case
        if leni == 1:
            dis = np.linalg.norm(pos[i] - mergeStats[i][0])
            if pos[i][0] < mergeStats[i][0][0]:
                candidateMerger.append([pos[i], mergeStats[i][0], dis])
            else:
                candidateMerger.append([mergeStats[i][0], pos[i], dis])
    sortCandidate = sorted(candidateMerger, key=lambda x: x[2])
    return sortCandidate


def drawLine(point1, point2, ax, color):
    x1, y1, z1 = point1
    x2, y2, z2 = point2
    ax.plot([x1, x2], [y1, y2], [z1, z2], color=color, linewidth=4)


def drawMerger(candidateMerger, width, ax, color, method):
    lencan = len(candidateMerger)
    if lencan != 0:
        cur1 = candidateMerger[0][0]
        cur2 = candidateMerger[0][1]
        print(method)
        print("pos1:" + str(cur1) + ", pos2: " + str(cur2) + ", distance: " + str(candidateMerger[0][2]))
        if method == 'Brute force to search:':
            drawLine(cur1, cur2, ax, color)
        else:
            drawBoxForMerger(cur1, cur2, ax, color)
        for i in range(1, lencan):
            tmp1 = candidateMerger[i][0]
            tmp2 = candidateMerger[i][1]
            if (cur1 == tmp1).all() or (cur1 == tmp2).all() or (cur2 == tmp1).all() or (cur2 == tmp2).all():
                continue
            else:
                cur1 = tmp1
                cur2 = tmp2
                print("pos1:" + str(cur1) + ", pos2: " + str(cur2) + ", distance: " + str(candidateMerger[i][2]))
                if method == 'Brute force to search:':
                    drawLine(cur1, cur2, ax, color)
                else:
                    drawBoxForMerger(cur1, cur2, ax, color)

        ax.set_xlim([-width / 2, width / 2])
        ax.set_ylim([-width / 2, width / 2])
        ax.set_zlim([-width / 2, width / 2])
        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=20)
        ax.tick_params(axis='z', labelsize=20)
        ax.set_xlabel('x', fontsize=25, fontweight='bold', labelpad=20)
        ax.set_ylabel('y', fontsize=25, fontweight='bold', labelpad=20)
        ax.set_zlabel('z', fontsize=25, fontweight='bold', labelpad=20)
        # Adjust transparency of axis panes (0.0 to 1.0)  # Remove grid from 3D plot
        ax.xaxis.pane.set_alpha(0.35)
        ax.yaxis.pane.set_alpha(0.35)
        ax.zaxis.pane.set_alpha(0.35)
        ax.grid(False)
        # plt.pause(0.001)
        return True
    return False

def analysisAndDrawMerger(mergeStatsByBrute, mergeStatsByTree, pos, N, width, ax):
    candidateMergerByBrute = getCandidateMerger(mergeStatsByBrute, pos, N)
    candidateMergerByTree = getCandidateMerger(mergeStatsByTree, pos, N)
    hasMergerByBrute = drawMerger(candidateMergerByBrute, width, ax, 'black', 'Brute force to search:')
    hasMergerByTree = drawMerger(candidateMergerByTree, width, ax, 'red', 'BHTree to search:')
    if hasMergerByBrute or hasMergerByTree:
        plt.show()
        sys.exit("Found merger")


def main():
    # parameters
    N = 60  # Number of objects
    dt = 0.1  # initial timestep
    softening = 0  # softening length
    G = 1  # 6.67 * 10 ** (-11)  # Newton's Gravitational Constant
    theta = 0.5  # threshold value, i.e. theta = s/d, of Barnes-Hut Tree
    width = 20  # biggest width of figure
    eta = 1.4  # constant associated with timestep, more details see: https://arxiv.org/pdf/2401.02849
    Nt = 10  # number of timesteps
    M = 1  # total mass of objects
    R = 2.5  # free parameter which determines the dimensions of the cluster in Plummer model for init condition
    scaleInitPos = 100  # scale initial position ratio
    plotRealTime = True  # plot in real time
    choosePlummerInit = True  # choose Plummer init conditions
    chooseBHTree = True  # choose BHTree to calculate acceleration
    saveRealTimeFig = False  # if save real time png files
    C = 1  # speed of light
    checkMerger = True  # if check binary merger
    maxb = 0.2  # maximum of impact parameter between binary

    # initial conditions
    pos, vel, mass = chooseInitCondition(N, G, M, R, scaleInitPos, choosePlummerInit)

    topRight = np.array([width / 2, width / 2, width / 2])
    root = constructBHTree(None, pos, None, mass, None, vel, None, width, topRight)

    # calculate initial accelerations
    acc = accOption(root, pos, mass, theta, G, softening, chooseBHTree)
    # print(acc)

    new_dt_pre = getDt(acc, eta, dt, 1)

    # 3D
    fig = plt.figure(figsize=(20, 20))  # Increase figure size
    ax1 = fig.add_subplot(111, projection='3d')
    saveFolder = f"N_{N}_dt_{dt}_width_{width}_Nt_{Nt}"
    if plotRealTime:
        if not os.path.exists(saveFolder):
            os.makedirs(saveFolder)

    pos_save = np.zeros((N, 3, Nt + 1))
    pos_save[:, :, 0] = pos

    for i in range(Nt):
        # (1/2) kick
        vel += acc * new_dt_pre / 2.0

        # drift
        pos += vel * new_dt_pre

        root = constructBHTree(None, pos, None, mass, None, vel, None, width, topRight)

        acc = accOption(root, pos, mass, theta, G, softening, chooseBHTree)

        new_dt = getDt(acc, eta, new_dt_pre, 0)

        # (1/2) kick
        vel += acc * new_dt / 2.0

        # update time
        new_dt_pre = new_dt

        pos_save[:, :, i + 1] = pos

        mergeStatsByBrute = []
        mergeStatsByTree = []
        if checkMerger:
            getInfoMerger(mergeStatsByBrute, mergeStatsByTree, root, pos, vel, mass, N, C, G, maxb, theta)

        # plot in real time
        if plotRealTime or (i == Nt - 1):
            plt.sca(ax1)
            plt.cla()
            cmap = cm.get_cmap('viridis', pos_save.shape[0])
            for j in range(pos_save.shape[0]):
                ax1.scatter(
                    pos_save[j, 0, max(i - 30, 0):i + 1],
                    pos_save[j, 1, max(i - 30, 0):i + 1],
                    pos_save[j, 2, max(i - 30, 0):i + 1],
                    color=cmap(j / pos_save.shape[0]),
                    alpha=0.6,
                    s=80
                )
            # ax1.scatter(pos_save[:, 0, max(i - 50, 0):i + 1].flatten(), pos_save[:, 1, max(i - 50, 0):i + 1].flatten(),
            #             pos_save[:, 2, max(i - 50, 0):i + 1].flatten(), color='#276ab3', alpha=0.3, s=40)
            ax1.scatter(pos[:, 0], pos[:, 1], pos[:, 2], color='#ff474c', s=100, alpha=0.5)
            if checkMerger:
                analysisAndDrawMerger(mergeStatsByBrute, mergeStatsByTree, pos, N, width, ax1)

            ax1.set_xlim([-width / 2, width / 2])
            ax1.set_ylim([-width / 2, width / 2])
            ax1.set_zlim([-width / 2, width / 2])
            ax1.tick_params(axis='x', labelsize=20)
            ax1.tick_params(axis='y', labelsize=20)
            ax1.tick_params(axis='z', labelsize=20)
            ax1.set_xlabel('x', fontsize=25, fontweight='bold', labelpad=20)
            ax1.set_ylabel('y', fontsize=25, fontweight='bold', labelpad=20)
            ax1.set_zlabel('z', fontsize=25, fontweight='bold', labelpad=20)
            # Adjust transparency of axis panes (0.0 to 1.0)  # Remove grid from 3D plot
            ax1.xaxis.pane.set_alpha(0.35)
            ax1.yaxis.pane.set_alpha(0.35)
            ax1.zaxis.pane.set_alpha(0.35)
            ax1.grid(False)
            if saveRealTimeFig:
                filename = f'realtime_{i}.png'
                plt.savefig(os.path.join(saveFolder, filename), dpi=240)
            plt.pause(0.001)

    # add labels/legend
    plt.sca(ax1)
    ax1.set_xlabel('x', fontsize=25, fontweight='bold', labelpad=20)
    ax1.set_ylabel('y', fontsize=25, fontweight='bold', labelpad=20)
    ax1.set_zlabel('z', fontsize=25, fontweight='bold', labelpad=20)
    ax1.tick_params(axis='x', labelsize=20)
    ax1.tick_params(axis='y', labelsize=20)
    ax1.tick_params(axis='z', labelsize=20)
    # ax1.grid(False)

    # Save figure
    plt.savefig('nbody_realtime_3d.png', dpi=240)
    plt.show()

    fig2 = plt.figure(figsize=(20, 20))  # Increase figure size
    ax2 = fig2.add_subplot(111, projection='3d')
    cmap = cm.get_cmap('viridis', pos_save.shape[0])
    for i in range(pos_save.shape[0]):
        ax2.scatter(
            pos_save[i, 0, max(Nt - 30, 0):Nt],
            pos_save[i, 1, max(Nt - 30, 0):Nt],
            pos_save[i, 2, max(Nt - 30, 0):Nt],
            color=cmap(i / pos_save.shape[0]),
            alpha=0.6,
            s=80
        )
    ax2.xaxis.pane.set_alpha(0.35)
    ax2.yaxis.pane.set_alpha(0.35)
    ax2.zaxis.pane.set_alpha(0.35)
    ax2.set_xlim([-width / 2, width / 2])
    ax2.set_ylim([-width / 2, width / 2])
    ax2.set_zlim([-width / 2, width / 2])
    ax2.tick_params(axis='x', labelsize=20)
    ax2.tick_params(axis='y', labelsize=20)
    ax2.tick_params(axis='z', labelsize=20)
    ax2.set_xlabel('x', fontsize=25, fontweight='bold', labelpad=20)
    ax2.set_ylabel('y', fontsize=25, fontweight='bold', labelpad=20)
    ax2.set_zlabel('z', fontsize=25, fontweight='bold', labelpad=20)
    ax2.grid(False)
    plt.savefig('nbody_realtime_3d_final.png', dpi=240)
    plt.show()

    return 0


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")
