import random
import math
import time
from typing import List
import sys
import os

# Add the directory to sys.path
# Change the path here!
sys.path.append('/Users/aryanahaghjoo/Documents/GitHub/SheldonCooper')
from constants import G as const_G

class Node:
    def __init__(self, mass: float, mposition: List[float], lowerbound: List[float], upperbound: List[float]):
        self.mass = mass
        self.mposition = mposition
        self.lowerbound = lowerbound
        self.upperbound = upperbound
        self.depth = 0
        self.index = -1
        self.left_node = None
        self.right_node = None
        self.last_node = None

class Particle:
    def __init__(self, mass: float, position: List[float], vel: List[float], acc: List[float], seq: int):
        self.mass = mass
        self.position = position
        self.vel = vel
        self.acc = acc
        self.seq = seq
        self.parent = None

    def update_time(self, new_time: float):
        pass  # Placeholder for time update logic

    def cal_gforce(self, p_array: List['Particle']):
        self.acc = [0.0, 0.0, 0.0]
        for p in p_array:
            if p.seq != self.seq:
                del_pos = [p.position[j] - self.position[j] for j in range(3)]
                rr = math.sqrt(sum(x**2 for x in del_pos))
                for j in range(3):
                    self.acc[j] += p.mass * del_pos[j] / rr**3
        self.acc = [const_G * x for x in self.acc]

def insert_Node_tree(par: Particle, list_par: List[Particle], Node_tree: Node):
    remainder_dep = Node_tree.depth % 3
    Node_tree.mass += par.mass
    for i in range(3):
        Node_tree.mposition[i] += par.mass * par.position[i]

    if Node_tree.left_node is None and Node_tree.right_node is None:
        if Node_tree.index == -1:
            Node_tree.index = par.seq
            par.parent = Node_tree
        else:
            new_mposition = [0.0, 0.0, 0.0]
            node1 = Node(0, new_mposition[:], Node_tree.lowerbound[:], Node_tree.upperbound[:])
            node2 = Node(0, new_mposition[:], Node_tree.lowerbound[:], Node_tree.upperbound[:])
            cut = (Node_tree.upperbound[remainder_dep] + Node_tree.lowerbound[remainder_dep]) / 2.0
            node1.upperbound[remainder_dep] = cut
            node2.lowerbound[remainder_dep] = cut
            node1.last_node = node2.last_node = Node_tree
            node1.depth = node2.depth = Node_tree.depth + 1
            Node_tree.left_node = node1
            Node_tree.right_node = node2

            insert_Node_tree(list_par[Node_tree.index], list_par, node1 if list_par[Node_tree.index].position[remainder_dep] <= cut else node2)
            insert_Node_tree(par, list_par, node1 if par.position[remainder_dep] <= cut else node2)
            Node_tree.index = -1
    else:
        cut = (Node_tree.upperbound[remainder_dep] + Node_tree.lowerbound[remainder_dep]) / 2.0
        if par.position[remainder_dep] <= cut:
            insert_Node_tree(par, list_par, Node_tree.left_node)
        else:
            insert_Node_tree(par, list_par, Node_tree.right_node)

def cal_force(par: Particle, Node_tree: Node) -> List[float]:
    if Node_tree is None or Node_tree.mass == 0 or Node_tree.index == par.seq:
        return [0.0, 0.0, 0.0]

    del_pos = [(Node_tree.mposition[i] / Node_tree.mass) - par.position[i] for i in range(3)]
    rr = math.sqrt(sum(x**2 for x in del_pos))

    if (math.sqrt(sum((Node_tree.upperbound[i] - Node_tree.lowerbound[i])**2 for i in range(3))) / rr < 1) or Node_tree.index != -1:
        return [const_G * Node_tree.mass * del_pos[i] / rr**3 for i in range(3)]
    else:
        force_left = cal_force(par, Node_tree.left_node)
        force_right = cal_force(par, Node_tree.right_node)
        return [force_left[i] + force_right[i] for i in range(3)]

if __name__ == "__main__":
    num = int(1e4)
    boxsize = 10
    box_lowerbound = [0.0, 0.0, 0.0]
    box_upperbound = [2**boxsize] * 3
    list_par = []
    root = Node(0, [0.0, 0.0, 0.0], box_lowerbound, box_upperbound)

    for i in range(num):
        position = [random.uniform(0, 2**boxsize) for _ in range(3)]
        pp = Particle(1, position, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], i)
        list_par.append(pp)

    for i in range(num):
        insert_Node_tree(list_par[i], list_par, root)

    start_time = time.time()
    for _ in range(num):
        force = cal_force(list_par[0], root)
    elapsed_time = time.time() - start_time
    print(f"Force: {force}")
    print(f"Tree Calculation Time = {elapsed_time} seconds")

    start_time = time.time()
    for _ in range(num):
        list_par[0].cal_gforce(list_par)
    elapsed_time = time.time() - start_time
    print(f"Direct Calculation Time = {elapsed_time} seconds")