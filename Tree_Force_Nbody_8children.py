import numpy as np
import time
import sys
#import os

# Add the directory to sys.path
# Change the path!
sys.path.append('/Users/aryanahaghjoo/Documents/GitHub/SheldonCooper')

# Now you can import your file (without the .py extension)
from constants import G as const_G  # Import the gravitational constant

# Define an OctreeNode class to represent nodes in the octree
class OctreeNode:
    def __init__(self, center, size):
        self.center = np.array(center)  # Center of the node
        self.size = size  # Size of the node (length of one side)
        self.mass = 0.0  # Total mass in this node
        self.mass_center = np.zeros(3)  # Center of mass for this node
        self.particle = None  # A single particle, if this node is a leaf
        self.children = [None] * 8  # Eight children nodes

    def is_leaf(self):
        # A node is a leaf if it contains one particle or has no children
        return self.particle is not None or all(child is None for child in self.children)

# Define a Particle class to represent each particle in the simulation
class Particle:
    def __init__(self, mass, position, vel, acc, seq):
        self.mass = mass  # Mass of the particle
        self.position = np.array(position)  # Position in 3D space
        self.vel = np.array(vel)  # Velocity vector
        self.acc = np.array(acc)  # Acceleration vector
        self.seq = seq  # Unique identifier for the particle

    def cal_gforce(self, particles):
        # O(N²) gravitational force calculation
        self.acc = np.zeros(3)  # Reset acceleration
        for other in particles:
            if other.seq != self.seq:  # Avoid self-interaction
                del_pos = other.position - self.position
                rr = np.linalg.norm(del_pos) + 1e-10  # Add small value to avoid division by zero
                self.acc += const_G * other.mass * del_pos / rr**3  # Acceleration due to gravity

# Insert a particle into the octree
def insert_particle(node, particle):
    if node.is_leaf():
        if node.particle is None:
            # If the node is empty, place the particle here
            node.particle = particle
            node.mass = particle.mass
            node.mass_center = particle.position
        else:
            # If a particle already exists, subdivide the node
            existing_particle = node.particle
            node.particle = None  # Clear the leaf status
            subdivide_node(node)  # Create child nodes
            insert_particle(node, existing_particle)  # Reinsert the existing particle
            insert_particle(node, particle)  # Insert the new particle
    else:
        # Update mass and center of mass
        node.mass += particle.mass
        node.mass_center = (node.mass_center * (node.mass - particle.mass) + particle.mass * particle.position) / node.mass
        # Determine the appropriate child and insert the particle
        index = get_octant_index(node, particle.position)
        if node.children[index] is None:
            child_center = get_child_center(node.center, node.size / 2, index)
            node.children[index] = OctreeNode(child_center, node.size / 2)
        insert_particle(node.children[index], particle)

# Subdivide a node into 8 children
def subdivide_node(node):
    size = node.size / 2  # Halve the size for child nodes
    for i in range(8):
        child_center = get_child_center(node.center, size, i)
        node.children[i] = OctreeNode(child_center, size)

# Determine the octant index for a given position
def get_octant_index(node, position):
    index = 0
    if position[0] >= node.center[0]: index += 4
    if position[1] >= node.center[1]: index += 2
    if position[2] >= node.center[2]: index += 1
    return index

# Compute the center of a child node based on the parent's center and index
def get_child_center(center, size, index):
    offset = np.array([size / 2 if index & (1 << i) else -size / 2 for i in range(3)])
    return center + offset

# Calculate gravitational force on a particle using the Barnes-Hut method
def calculate_force(particle, node, num_par, theta=0.5, coe=2e-3):
    if node is None or node.mass == 0 or (node.is_leaf() and node.particle.seq == particle.seq):
        return np.zeros(3)

    displacement = node.mass_center - particle.position
    distance = np.linalg.norm(displacement) + 1e-10  # Avoid division by zero

    length_acc = np.linalg.norm(particle.acc)
    if length_acc != 0:
        theta = np.sqrt(coe * length_acc / (const_G * num_par * particle.mass)) * distance
    if node.is_leaf() or (node.size / distance < theta):
        # Treat the node as a single mass
        return const_G * node.mass * displacement / distance**3
    else:
        # Recursively calculate force from children
        force = np.zeros(3)
        for child in node.children:
            if child is not None:
                force += calculate_force(particle, child, num_par, theta, coe)
        return force

# Print the structure of the octree
def print_tree(node, depth=0):
    if node is None:
        return
    indent = "  " * depth
    if node.is_leaf():
        if node.particle is not None:
            print(f"{indent}Particle at {node.particle.position}, mass={node.mass}")
    else:
        print(f"{indent}Node at {node.center}, size={node.size}, mass={node.mass}")
        for child in node.children:
            print_tree(child, depth + 1)

# Move a particle and update its position in the tree
def move_particle(particle, root):
    old_position = particle.position.copy()
    particle.position += particle.vel + 0.5 * particle.acc  # Simple Euler integration
    if not np.allclose(old_position, particle.position):
        insert_particle(root, particle)

# # Remove empty leaves from the octree
# def clip_tree(node):
#     if node is None:
#         return
#     if node.is_leaf():
#         if node.particle is None:
#             return None  # Remove empty leaf
#     else:
#         for i in range(8):
#             node.children[i] = clip_tree(node.children[i])
#     return node

if __name__ == "__main__":
    num_particles = 1000
    box_size = 100.0
    root = OctreeNode([box_size / 2] * 3, box_size)

    # Generate particles with random positions
    particles = []
    for i in range(num_particles):
        position = np.random.uniform(0, box_size, 3)
        particles.append(Particle(1.0, position, np.zeros(3), np.zeros(3), i))

    # Insert particles into the octree
    for particle in particles:
        insert_particle(root, particle)

    # Compare force calculation times
    print("Calculating forces using Octree...")
    start_time = time.time()
    for particle in particles:
        force = calculate_force(particle, root, num_particles)
    elapsed_time = time.time() - start_time
    print(f"Octree Force Calculation Time = {elapsed_time:.5f} seconds")

    print("Calculating forces using O(N²) method...")
    start_time = time.time()
    for particle in particles:
        particle.cal_gforce(particles)
    elapsed_time = time.time() - start_time
    print(f"O(N²) Force Calculation Time = {elapsed_time:.5f} seconds")

    # Uncomment to print the octree structure
    # print_tree(root)

    # Example of updating particles
    # for particle in particles:
    #     move_particle(particle, root)

    # Uncomment to clip and print the updated tree
    # root = clip_tree(root)
    # print_tree(root)