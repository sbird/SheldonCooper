

# Description: This script calculates the total energy, angular momentum, and period of a binary star system.
# It is used to verify that the total energy and angular momentum are conserved over time.
# The script also calculates the period of the binary star system.
# Writers: Lauren, Hui, Aryana, Negin, Fangyi
import numpy as np
import matplotlib.pyplot as plt
from constants import G as const_G  # Gravitational constant
from plot_3D import visualization   # Importing 3D plotting function for visualization

N = 5

def choose_initial_condition(number_particles=5):
    # Generate random masses for each particle
    masses = np.random.randint(1, 9, size=number_particles) * 1e6
    print('Masses = ', masses)

    # Generate initial random positions within specified range
    positions = 11e-6 + (99e-6 - 11e-6) * np.random.rand(number_particles, 3)

    # Generate initial random velocities within specified range
    velocities = 1e-7 + (9e-7 - 1e-7) * np.random.rand(number_particles, 3)

    # Calculate the initial center of mass position and velocity
    total_mass = np.sum(masses)
    com_position = np.sum(masses[:, None] * positions, axis=0) / total_mass
    com_velocity = np.sum(masses[:, None] * velocities, axis=0) / total_mass

    # Adjust positions and velocities to set center of mass position and velocity to zero
    positions  -= com_position
    velocities -= com_velocity

    # Recalculate the center of mass position and velocity to verify they are now zero
    final_com_position = np.sum(masses[:, None] * positions, axis=0) / total_mass
    final_com_velocity = np.sum(masses[:, None] * velocities, axis=0) / total_mass

    print("Final Center of Mass Position (r_com):", final_com_position)
    print("Final Center of Mass Velocity (v_com):", final_com_velocity)

    # Calculate reduced mass
    mu = np.prod(masses) / np.sum(masses)

    # Calculate total energy
    kinetic_energy = 0.5 * np.sum(masses * np.sum(velocities**2, axis=1))
    potential_energy = 0
    for i in range(number_particles):
        for j in range(i + 1, number_particles):
            r_ij = np.linalg.norm(positions[i] - positions[j])
            potential_energy += -const_G * masses[i] * masses[j] / r_ij
    total_energy = kinetic_energy + potential_energy
    print('Total Energy = ', total_energy)

    return masses, positions, velocities, total_energy

# Initialize system parameters
m, r, v, E0 = choose_initial_condition(N)

# # Function to calculate center of mass (position and velocity)
# def center_of_mass(masses, positions, velocities):
#     cm = np.dot(masses, positions) / np.sum(masses)      # Calculate COM position
#     v_cm = np.dot(masses, velocities) / np.sum(masses)   # Calculate COM velocity
#     return cm, v_cm

# # Function to calculate reduced mass of the system
# def reduced_mass(masses): 
#     mu = np.prod(masses) / np.sum(masses)  # Reduced mass formula
#     return mu

# # Function to calculate total energy of the system
# def total_energy(masses, positions, velocities):
#     del_pos = np.array(positions[0] - positions[1])     # Relative position vector
#     rr = np.linalg.norm(del_pos)                        # Distance between particles
#     del_vel = np.array(velocities[0] - velocities[1])   # Relative velocity vector
#     vv = np.linalg.norm(del_vel) ** 2

#     U = -const_G * masses[0] * masses[1] / rr           # Potential energy
#     K = 0.5 * reduced_mass(masses) * vv                 # Kinetic energy
#     E = K + U                                           # Total energy
#     return E, rr

# print("The total Energy of the system is:", total_energy(m, r, v)[0], "M_sun*kpc^2/year^2")

# # Function to calculate angular momentum of the system
# def angular_momentum(masses, positions, velocities):
#     r_cross_v = np.cross(positions[0] - positions[1], velocities[0] - velocities[1])
#     L_arr = reduced_mass(masses) * r_cross_v  # Calculate angular momentum vector
#     return L_arr

# # Function to calculate the period, semi-major axis, and semi-minor axis
# def calculate_period(masses, positions, velocities):
#      tot_E, rr = total_energy(masses, positions, velocities)       # Calculate total energy
#      a = -0.5 * const_G * masses[0] * masses[1] / tot_E            # Semi-major axis
#      T = np.sqrt(4 * np.pi ** 2 * a ** 3 / (const_G * np.sum(masses)))  # Orbital period
#      tot_L = np.linalg.norm(angular_momentum(m, r, v))             # Angular momentum magnitude
#      elli = np.sqrt(1 + 2 * tot_E * tot_L ** 2 / (reduced_mass(m) * (const_G * m[0] * m[1]) ** 2))

#      return T, a * (1 + elli), a * (1 - elli)                      # Return period and axes

# # Calculate and print system period and distances
# T = calculate_period(m, r, v)
# print("The period of the system is:", T[0], "years")
# print("The maximum distance is:", T[1], "kpc")
# print("The minimum distance is:", T[2], "kpc")

# Import necessary functions for time evolution and force calculation
from sc_time_evolution import evolve_position, evolve_velocity, set_t
from Force_Nbody import cal_gforce

# # Initialize lists for energy, angular momentum, and position tracking
# energy_list = []
# angular_momentum_list = []
# relative_positions = []

# Set initial values for mass, position, and velocity
mass = m
position = r
velocity = v
position_matrix_binary = np.empty((N, 1, 3))  # Matrix to store position data for visualization

# Simulation parameters 
total_time = 0             # count the total evolving time
s = 0                      # Step count 
tracking_frequency = 50  # Frequency for calculating the parameters
# Main evolution loop
while(1):
    if s%tracking_frequency==0:
        # energy, distance = total_energy(mass,position,velocity)
        # energy_list += [energy]
        # relative_positions += [distance]
        # ang_mom = angular_momentum(mass,position,velocity)
        # angular_momentum_list += [ang_mom]
        position_matrix_binary = np.append(position_matrix_binary,position[:, np.newaxis, :],axis=1)
    # Store current velocity, acceleration, and position for each particle at time step s
    acceleration = cal_gforce(position, m)                           # Calculate acceleration
    dt = set_t(v, acceleration, coeff=1e-3)

    # Update velocity and position using half-step method
    velocity_temp = evolve_velocity(velocity, acceleration, dt / 2)  # Half-step velocity
    position = evolve_position(position, velocity, dt)               # Update position
    acceleration = cal_gforce(position, m)                           # Recalculate force
    velocity = evolve_velocity(velocity_temp, acceleration, dt / 2)  # Finalize velocity

    # Update total time and step count
    total_time += dt
    s += 1

    # End simulation if total time exceeds the set number of periods
    if total_time > 500:
        break

print(dt)

# Visualization of position evolution
visualization(position=position_matrix_binary, lim_bound=(-5e-5,5e-5))

# # Plot and save energy and angular momentum data over time
# fig, axes = plt.subplots(1, 3, figsize=(17, 5))
# fig.subplots_adjust(wspace=0.3, hspace=0.3)
# time = np.array([i for i in range(len(energy_list))])

# # Plot normalized energy
# ax = axes[0]
# ax.plot(time, energy_list / E0)
# ax.set_xlabel("time (years)")
# ax.set_ylabel(r"$\frac{E}{E_{Initial}}$")

# # Plot normalized angular momentum
# angular_momentum_list = np.array(angular_momentum_list)
# ax = axes[1]
# ax.plot(time, angular_momentum_list[:, 2] / L0)
# ax.set_xlabel("time (years)")
# ax.set_ylabel(r"$\frac{L_z}{L_{z,Initial}}$")

# # Plot relative positions
# ax = axes[2]
# ax.plot(time, relative_positions)
# ax.set_xlabel("time (years)")
# ax.set_ylabel("relative positions (kpc)")

# # Save plot and finish
# plt.savefig("binary_conserved.png")
print('Done!')