# Description: This script calculates the total energy, angular momentum, and period of a binary star system.
# It is used to verify that the total energy and angular momentum are conserved over time.
# The script also calculates the period of the binary star system.
# Writers: Lauren, Hui, Aryana, Negin

import numpy as np
import matplotlib.pyplot as plt
from constants import G as const_G  # Gravitational constant
from plot_3D import visualization   # Importing 3D plotting function for visualization

# Function to choose initial conditions for the binary system
def choose_initial_condition(number_particles=2):
    masses = []
    # Assign random masses to particles
    for i in range(number_particles): 
        masses.append(np.random.randint(1, 9) * 1e6)
    masses = np.sort(np.array(masses))[::-1]  # Sort masses in descending order
    print('Masses = ', masses)

    # Set initial separation and eccentricity
    distance = np.random.randint(1, 9) * 1e-5
    e = np.random.randint(10, 40) * 0.01
    print('e = ', e)

    # Calculate relative velocity based on masses and distance
    v_rel = np.sqrt(const_G * np.sum(masses) * (1 - e) / distance)
    print('v/m_tot = ', v_rel / np.sum(masses))

    # Calculate reduced mass and total energy
    mu = np.prod(masses) / np.sum(masses)
    total_energy = 0.5 * mu * v_rel ** 2 - const_G * np.prod(masses) / distance
    print('E = ', total_energy)

    # Semi-major axis and orbital period calculation
    a = -0.5 * const_G * np.prod(masses) / total_energy
    print('a = ', a)
    period = np.sqrt((4 * np.pi ** 2 * a ** 3) / (const_G * np.sum(masses)))
    print('T = ', period)

    # Set initial positions and velocities
    velocities = np.array([[0, -masses[0] / np.sum(masses) * v_rel, 0],
                           [0, masses[1] / np.sum(masses) * v_rel, 0]])
    positions = np.array([[masses[0] / np.sum(masses) * distance, 0, 0],
                          [-masses[1] / np.sum(masses) * distance, 0, 0]])

    angular_mom = mu * distance * v_rel  # Calculate angular momentum
    return masses, positions, velocities, total_energy, angular_mom

# Initialize system parameters
m, r, v, E0, L0 = choose_initial_condition()

# Function to calculate center of mass (position and velocity)
def center_of_mass(masses, positions, velocities):
    cm = np.dot(masses, positions) / np.sum(masses)      # Calculate COM position
    v_cm = np.dot(masses, velocities) / np.sum(masses)   # Calculate COM velocity
    return cm, v_cm

# Function to calculate reduced mass of the system
def reduced_mass(masses): 
    mu = np.prod(masses) / np.sum(masses)  # Reduced mass formula
    return mu

# Function to calculate total energy of the system
def total_energy(masses, positions, velocities):
    del_pos = np.array(positions[0] - positions[1])     # Relative position vector
    rr = np.linalg.norm(del_pos)                        # Distance between particles
    del_vel = np.array(velocities[0] - velocities[1])   # Relative velocity vector
    vv = np.linalg.norm(del_vel) ** 2

    U = -const_G * masses[0] * masses[1] / rr           # Potential energy
    K = 0.5 * reduced_mass(masses) * vv                 # Kinetic energy
    E = K + U                                           # Total energy
    return E, rr

print("The total Energy of the system is:", total_energy(m, r, v)[0], "M_sun*kpc^2/year^2")

# Function to calculate angular momentum of the system
def angular_momentum(masses, positions, velocities):
    r_cross_v = np.cross(positions[0] - positions[1], velocities[0] - velocities[1])
    L_arr = reduced_mass(masses) * r_cross_v  # Calculate angular momentum vector
    return L_arr

# Function to calculate the period, semi-major axis, and semi-minor axis
def calculate_period(masses, positions, velocities):
     tot_E, rr = total_energy(masses, positions, velocities)       # Calculate total energy
     a = -0.5 * const_G * masses[0] * masses[1] / tot_E            # Semi-major axis
     T = np.sqrt(4 * np.pi ** 2 * a ** 3 / (const_G * np.sum(masses)))  # Orbital period
     tot_L = np.linalg.norm(angular_momentum(m, r, v))             # Angular momentum magnitude
     elli = np.sqrt(1 + 2 * tot_E * tot_L ** 2 / (reduced_mass(m) * (const_G * m[0] * m[1]) ** 2))

     return T, a * (1 + elli), a * (1 - elli)                      # Return period and axes

# Calculate and print system period and distances
T = calculate_period(m, r, v)
print("The period of the system is:", T[0], "years")
print("The maximum distance is:", T[1], "kpc")
print("The minimum distance is:", T[2], "kpc")

# Import necessary functions for time evolution and force calculation
from sc_time_evolution import evolve_position, evolve_velocity, set_t
from Force_Nbody import cal_gforce

# Initialize lists for energy, angular momentum, and position tracking
energy_list = []
angular_momentum_list = []
relative_positions = []

# Set initial values for mass, position, and velocity
mass = m
position = r
velocity = v
position_matrix_binary = np.empty((2, 1, 3))  # Matrix to store position data for visualization

# Simulation parameters
total_time = 0
period_num = 3               # Number of periods to simulate
s = 0                        # Step count
correction_frequency = 1000  # Frequency for resetting center of mass
tracking_frequency = 1000    # Frequency for calculating the parameters

# Main evolution loop
while True:
    # Every correction_frequency steps, reset the COM to prevent drift
    if s % correction_frequency == 0:
        cm_position, cm_velocity = center_of_mass(mass, position, velocity)
        position -= cm_position  # Recenter positions around COM
        # velocity -= cm_velocity  # Set COM velocity to zero

    # Track and store energy, angular momentum, and relative position data every tracking_frequency steps
    if s % tracking_frequency == 0:
        energy, distance = total_energy(mass, position, velocity)
        energy_list.append(energy)
        relative_positions.append(distance)
        ang_mom = angular_momentum(mass, position, velocity)
        angular_momentum_list.append(ang_mom)
        position_matrix_binary = np.append(position_matrix_binary, position[:, np.newaxis, :], axis=1)

    # Calculate forces and update system
    acceleration = cal_gforce(position, m)                    # Calculate gravitational force
    dt = set_t(velocity, acceleration, coeff=1e-4)            # Calculate time step

    # Update velocity and position using half-step method
    velocity_temp = evolve_velocity(velocity, acceleration, dt / 2)  # Half-step velocity
    position = evolve_position(position, velocity, dt)               # Update position
    acceleration = cal_gforce(position, m)                           # Recalculate force
    velocity = evolve_velocity(velocity_temp, acceleration, dt / 2)  # Finalize velocity

    # Update total time and step count
    total_time += dt
    s += 1

    # End simulation if total time exceeds the set number of periods
    if total_time > period_num * T[0]:
        break

# Visualization of position evolution
visualization(position=position_matrix_binary, lim_bound=(-T[1], T[1]))

# Plot and save energy and angular momentum data over time
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
time = np.array([i for i in range(len(energy_list))])

# Plot normalized energy
ax = axes[0]
ax.plot(time, energy_list / E0)
ax.set_xlabel("time (years)")
ax.set_ylabel(r"$\frac{E}{E_{Initial}}$")

# Plot normalized angular momentum
angular_momentum_list = np.array(angular_momentum_list)
ax = axes[1]
ax.plot(time, angular_momentum_list[:, 2] / L0)
ax.set_xlabel("time (years)")
ax.set_ylabel(r"$\frac{L_z}{L_{z,Initial}}$")

# Plot relative positions
ax = axes[2]
ax.plot(time, relative_positions)
ax.set_xlabel("time (years)")
ax.set_ylabel("relative positions (kpc)")

# Save plot and finish
plt.savefig("binary_conserved.png")
print('Done!')