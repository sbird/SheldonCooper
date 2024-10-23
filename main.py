import numpy as np

from initial_conditions_3d import boundary_cond, Particles_ini
from sc_time_evolution import evolve_position, evolve_velocity
from Force_Nbody import cal_gforce
from plot_3D import visualization

n = 10
mu = 5
sigma_pos = 5
Temperature = 10000


# pick 10000K to make everything moving
particles_ini = Particles_ini(n=n, mu=mu, sigma_pos=sigma_pos, T=Temperature) 
# Constants
dt = 100  # Time step in years
T = 1000000  # Total time in years
step = int(T / dt)  # Number of steps

velocity = particles_ini.velocity*0.01
position = particles_ini.position
mass = 1000*np.ones(n)


position_matrix = np.empty((n, step, 3))  # Initialize position matrix to store positions at each time step

for s in range(step):   # Iterate through each time step
    # Store current velocity, acceleration, and position for each particle at time step s
    acceleration = cal_gforce(position, mass)

    # Evolve velocity and position using the current state
    velocity_temp = evolve_velocity(velocity, acceleration)          # Update velocity
    position = evolve_position(position, velocity)             # Update position based on the current velocity
    velocity = velocity_temp
    position_matrix[:, s, :] = position  # Store updated positions in the matrix


# Visualize the particle motion in 3D
visualization(position_matrix, lim_bound=(0,20))  # Call the visualization function to animate the particle motion
