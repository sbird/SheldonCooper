import numpy as np
import matplotlib.pyplot as plt
from initial_conditions_3d import boundary_cond, Particles_ini
from sc_time_evolution import evolve_position, evolve_velocity
from Force_Nbody import cal_gforce
from plot_3D import visualization

n = 20
boundary_size = 10 * 1e-5
mu = 0
sigma_pos = 5 * 1e-5
Temperature = 1e67


# pick 10000K to make everything moving
particles_ini = Particles_ini(n=n, boundary_size=boundary_size, mu=mu, sigma_pos=sigma_pos, T=Temperature) 
# Constants
dt = 10000  # Time step in years
Period = 10000000  # Total time in years
step = int(Period / dt)  # Number of steps

velocity = particles_ini.velocity
print(np.sum(velocity[:,1]))
position = particles_ini.position/10
mass = particles_ini.mass*10000

position_matrix = np.empty((n, step, 3))  # Initialize position matrix to store positions at each time step

for s in range(step):   # Iterate through each time step
    # Store current velocity, acceleration, and position for each particle at time step s
    acceleration = cal_gforce(position, mass)

    # Evolve velocity and position using the current state
    velocity_temp = evolve_velocity(velocity, acceleration)          # Update velocity
    position = evolve_position(position, velocity)             # Update position based on the current velocity
    velocity = velocity_temp
    position_matrix[:, s, :] = position  # Store updated positions in the matrix

fname= f'N{n}_logT{np.log10(Temperature):.1f}_logsigma{np.log10(sigma_pos):.1f}.gif'
# Visualize the particle motion in 3D
visualization(position_matrix, lim_bound=(-boundary_size, boundary_size), savegif=False, fname=fname)  # Call the visualization function to animate the particle motion
