import numpy as np

from initial_conditions_3d import Particles_ini
from sc_time_evolution import evolve_position, evolve_velocity, set_t
from plot_3D import visualization
from bound_binary import evolution_loop
from Tree_Force_Nbody_8children import acc_cal

import sys
sys.setrecursionlimit(100000000)


# number_particles = 100
# boundary = 1e-5
# evolution_loop(N=number_particles,m=mass,r=position,v=velocity,boundary_size=boundary,end_time=10)
# particles_ini = Particles_ini(n=number_particles, boundary_size=boundary_size, init_method='plummer', mu=mu, sigma_pos=sigma_pos, Temperature=Temperature, mass=mass, diff_mass=False) 



# Constants
number_particles = 200
boundary_size = 1e-3
init_method = 'plummer'
mu = 0
sigma_pos = 5e-3
Temperature = 1e62
particle_mass = 1e3

delta_t = 500  # Time step in years that will be stored
total_time = 100000  # Total time in years
total_timestep = int(total_time / delta_t)  # Number of steps
current_time = 0  # Current time in years
tracking_timestep = 0  # Time interval to track when the positions will be saved

particles_ini = Particles_ini(n=number_particles, boundary_size=boundary_size, init_method='plummer', mu=mu, sigma_pos=sigma_pos, Temperature=Temperature, mass=particle_mass, diff_mass=False) 
position = particles_ini.position
mass = particles_ini.mass
velocity = particles_ini.velocity  # There is residual momentum in the each direction i.e. the total momentum is not 0
acceleration = np.zeros((number_particles, 3))
position_matrix = np.empty((number_particles, total_timestep, 3))  # Initialize position matrix to store positions at each time step


# Evolve the system using the Tree-Force algorithm
while(current_time < total_time):
    acceleration = acc_cal(position, mass, acceleration, box_size=boundary_size) # Calculate acceleration
    dt = set_t(velocity, acceleration, coeff=1e1)
    dt = min(20, dt)

    # Update velocity and position using half-step method
    velocity_temp = evolve_velocity(velocity, acceleration, dt / 2)  # Half-step velocity
    position = evolve_position(position, velocity, dt)               # Update position
    acceleration = acc_cal(position, mass, acceleration, box_size=boundary_size)    # Recalculate force
    velocity = evolve_velocity(velocity_temp, acceleration, dt / 2)  # Finalize velocity

    # Update current time 
    current_time += dt

    # Store updated positions in the matrix
    if current_time > tracking_timestep * delta_t:
        position_matrix[:, tracking_timestep-1, :] = position  # Store updated positions in the matrix
        tracking_timestep += 1  # Increment the tracking timestep counter

if init_method == 'plummer':
    fname = f'N{number_particles}_logM{np.log10(particle_mass):.1f}_plummer.gif'
elif init_method == 'boltzmann':
    fname= f'N{number_particles}_logM{np.log10(particle_mass):.1f}_logT{np.log10(Temperature):.1f}_logSigma{np.log10(sigma_pos):.1f}.gif'

# Visualize the particle motion in 3D
visualization(position_matrix, lim_bound=(-boundary_size/2, boundary_size/2), savegif=True, fname=fname)  # Call the visualization function to animate the particle motion

# for s in range(step):   # Iterate through each time step
#     # Store current velocity, acceleration, and position for each particle at time step s
#     acceleration = cal_gforce(position, mass)

#     # Evolve velocity and position using the current state
#     velocity_temp = evolve_velocity(velocity, acceleration, dt)          # Update velocity
#     position = evolve_position(position, velocity, dt)             # Update position based on the current velocity
#     velocity = velocity_temp

#     # Store updated positions in the matrix
#     position_matrix[:, s, :] = position 