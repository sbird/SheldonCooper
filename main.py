import numpy as np
from initial_conditions_3d import boundary_cond, Particles_ini
from sc_time_evolution import evolve_position, evolve_velocity
from Force_Nbody import cal_gforce
from plot_3D import visualization
from bound_binary import choose_initial_condition, evolution_loop
from Tree_Force_Nbody_8children import acc_cal
import sys
sys.setrecursionlimit(100000000)


number_particles = 100
boundary = 1e-5

particles_ini = Particles_ini(n=number_particles, boundary_size=boundary, init_method='plummer', mu=0, sigma_pos=50, Temperature=1e20, mass=1000, diff_mass=False) 


# n = 20
# boundary_size = 10 * 1e-3
# init_method = 'plummer'
# mu = 0
# sigma_pos = 5 * 1e-3
# Temperature = 1e62
# mass = 1e8


# # pick 10000K to make everything moving
# # Constants
# dt = 5  # Time step in years
# Period = 10000  # Total time in years
# step = int(Period / dt)  # Number of steps

# print(np.median(velocity[:, 0]))
# print(np.sum(velocity[:, 0]))
position = particles_ini.position
mass = particles_ini.mass
velocity = particles_ini.velocity  # There is residual momentum in the each direction i.e. the total momentum is not 0


evolution_loop(N=number_particles,m=mass,r=position,v=velocity,boundary_size=boundary,end_time=10)


# position_matrix = np.empty((n, step, 3))  # Initialize position matrix to store positions at each time step

# for s in range(step):   # Iterate through each time step
#     # Store current velocity, acceleration, and position for each particle at time step s
#     acceleration = cal_gforce(position, mass)

#     # Evolve velocity and position using the current state
#     velocity_temp = evolve_velocity(velocity, acceleration, dt)          # Update velocity
#     position = evolve_position(position, velocity, dt)             # Update position based on the current velocity
#     velocity = velocity_temp

#     # Store updated positions in the matrix
#     position_matrix[:, s, :] = position 


# fname= f'./gif/N{n}_logT{np.log10(Temperature):.1f}_logsigma{np.log10(sigma_pos):.1f}.gif'
# # print(position_matrix)
# # Visualize the particle motion in 3D
# visualization(position_matrix, lim_bound=(-boundary_size/2, boundary_size/2), savegif=True, fname=fname)  # Call the visualization function to animate the particle motion
