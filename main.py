from initial_conditions_3d import boundary_cond, Particles_ini
from sc_time_evolution import evolve_position, evolve_velocity
from Force_Nbody import cal_gforce

# pick 10000K to make everything moving
particles_ini = Particles_ini(n=10, mu=5, sigma_pos=5, T=10000)

# Constants
dt = 1  # Time step in years
T = 100  # Total time in years
step = int(T / dt)  # Number of steps

velocity = particles_ini.velocity
position = particles_ini.position
mass = particles_ini.mass

for s in range(step):   # Iterate through each time step
    # Store current velocity, acceleration, and position for each particle at time step s
    acceleration = cal_gforce(position, mass)

    # Evolve velocity and position using the current state
    velocity_temp = evolve_velocity(velocity, acceleration)          # Update velocity
    position = evolve_position(position, velocity)             # Update position based on the current velocity
    velocity = velocity_temp


