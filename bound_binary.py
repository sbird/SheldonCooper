# Description: This script calculates the total energy, angular momentum, and period of a binary star system.
# It is used to verify that the total energy and angular momentum are conserved over time.
# The script also calculates the period of the binary star system.
#Writers: Lauren, Hui, Aryana, Negin
import numpy as np
import matplotlib.pyplot as plt
from constants import G as const_G
from constants import c as const_c
from plot_3D import visualization

# m = np.array([1e6,1e6])
# r = np.array([[0,0,0],[1e-5,0,0]])
# v = np.array([[0,0,0],[0,1e-6,0]])

def choose_initial_condition(number_perticles=2):
    masses = []
    for i in range(number_perticles): masses.append(np.random.randint(1,9) * 1e6)
    masses       = np.sort(np.array(masses))[::-1]
    distance     = np.random.randint(1,9)   * 1e-5
    e            = np.random.randint(10,40) * 0.01
    print(e)
    mu           = np.prod(masses) / np.sum(masses)
    g_m_m_r      = const_G * np.prod(masses) / distance
    total_energy = -0.5 * (g_m_m_r - np.sqrt(g_m_m_r**2 + mu*const_G**2*(e**2-1)/distance**2))
    v1           = np.sqrt((2*masses[1])/(masses[0]*np.sum(masses)) * (total_energy + (const_G * np.prod(masses)) / distance))
    velocities   = np.array([[0,-(-masses[1]/masses[0])*v1,0],[0,v1,0]])
    positions    = np.array([[masses[1]/np.sum(masses)*distance,0,0],[-masses[0]/np.sum(masses)*distance,0,0]])
    angular_mom  = mu*distance*v1*(1+masses[0]/masses[1])
    print('eccentricity: ', np.sqrt(1 + 2*total_energy*angular_mom**2 / const_G**2*mu**2))
    return masses, positions, velocities
    


m, r, v = choose_initial_condition()

def center_of_mass(masses, positions,velocities):
    """
        masses is a Nx1 array storing the masses of N particles
        positions is a Nx3 array storing the positions of N particles
        velocities is a Nx3 array storing the velocities of N particles
        returns the 3D coordination of center of mass and the velocity of the center of mass

     """
    cm = np.dot(masses,positions)/np.sum(masses) #calculating the center of mass
    v_cm = np.dot(masses,velocities)/np.sum(masses) #calculating the velocity of the center of mass
    return cm, v_cm

def reduced_mass(masses): 
    """
        masses is a Nx1 array storing the masses of N particles
        returns the reduced mass of the system
        This will be used to calculate the total energy in the coordinates of the center of mass coordinates
    """
    mu = np.prod(masses)/np.sum(masses)
    return mu


def total_energy(masses, positions, velocities):
    """
        masses is a Nx1 array storing the masses of N particles
        positions is a Nx3 array storing the positions of N particles
        velocities is a Nx3 array storing the velocities of N particles
        returns the total energy of the system, and the relative distance between the two particles
    """
    del_pos = np.array(positions[0]-positions[1])          # Calculate relative position
    rr = np.linalg.norm(del_pos)                           # Calculate the distance from the particle
    del_vel = np.array(velocities[0]-velocities[1])        # Calculate the magnitude of velocity for each particle
    vv = np.linalg.norm(del_vel)**2


    U = -const_G*masses[0]*masses[1]/rr      # Calculate potential energy
    K = 0.5*reduced_mass(masses)*vv          # Calculate kinetic energy
    E = K + U                                # Sum total energy
    return E,rr

print("The total Energy of the system is:", total_energy(m,r,v)[0], "M_sun*kpc^2/year^2")


def angular_momentum(masses,positions,velocities):
    """
        masses is a Nx1 array storing the masses of N particles
        positions is a Nx3 array storing the positions of N particles
        velocities is a Nx3 array storing the velocities of N particles
        returns the angular momentum of the system with respect to the center of mass
    """
    r_cross_v = np.cross(positions[0]-positions[1], velocities[0]-velocities[1])  
    L_arr = reduced_mass(masses)*r_cross_v
    # L = np.sqrt(np.sum(L_arr**2))
    return L_arr

def calculate_period(masses,positions,velocities):
     """
        masses is a Nx1 array storing the masses of N particles
        positions is a Nx3 array storing the positions of N particles
        velocities is a Nx3 array storing the velocities of N particles
        returns the period of the binary star system, and the semi-major and semi-minor axis of the binary star system
     """
     tot_E,rr=total_energy(masses,positions,velocities)      # calculate the total energy of the system
     a = -1/2 * const_G * masses[0] * masses[1] / tot_E      # define a semi-major axis
     T = np.sqrt(4*np.pi**2 * a**3 / (const_G * np.sum(masses))) #calculate the period
     tot_L=np.linalg.norm(angular_momentum(m,r,v))           # calculate the angular momentum
     elli=np.sqrt(1+2*tot_E*tot_L**2/(reduced_mass(m)*(const_G*m[0]*m[1])**2))

     return T, a*(1+elli), a*(1-elli)                             # return the period, semi-major and semi-minor axis

print("The priode of the system is period:",calculate_period(m,r,v)[0], "years")
print("The semi-major axis of the system is:",calculate_period(m,r,v)[1], "kpc")
print("The semi-minor axis of the system is:",calculate_period(m,r,v)[2], "kpc")


from sc_time_evolution import evolve_position, evolve_velocity
from Force_Nbody import cal_gforce
from sc_time_evolution import dt, T, step

energy_list = []
angular_momentum_list = []
relative_positions = []

mass = m
position = r
velocity = v

position_matrix_binary = np.empty((2, step, 3))
n = 0
#Evolution of binary system
for s in range(step):   # Iterate through each time step
    if s%100==0:
        energy, distance = total_energy(mass,position,velocity)
        energy_list += [energy]
        relative_positions += [distance]
        ang_mom = angular_momentum(mass,position,velocity)
        angular_momentum_list += [ang_mom]
        position_matrix_binary[:, n, :] = position
        n = n+1           
    # Store current velocity, acceleration, and position for each particle at time step s
    acceleration = cal_gforce(position, m)                      # Calculate acceleration

    # Evolve velocity and position using the current state
    velocity_temp = evolve_velocity(velocity, acceleration)          # Update velocity
    position = evolve_position(position, velocity)             # Update position based on the current velocity
    velocity = velocity_temp

# visualization(position=position_matrix_binary, lim_bound=(0,3e-5))

fig,axes = plt.subplots(1, 3, figsize=(17,5))
time = np.array([i for i in range(len(energy_list))])

ax = axes[0]
ax.plot(time,energy_list)
ax.set_xlabel("time (years)")
ax.set_ylabel(r"total energy $[M_{\odot}*\frac{kpc}{year}^2]$")

angular_momentum_list = np.array(angular_momentum_list)
ax = axes[1]
ax.plot(time,angular_momentum_list[:,2])
ax.set_xlabel("time (years)")
ax.set_ylabel(r"angular momentum $M_{\odot}\frac{kpc^2}{year}$")

ax = axes[2]
ax.plot(time,relative_positions)
ax.set_xlabel("time (years)")
ax.set_ylabel("relative positions (kpc)")

plt.savefig("binary_conserved.png")
print('Done!')