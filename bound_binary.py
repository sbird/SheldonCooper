# from initial_conditions_3d import Particles_ini
from Force_Nbody import const_G
import numpy as np
import matplotlib.pyplot as plt
import pint

ureg = pint.UnitRegistry()


#generating the boundry size
#This size also depends on the phase of galaxy formation,
#but I tried to choose the aprroximate maximum possible size of a primordial galaxy formaing gas cloud
# boundary_size = 100*ureg.kpc

# #typical rotating speed in a primordial galaxy forming environment
# velocity_typical = 100*(ureg.km/ureg.s)
# velocity_typical = velocity_typical.to(ureg.kpc/ureg.year)

N = 2 #number of particles
M = np.ones ((N, 1)) ##generating the mass of particles,each particle has a mass equal to solar_mass
v = np.zeros((N, 3)) #velocity of each particle in xyz coordinates
xyz = np.zeros ((N, 3)) #location of each particle in xyz coordinates

# #Locations
# #coordinates xyz[:, 0] -> x, xyz[:, 1] -> y, xyz[:, 2] -> z
# # first particle
# xyz[0, 0] = boundary_size.magnitude/2
# xyz[0, 1] = 0
# xyz[0, 2] = 0

# #second particle
# xyz[1, 0] = -boundary_size.magnitude/2
# xyz[1, 1] = 0
# xyz[1, 2] = 0


# # v[:, 0] -> vx, v[:, 1] -> vy, v[:, 2] -> vz
# #first particles
# v[0, 0] = velocity_typical.magnitude #x
# v[0, 1] = velocity_typical.magnitude #y
# v[0, 2] = 0 #z

# #second particle
# v[1, 0] = -velocity_typical.magnitude #x
# v[1, 1] = -velocity_typical.magnitude #y
# v[1, 2] = 0 #z


class Particles:
    def __init__(self, position, velocity, mass, cm):
        """cm is center of mass"""
        self.position = position
        self.velocity = velocity
        self.mass = mass
    def __getitem__(self, index):
        return {
            'position': self.position[index],
            'velocity': self.velocity[index],
            'mass': self.mass[index] if not np.isscalar(self.mass) else self.mass
        }
    

# particles = Particles_ini(position=xyz,velocity=v,mass=M)

m = np.array([1,1])
r = np.array([[0,0,0],[1,0,0]])
v = np.array([[0,1,0],[0,-1,0]])


def center_of_mass(masses, positions):
    """positions is an nx3 array of positions for multiple particles
        masses is a nx1 array of masses for multiple particles"""
    cm = np.dot(masses,positions)/np.sum(masses)
    return cm 

# print(center_of_mass(m,r))


def total_energy(masses, positions, velocities):
    """positions is an nx3 array of positions for n particles
        masses is a nx1 array of masses for n particles
        velocities is an nx3 array of velocities for n particles"""
    del_pos = np.array(positions[0]-positions[1])   # Calculate relative position
    rr = np.sqrt(np.sum(del_pos**2))                # Calculate the distance from the particle
    vv = np.sum(velocities**2, axis=1)              # Calculate the magnitude of velocity for each particle

    const_G = 1

    U = -np.sum(const_G*masses[0]*masses[1]/rr)      # Calculate potential energy
    K = np.sum(0.5*masses*vv)                       # Calculate kinetic energy
    # print("potential energy:", U/const_G)
    # print("kinetic energy:", K)

    E = K + U                                       # Sum total energy
    # print(E)
    return E,rr

print(total_energy(m,r,v))


def angular_momentum(masses,positions,velocities):
    """positions is an nx3 array of positions for n particles
        masses is a nx1 array of masses for n particles
        velocities is an nx3 array of velocities for n particles"""
    r_cross_v = np.cross(positions, velocities)  
    L_arr = np.dot(masses,r_cross_v)
    L = np.sqrt(np.sum(L_arr**2))
    # print(np.shape(L_arr),L_arr)
    return L_arr

# print(angular_momentum(m,r,v))


def calculate_period(masses,positions,velocities):
    a = -1/2 * const_G * masses[0] * masses[1] / total_energy(masses,positions,velocities)      # define a semi-major axis
    T = np.sqrt(4*np.pi**2 * a**3 / (const_G * np.sum(masses)))                                 # define a period
    L = angular_momentum(masses,positions,velocities)
    mu = masses[1]*masses[0]/np.sum(masses)
    e = np.sqrt(1 + 2*total_energy(masses,positions,velocities)[0] * np.sqrt(np.sum(L**2)) / (const_G**2 * np.sum(masses)**2 * mu**3))
    b = e*a
    return T,e,b

# print("Period", calculate_period(m,r,v))






from sc_time_evolution import evolve_position, evolve_velocity
from Force_Nbody import cal_gforce
# from sc_time_evolution import dt, T, step

dt = 0.1  # Time step in years
T = 6060 # Total time in years
step = int(T / dt)  # Number of steps

#define intial condidtions
energy_list = []
angular_momentum_list = []
relative_positions = []

mass = m
position = r
velocity = v

for s in range(step):   # Iterate through each time step

    if s%100==0:
        energy, distance = total_energy(mass,position,velocity)
        energy_list += [energy]
        relative_positions += [distance]
        ang_mom = angular_momentum(mass,position,velocity)
        angular_momentum_list += [ang_mom]

    # Store current velocity, acceleration, and position for each particle at time step s
    acceleration = cal_gforce(position, m)
    # print(acceleration)

    # Evolve velocity and position using the current state
    velocity_temp = evolve_velocity(velocity, acceleration)          # Update velocity
    position = evolve_position(position, velocity)             # Update position based on the current velocity
    # print("pos,vel:",velocity_temp,position)
    velocity = velocity_temp

fig,axes = plt.subplots(1,3,figsize=(15,5))
time=np.array([i for i in range(len(energy_list))])

ax = axes[0]
ax.plot(time,energy_list)

angular_momentum_list = np.array(angular_momentum_list)
ax = axes[1]
ax.plot(time,angular_momentum_list[:,2])

ax = axes[2]
ax.plot(time,relative_positions)

plt.savefig("binary_conserved.png")