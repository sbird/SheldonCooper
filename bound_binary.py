# from initial_conditions_3d import Particles_ini
# from Force_Nbody import const_G
import numpy as np
import matplotlib.pyplot as plt
import pint

const_G = 4.5e-24
ureg = pint.UnitRegistry()
print(const_G)


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

m = np.array([1e6,1e6])
r = np.array([[0,0,0],[1e-5,0,0]])
v = np.array([[0,0,0],[0,1e-6,0]])


def center_of_mass(masses, positions,velocities):
    """positions is an nx3 array of positions for multiple particles
        masses is a nx1 array of masses for multiple particles"""
    cm = np.dot(masses,positions)/np.sum(masses)
    v_cm = np.dot(masses,velocities)/np.sum(masses)
    return cm, v_cm

# print(center_of_mass(m,r))

def reduced_mass(masses):
    mu = np.prod(masses)/np.sum(masses)
    return mu


def total_energy(masses, positions, velocities):
    """positions is an nx3 array of positions for n particles
        masses is a nx1 array of masses for n particles
        velocities is an nx3 array of velocities for n particles"""
    del_pos = np.array(positions[0]-positions[1])                   # Calculate relative position
    rr = np.linalg.norm(del_pos)                                    # Calculate the distance from the particle
    del_vel = np.array(velocities[0]-velocities[1])                 # Calculate the magnitude of velocity for each particle
    vv = np.linalg.norm(del_vel)**2


    U = -const_G*masses[0]*masses[1]/rr      # Calculate potential energy
    K = 0.5*reduced_mass(masses)*vv          # Calculate kinetic energy
    # print("potential energy:", U/const_G)
    # print("kinetic energy:", K)

    E = K + U           # Sum total energy
    # print(E)
    return E,rr

print(total_energy(m,r,v))


def angular_momentum(masses,positions,velocities):
    """positions is an nx3 array of positions for n particles
        masses is a nx1 array of masses for n particles
        velocities is an nx3 array of velocities for n particles"""
    r_cross_v = np.cross(positions[0]-positions[1], velocities[0]-velocities[1])  
    L_arr = reduced_mass(masses)*r_cross_v
    # L = np.sqrt(np.sum(L_arr**2))
    return L_arr

# print(angular_momentum(m,r,v))


def calculate_period(masses,positions,velocities):
     tot_E,rr=total_energy(masses,positions,velocities)
     a = -1/2 * const_G * masses[0] * masses[1] / tot_E      # define a semi-major axis
     T = np.sqrt(4*np.pi**2 * a**3 / (const_G * np.sum(masses)))
     tot_L=np.linalg.norm(angular_momentum(m,r,v))
     elli=np.sqrt(1+2*tot_E*tot_L**2/(reduced_mass(m)*(const_G*m[0]*m[1])**2))

     return T,a*(1+elli),a*(1-elli)                               # define a period
#     L = angular_momentum(masses,positions,velocities)
#     L_mag = np.linalg.norm(L)

#     mu = masses[1]*masses[0]/np.sum(masses)
#     e = np.sqrt(1 + 2*total_energy(masses,positions,velocities)[0] * L_mag / (const_G**2 * np.sum(masses)**2 * mu**3))
#     b = e*a
#     return T,e,b

# print("Period", calculate_period(m,r,v))






from sc_time_evolution import evolve_position, evolve_velocity
from Force_Nbody import cal_gforce
from sc_time_evolution import dt, T, step

#dt = 0.1  # Time step in years
#T = 6060 # Total time in years
#step = int(T / dt)  # Number of steps

#define intial condidtions
energy_list = []
angular_momentum_list = []
relative_positions = []

mass = m
position = r
velocity = v

# T,e,b=calculate_period(m,r,v)
print("period:",calculate_period(m,r,v))
for s in range(step):   # Iterate through each time step
    # dt,step=set_t(v,a,T)
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

fig,axes = plt.subplots(1,3,figsize=(17,5))
time=np.array([i for i in range(len(energy_list))])

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