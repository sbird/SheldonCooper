import numpy as np
from scipy.stats import truncnorm
import pint
from constants import k_B

ureg = pint.UnitRegistry()
ureg.define('solar_mass = 1.98847e30 * kilogram')

#generating the boundry size


def velocity_ini(N, T, m, k_B):  

    # Velocity dispersion (standard deviation of the velocity components)
    sigma_v = np.sqrt((k_B * T)/m)

    # Generate Maxwell-Boltzmann velocities for each particle
    # velocity of each particle in xyz coordinates, the result is in the shape of (N, 3)
    velocities = np.random.normal(0, sigma_v, (N, 3))

    return velocities


#Generate random gaussian positions for each particle with units
def position_ini(N, bound, sigma, mu=0):
    a = (-bound - mu) / sigma #lower truncated bound
    b = (bound - mu) / sigma #upper truncated bound
    x = truncnorm.rvs(a,b, loc=mu, scale=sigma, size=N)
    y = truncnorm.rvs(a,b, loc=mu, scale=sigma, size=N)
    z = truncnorm.rvs(a,b, loc=mu, scale=sigma, size=N)
    pos = np.column_stack((x, y, z))

    return pos


#Now, we make a class to store all of these initial conditions (mass, positions, velocities)
class Particles_ini:
    def __init__(self, n=100, boundary_size=100, mu=50, sigma_pos=50, T=1e20):
        #calling the position_ini funtion to generate the location of each particle in xyz coordinates,
        #the result is in the shape of (N, 3)
        #This size also depends on the phase of galaxy formation,
        #but we tried to choose the aprroximate maximum possible size of a primordial galaxy formaing gas cloud
        self.position = position_ini(N=n, bound=boundary_size, mu=mu, sigma=sigma_pos)
        self.velocity = velocity_ini(N=n, T=T, m=1, k_B = k_B)
        self.mass = np.ones(n)
        
        
    def __getitem__(self, index):
        return {
            'position': self.position[index],
            'velocity': self.velocity[index],
            'mass': self.mass[index] if not np.isscalar(self.mass) else self.mass
        }


#Now we make a function to keep every particle inside the defined boundary
#This needs to be run everytime positions are updated
def boundary_cond(particles):
    #reverse if out-of-bound
    particles_in = [] #inside the boundary
    boundary_size = 100
    for particle in particles:
        if np.all(np.abs(particle.position) < boundary_size / 2):
            particles_in.append(particle)
        else:
            particle.velocity *= -1

    particles = particles_in