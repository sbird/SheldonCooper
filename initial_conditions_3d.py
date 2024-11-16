import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import truncnorm
import pint

from constants import K_B, G

ureg = pint.UnitRegistry()
ureg.define('solar_mass = 1.98847e30 * kilogram')



def boltzmann_ini_velocities(n, Temperature, mass):  
    # Velocity dispersion (standard deviation of the velocity components)
    sigma_v = np.sqrt((K_B * Temperature) / mass)

    # Generate Maxwell-Boltzmann velocities for each particle
    # velocity of each particle in xyz coordinates, the result is in the shape of (N, 3)
    velocities = np.random.normal(0, sigma_v, (n, 3))

    return velocities



#Generate random gaussian positions for each particle with units
def boltzmann_ini_positions(n, boundary_size, sigma_pos, mu=0):
    a = (-boundary_size / 2 - mu) / sigma_pos #lower truncated bound
    b = (boundary_size / 2 - mu) / sigma_pos #upper truncated bound
    x = truncnorm.rvs(a, b, loc=mu, scale=sigma_pos, size=n)
    y = truncnorm.rvs(a, b, loc=mu, scale=sigma_pos, size=n)
    z = truncnorm.rvs(a, b, loc=mu, scale=sigma_pos, size=n)
    pos = np.column_stack((x, y, z))

    return pos



def plummer_ini_conditions(n, boundary_size, mass):
    # Generate all random numbers used at once
    X1, X2, X3, X4, X5, X6, X7 = np.random.uniform(0, 1, (7, n))
    # choose a = boundary_size / 40 to ensure that almost all (0.996) the particles are inside the boundary initially
    a = boundary_size / 40 

    # Calculate the distance of particles from center, Eq(A2)
    k = (X1 ** (-2 / 3) - 1) ** (-1 / 2) 
    r = k * a

    valid_r = r < boundary_size / 2
    while not np.all(valid_r):
        print(np.sum(~valid_r), "particles are outside the boundary")
        X1[~valid_r] = np.random.uniform(0, 1, np.sum(~valid_r))  # get the ones that do not satisfy the condition
        k = (X1 ** (-2 / 3) - 1) ** (-1 / 2)
        r = k * a
        valid_r = r < boundary_size / 2
    
    r = k * a
    assert np.all(r > 0), "r is not greater than 0"
    assert np.all(r < boundary_size / 2), "r is not less than half of the boundary size"

    # Calculate x, y, z positions based on r, Eq(A3)
    z = (1 - 2 * X2) * r
    x = np.sqrt(r ** 2 - z ** 2) * np.cos(2 * np.pi * X3)
    y = np.sqrt(r ** 2 - z ** 2) * np.sin(2 * np.pi * X3)

    # Ensure x^2 + y^2 + z^2 = r^2
    assert np.allclose(x**2 + y**2 + z**2, r**2), "x^2 + y^2 + z^2 does not equal r^2"

    # Calculate escape velocity
    total_mass = n * mass
    V_esc = np.sqrt(2 * G * total_mass / a) * (1 + k ** 2) ** (-1 / 4) # Eq(A4)

    # Calculate q and ensure the condition is met
    valid_q = 0.1 * X5 < X4 ** 2 * (1 - X4 ** 2) ** (7 / 2) # Eq(A5)
    while not np.all(valid_q):
        X4[~valid_q] = np.random.uniform(0, 1, np.sum(~valid_q))  # get the ones that do not satisfy the condition
        X5[~valid_q] = np.random.uniform(0, 1, np.sum(~valid_q))  # get the ones that do not satisfy the condition
        valid_q = 0.1 * X5 < X4 ** 2 * (1 - X4 ** 2) ** (7 / 2)  # regenerate q

    assert np.all(0.1 * X5 < X4 ** 2 * (1 - X4 ** 2) ** (7 / 2)), "q condition not met"
    V = V_esc * X4

    w = (1 - 2 * X6) * V
    u = np.sqrt(V ** 2 - w ** 2) * np.cos(2 * np.pi * X7)
    v = np.sqrt(V ** 2 - w ** 2) * np.sin(2 * np.pi * X7)

    ini_positions = np.column_stack((x, y, z))  # each row represents the position of a particle (x, y, z)
    ini_velocities = np.column_stack((u, v, w))  # each row represents the velocity of a particle (u, v, w)

    return ini_positions, ini_velocities


#Now, we make a class to store all of these initial conditions (mass, positions, velocities)
class Particles_ini:
    def __init__(self, n=100, boundary_size=100, init_method='plummer', mu=0, sigma_pos=50, Temperature=1e20, mass=1, diff_mass=True):
        #calling the position_ini funtion to generate the location of each particle in xyz coordinates,
        #the result is in the shape of (N, 3)
        #This size also depends on the phase of galaxy formation,
        #but we tried to choose the aprroximate maximum possible size of a primordial galaxy formaing gas cloud
        
        if diff_mass:
            self.mass = np.random.randint(1, 9, size=n) * 1e6
        else:
            self.mass = np.ones(n) * mass

        if init_method == 'plummer':
            self.position, self.velocity = plummer_ini_conditions(n=n, boundary_size=boundary_size, mass=mass)
        elif init_method == 'boltzmann':
            self.position = boltzmann_ini_positions(n=n, boundary_size=boundary_size, mu=mu, sigma_pos=sigma_pos)
            self.velocity = boltzmann_ini_velocities(n=n, Temperature=Temperature, mass=mass)
        else: 
            raise ValueError("Invalid initial conditions method")


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
    
### test code
if __name__ == "__main__":
    # n = 1000000
    # a = 10

    # positions, velocities = plummer_ini_conditions(n, a)
    # distance = np.sqrt(positions[:, 0] ** 2 + positions[:, 1] ** 2 + positions[:, 2] ** 2)
    # bins, bin_edges = np.histogram(distance, bins=100000)
    # r = np.linspace(0, bin_edges[-1], 100000)

    # fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    # f_m = r ** 3 / (r ** 2 + a ** 2) ** (3 / 2) # Eq(2)
    # m_r = bins
    # m_r = np.cumsum(m_r) / np.sum(m_r)
    # ax[0].plot(r, f_m, label='analytical')
    # ax[0].plot(bin_edges[:-1], m_r, label='numerical')
    # ax[0].legend()
    # ax[0].set_xlim(0, 10 * a)
    # ax[0].set_xlabel('r')
    # ax[0].set_ylabel('M(r)')

    # bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    # rho_r = bins / (4 * np.pi * (bin_centers ** 2) * np.diff(bin_edges))
    # f_rho = (1 + r ** 2 / a ** 2) ** (-5 / 2) * 3 / (4 * np.pi) * n / a ** 3 # Eq(1)
    # ax[1].plot(bin_edges[:-1], rho_r, label='numerical')
    # ax[1].plot(r, f_rho, label='analytical')
    # ax[1].legend()
    # ax[1].set_xlim(0, 3 * a)
    # ax[1].set_xlabel('r')
    # ax[1].set_ylabel('rho(r)')
    # plt.savefig('plummer_ini_conditions.png')
    # plt.show()
    n = 10000
    boundary_size = 10
    mass = 1
    positions, velocities = plummer_ini_conditions(n, boundary_size, mass)
    print(positions)
    print(velocities)
    # distances = np.sqrt(positions[:, 0] ** 2 + positions[:, 1] ** 2 + positions[:, 2] ** 2)
    # plt.hist(distances, bins=100)
    # plt.axvline(boundary_size/2)
    # plt.show()