import numpy as np
from scipy.stats import truncnorm
import pint
from constants import k_B

ureg = pint.UnitRegistry()
ureg.define('solar_mass = 1.98847e30 * kilogram')

#generating the boundry size


def velocity_ini(n, Temperature, mass, k_B):  
    # Velocity dispersion (standard deviation of the velocity components)
    sigma_v = np.sqrt((k_B * Temperature) / mass)

    # Generate Maxwell-Boltzmann velocities for each particle
    # velocity of each particle in xyz coordinates, the result is in the shape of (N, 3)
    velocities = np.random.normal(0, sigma_v, (n, 3))

    return velocities


#Generate random gaussian positions for each particle with units
def position_ini(n, boundary_size, sigma_pos, mu=0):
    a = (-boundary_size / 2 - mu) / sigma_pos #lower truncated bound
    b = (boundary_size / 2 - mu) / sigma_pos #upper truncated bound
    x = truncnorm.rvs(a, b, loc=mu, scale=sigma_pos, size=n)
    y = truncnorm.rvs(a, b, loc=mu, scale=sigma_pos, size=n)
    z = truncnorm.rvs(a, b, loc=mu, scale=sigma_pos, size=n)
    pos = np.column_stack((x, y, z))

    return pos


def position_ini_plum(n, boundary_size):
    velocity_total = []
    positions = []

    # Generate all random numbers used at once
    epsilon = 1e-10
    X1 = np.random.uniform(0, 1, n) + epsilon
    X2, X3, X4, X5 = np.random.uniform(0, 1, (4, n)) + epsilon  # avoid r being negative

    # Calculate r (the distance of particles from center) and ensure it is within the boundary size
    r = np.sqrt(X1**(-2/3) - 1)
    valid_r = (r > 0) & (r < boundary_size)  # check valid r
    while not np.all(valid_r):
        X1[~valid_r] = np.random.uniform(0, 1, np.sum(~valid_r))  # get the ones that do not satisfy the condition
        r = np.sqrt(X1**(-2/3) - 1)  # regenerate r
        valid_r = (r > 0) & (r < boundary_size)
    print(f'r: {r}')
    # Calculate x, y, z positions based on r
    z = np.abs((1 - 2 * X2) * r)
    xy_plane_radius = np.sqrt(r**2 - z**2)
    x = np.abs(xy_plane_radius * np.cos(2 * np.pi * X3))
    y = np.abs(xy_plane_radius * np.sin(2 * np.pi * X3))

    # Ensure x^2 + y^2 + z^2 = r^2
    assert np.allclose(x**2 + y**2 + z**2, r**2), "x^2 + y^2 + z^2 does not equal r^2"

    # Calculate escape velocity
    V_esc = np.sqrt(2) * (1 + r**2)**(-1/4)

    # Calculate q and ensure the condition is met
    valid_q = 0.1 * X5 < X4**2 * (1 - X4**2)**(7/2)
    while not np.all(valid_q):
        X4[~valid_q] = np.random.uniform(0, 1, np.sum(~valid_q)) + epsilon  # get the ones that do not satisfy the condition
        X5[~valid_q] = np.random.uniform(0, 1, np.sum(~valid_q)) + epsilon  # get the ones that do not satisfy the condition
        valid_q = 0.1 * X5 < X4**2 * (1 - X4**2)**(7/2)  # regenerate q

    q = X4
    V = V_esc * q

    positions = np.column_stack((x, y, z))  # each row represents the position of a particle (x, y, z)
    velocity_total = V.tolist()

    return positions, velocity_total



#Now, we make a class to store all of these initial conditions (mass, positions, velocities)
class Particles_ini:
    def __init__(self, n=100, boundary_size=100, mu=0, sigma_pos=50, Temperature=1e20, mass=1):
        #calling the position_ini funtion to generate the location of each particle in xyz coordinates,
        #the result is in the shape of (N, 3)
        #This size also depends on the phase of galaxy formation,
        #but we tried to choose the aprroximate maximum possible size of a primordial galaxy formaing gas cloud
        self.position = position_ini(n=n, boundary_size=boundary_size, mu=mu, sigma_pos=sigma_pos)
        self.velocity = velocity_ini(n=n, Temperature=Temperature, mass=mass, k_B = k_B)
        self.mass = np.ones(n) * mass

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
    n = 3  
    boundary_size = 10

    positions, velocities = position_ini_plum(n, boundary_size)

    print("Positions:")
    print(positions)
    print("\nVelocities:")
    print(velocities)