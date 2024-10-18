
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import cnames
from matplotlib import animation
from scipy.stats import truncnorm



def visualization(position):


    # Set up figure & 3D axis for animation
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1], projection='3d')
    ax.axis('off')

    # choose a different color for each particle
    num_particles = position.shape[0]
    colors = plt.cm.jet(np.linspace(0, 1, num_particles))

    # randomly select 10% of the particles for drawing lines
    random_indices = np.random.choice(num_particles, size=int(0.1 * num_particles), replace=False)
    # set up lines and points
    # we need to pick 10% of the particles' trajectories to have the lines if we have large number of particles in the future
    lines = sum([ax.plot([], [], [], '-', c=colors[i]) for i in random_indices], [])
    
    pts = sum([ax.plot([], [], [], 'o', c=colors[i]) for i in range(num_particles)], [])
    # prepare the axes limits
    ## change the axes limits to boundary size
    lim_bound = (0,100)
    ax.set_xlim(lim_bound)
    ax.set_ylim(lim_bound)
    ax.set_zlim(lim_bound)

    # set point-of-view: specified by (altitude degrees, azimuth degrees)
    ax.view_init(30, 0)

    # initialization function: plot the background of each frame
    def init():
        for line, pt in zip(lines, pts):
            line.set_data([], [])
            line.set_3d_properties([])

            pt.set_data([], [])
            pt.set_3d_properties([])
        return lines + pts

    # animation function.  This will be called sequentially with the frame number
    def animate(i):
        # we'll step 1 time-steps per frame.
        i = i % position.shape[1]

        # Update lines for the 10% of particles
        for line, idx in zip(lines, random_indices):
            x, y, z = position[idx, :i].T
            line.set_data(x, y)
            line.set_3d_properties(z)

        # Update points for all particles
        for pt, xi in zip(pts, position):
            x, y, z = xi[i]
            pt.set_data([x], [y])
            pt.set_3d_properties([z])

        ax.view_init(30, 0.3 * i)
        fig.canvas.draw()
        return lines + pts

    # instantiate the animator.
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                frames=500, interval=30, blit=True)

    plt.show()


def position_ini(N, bound, mu, sigma):
  #bound = bound.to('kpc').magnitude
  a = 0 # lower truncated bound
  b = (bound - mu) / sigma #upper truncated bound
  x = truncnorm.rvs(a,b, loc=mu, scale=sigma, size=N)
  y = truncnorm.rvs(a,b, loc=mu, scale=sigma, size=N)
  z = truncnorm.rvs(a,b, loc=mu, scale=sigma, size=N)
  pos = np.column_stack((x, y, z))
  return pos

if __name__ == '__main__':
# we generate a random initial position for 10 particles
    pos_ini = position_ini(10, 100, 50, 50)

    # we generate the initial position 50 times to make 50 fake time steps
    step = 50
    n = 10
    position_matrix = np.empty((n, step, 3))
    for s in range(step):   # Iterate through each time step
        pos_temp = position_ini(10, 100, 50, 50)
        position_matrix[:, s, :] = pos_temp
    visualization(position_matrix)