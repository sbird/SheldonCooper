
from matplotlib import pyplot as plt
from matplotlib import animation
import numpy as np
from scipy.stats import truncnorm



def visualization(position, lim_bound=(0, 100)):
    # Set up figure & 3D axis for animation
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1], projection='3d')
    ax.axis('off')

    # choose a different color for each particle
    num_particles = position.shape[0]
    colors = plt.cm.jet(np.linspace(0, 1, num_particles))

    # set up lines and points for each particle
    lines = sum([ax.plot([], [], [], '-', c=colors[i]) for i in range(num_particles)], [])
    pts = sum([ax.plot([], [], [], 'o', c=colors[i]) for i in range(num_particles)], [])

    ## change the axes limits to boundary size
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
        # Determine the range of timesteps to display with a window size
        window_size = 10
        start = max(0, i - window_size + 1)  # Starting index for the window
        end = i + 1  # Ending index for the window

        # Update the lines to show only the most recent timesteps for each particle
        for line, idx in zip(lines, range(num_particles)):
            x, y, z = position[idx, start:end].T  # Extract coordinates for the current window
            line.set_data(x, y)  # Update line data for x and y
            line.set_3d_properties(z)  # Update line data for z

        # Update the points for all particles to reflect their current position
        for pt, xi in zip(pts, position):
            x, y, z = xi[i]  # Get the current coordinates for the particle
            pt.set_data([x], [y])  # Update point data for x and y
            pt.set_3d_properties([z])  # Update point data for z

        ax.view_init(30, 0.3 * i)  # Animate the viewpoint
        fig.canvas.draw()

        return lines + pts

    # instantiate the animator.
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                frames=position.shape[1], interval=300, blit=True)  # set interval to 300ms to see the trajectory clearly

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
    # we generate the initial position 50 times to make 50 fake time steps
    step = 50
    n = 10
    position_matrix = np.empty((n, step, 3))

    for s in range(step):   # Iterate through each time step
        pos_temp = position_ini(N=n, bound=100, mu=50, sigma=50)
        position_matrix[:, s, :] = pos_temp

    visualization(position_matrix)