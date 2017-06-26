import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
from matplotlib.colors import LogNorm


dx = 15.416408  # pixel separation in x-direction [mm]
dy = 13.351001  # pixel separation in x-direction [mm]
radius = dx / 2 - 1  # pixel radius [mm]

# coordinates (axial hex-grid) for each pixel
pixel_coords = np.array([
    [ 0,  0,  0],
    [ 1,  1,  0],
    [ 2, -1,  0],
    [ 3,  0,  1],
    [ 4,  0, -1],
    [ 5,  1, -1],
    [ 6, -1,  1],
    [ 7, -1, -1],
    [ 8,  1,  1],
    [ 9, -2,  1],
    [10,  2, -1],
    [11, -1,  2],
    [12,  1, -2],
    [13,  2,  0],
    [14, -2,  0],
    [15,  2, -2],
    [16,  0,  2],
    [17,  0, -2],
    [18, -2,  2],
    [19,  1,  2],
    [20, -3,  2],
    [21,  3, -2],
    [22, -1, -2],
    [23,  2,  1],
    [24, -3,  1],
    [25,  3, -1],
    [26, -2, -1],
    [27, -2,  3],
    [28, -1,  3],
    [29,  1, -3],
    [30,  2, -3],
    [31,  3,  0],
    [32, -3,  0],
    [33, -3,  3],
    [34,  0, -3],
    [35,  3, -3],
    [36,  0,  3],
    [37,  2,  2],
    [38, -2, -2],
    [39,  4, -2],
    [40, -4,  2],
    [41, -2,  4],
    [42,  2, -4],
    [43, -4,  1],
    [44, -3,  4],
    [45,  3,  1],
    [46, -3, -1],
    [47,  4, -1],
    [48, -1,  4],
    [49,  1, -4],
    [50,  3, -4],
    [51, -4,  3],
    [52, -1, -3],
    [53,  4, -3],
    [54,  1,  3],
    [55, -4,  0],
    [56,  4,  0],
    [57,  0,  4],
    [58, -4,  4],
    [59,  0, -4],
    [60,  4, -4]
    ])


def vector2matrix(image):
    """
    Converts an image 61-vector to a 9x9 matrix according to the pixel definition in coords.
    """
    p,x,y = pixel_coords.T
    matrix = np.zeros((9,9))
    matrix[x+4, y+4] = image
    return matrix

def display_matrix(image=None):
    """
    Display matrix for given image (shape = 61)
    """
    fig, ax = plt.subplots(1)
    circles = [Circle((x, y), 0.45) for p,x,y in pixel_coords]
    coll1 = PatchCollection(circles, facecolor='none', edgecolor='black')
    ax.add_collection(coll1)
    if image is None:
        [plt.text(x, y, p, ha='center', va='center') for p,x,y in pixel_coords]
    else:
        coll2 = PatchCollection(circles, norm=LogNorm())
        coll2.set_array(np.array(image))
        ax.add_collection(coll2)
        cbar = plt.colorbar(coll2)
        cbar.set_label('number of photons')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_aspect('equal')
    return fig, ax

def display_camera(image=None):
    """ 
    Display camera for given image (shape = 61)
    """
    fig, ax = plt.subplots(1)
    circles = [Circle(((x+y/2.)*dx, y*dy), radius) for p,x,y in pixel_coords]
    coll1 = PatchCollection(circles, facecolor='none', edgecolor='black')
    ax.add_collection(coll1)
    if image is None:
        [plt.text((x+y/2.)*dx, y*dy, p, ha='center', va='center') for p,x,y in pixel_coords]
    else:
        coll2 = PatchCollection(circles, norm=LogNorm())
        coll2.set_array(np.array(image))
        ax.add_collection(coll2)
        cbar = plt.colorbar(coll2)
        cbar.set_label('number of photons')
    ax.set_xlabel('x / mm')
    ax.set_ylabel('y / mm')
    ax.set_xlim(-5*dx, 5*dx)
    ax.set_ylim(-5*dy, 5*dy)
    ax.set_aspect('equal')
    return fig, ax


if __name__ == "__main__":
    # plot camera layout
    fig, ax = display_camera()
    fig.savefig('layout_camera.png')    
    fig, ax = display_matrix()
    fig.savefig('layout_matrix.png')

    # plot a random image
    image = np.random.randint(0, 10, 61) * np.linspace(100, 1, 61)
    fig, ax = display_camera(image)
    fig.savefig('event_camera.png')
    fig, ax = display_matrix(image)
    fig.savefig('event_matrix.png')