import numpy as np
import matplotlib.image as image
import matplotlib.pyplot as plt

lattice_dimensions = [20, 20, 20]
floor = image.imread('Minecraft Textures/cobblestone.png')
wall = image.imread('Minecraft Textures/oak_planks.png')
roof = image.imread('Minecraft Textures/oak_log.png')
roof_top = image.imread('Minecraft Textures/oak_log_top.png')

xp, yp, __ = roof_top.shape
x_range = np.arange(0, xp, 1)
y_range = np.arange(0, yp, 1)
Y, X = np.meshgrid(y_range, x_range)


def place_roof(axes, x, y, z):
    x *= 16
    y *= 16
    z *= 16
    strides = 2
    axes.plot_surface(X + x, Y + y, X - X + yp - 1 + z, facecolors=roof_top,
                      rstride=strides, cstride=strides,
                      antialiased=True, shade=False)
    axes.plot_surface(X + x, X - X + y, Y + z, facecolors=roof,
                      rstride=strides, cstride=strides,
                      antialiased=True, shade=False)
    axes.plot_surface(X - X + xp - 1 + x, X + y, Y + z, facecolors=roof,
                      rstride=strides, cstride=strides,
                      antialiased=True, shade=False)


def place_floor(axes, x, y, z):
    x *= 16
    y *= 16
    z *= 16
    strides = 2
    axes.plot_surface(X + x, Y + y, X - X + yp - 1 + z, facecolors=floor,
                      rstride=strides, cstride=strides,
                      antialiased=True, shade=False)
    axes.plot_surface(X + x, X - X + y, Y + z, facecolors=floor,
                      rstride=strides, cstride=strides,
                      antialiased=True, shade=False)
    axes.plot_surface(X - X + xp - 1 + x, X + y, Y + z, facecolors=floor,
                      rstride=strides, cstride=strides,
                      antialiased=True, shade=False)


def place_walls(axes, x, y, z):
    x *= 16
    y *= 16
    z *= 16
    strides = 2
    axes.plot_surface(X + x, Y + y, X - X + yp - 1 + z, facecolors=wall,
                      rstride=strides, cstride=strides,
                      antialiased=True, shade=False)
    axes.plot_surface(X + x, X - X + y, Y + z, facecolors=wall,
                      rstride=strides, cstride=strides,
                      antialiased=True, shade=False)
    axes.plot_surface(X - X + xp - 1 + x, X + y, Y + z, facecolors=wall,
                      rstride=strides, cstride=strides,
                      antialiased=True, shade=False)


def new_voxel_plot(fig, ax, lattice):
    ax.set_axis_off()
    ax.set_xlim([0, lattice_dimensions[0] * 16])
    ax.set_ylim([0, lattice_dimensions[0] * 16])
    ax.set_zlim([0, lattice_dimensions[0] * 16])
    counter = 0
    for x in range(10):
        for y in range(10):
            for z in range(10):
                counter += 1
                print(counter)
                if lattice[x][y][z] == 2:
                    place_walls(ax, x, y, z)
                if lattice[x][y][z] == 3:
                    place_floor(ax, x, y, z)
                if lattice[x][y][z] == 4:
                    place_roof(ax, x, y, z)

