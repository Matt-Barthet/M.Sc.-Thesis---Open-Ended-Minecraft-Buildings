# Filter's main function which executes the functionality desired.
import numpy as np
from tensorflow.python.keras.utils.np_utils import to_categorical

from Constraints import apply_constraints
from Visualization import voxel_plot

if __name__ == "__main__":

    buildings = []
    for i in range(200):
        voxels = np.zeros((20, 20, 20), dtype=int)
        anchor = [np.random.randint(0, 10), 0, np.random.randint(0, 10)]
        dimensions = [np.random.randint(anchor[i] + 9, 20) for i in range(3)]

        for x in range(anchor[0], dimensions[0]):
            for y in range(anchor[1], dimensions[1]):
                for z in range(anchor[2], dimensions[2]):
                    voxels[x][z][y] = 1

        voxels = apply_constraints(voxels)[1]
        buildings.append(to_categorical(voxels, num_classes=5))

    np.save("Block_Buildings.npy", np.asarray(buildings, dtype=bool))
