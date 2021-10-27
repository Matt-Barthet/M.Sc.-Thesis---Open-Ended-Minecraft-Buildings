import numpy as np
from scipy.ndimage import center_of_mass
from scipy.spatial import distance
from Delenox_Config import lattice_dimensions, value_range
from Visualization import voxel_plot

materials = {'External_Space': 0, 'Interior_Space': 1, 'Wall': 2, 'Floor': 3, 'Roof': 4}

door_frames_ew = [
    np.array([[[3, 2, 2, 2], [3, 2, 2, 2], [3, 2, 2, 2]], [[3, 1, 1, 1], [3, 1, 1, 1], [3, 1, 1, 1]]]),
    np.array([[[3, 1, 1, 1], [3, 1, 1, 1], [3, 1, 1, 1]], [[3, 2, 2, 2], [3, 2, 2, 2], [3, 2, 2, 2]]]),
]

door_frames_ns = [
    np.array([[[3, 2, 2, 2], [3, 1, 1, 1]], [[3, 2, 2, 2], [3, 1, 1, 1]], [[3, 2, 2, 2], [3, 1, 1, 1]]]),
    np.array([[[3, 1, 1, 1], [3, 2, 2, 2]], [[3, 1, 1, 1], [3, 2, 2, 2]], [[3, 1, 1, 1], [3, 2, 2, 2]]])
]


if __name__ == "__main__":
    voxel_plot(door_frames_ns[0], "")
    voxel_plot(door_frames_ew[1], "")


class InfeasibleError(Exception):
    pass


class InfeasibleRoof(Exception):
    pass


class InfeasibleEntrance(Exception):
    pass


class InfeasibleVoxelCount(Exception):
    pass


class InfeasibleBoundingBox(Exception):
    pass


class InfeasibleInteriorVolume(Exception):
    pass


class InfeasibleLateralStability(Exception):
    pass


def apply_constraints(lattice):
    """
    Take the lattice generated by a CPPN-NEAT genome and apply the set of constraints implemented in
    the pipeline.
    :param lattice: Input lattice generated by CPPN-NEAT genome.
    :return: Feasibility, repaired lattice.
    """
    try:
        lattice = iterative_flood(lattice)
        lattice = identify_materials(lattice)
        lattice = assess_quality(lattice)
        return True, lattice
    except InfeasibleError:
        return False, lattice


def assess_quality(lattice):
    """
    Assess the quality of the given lattice, determining its feasibility and returning a list of metrics.
    :param lattice: Input lattice.
    :return list of metrics:
    """

    """
    (horizontal_footprint, depth_footprint, vertical_footprint, horizontal_middle, depth_middle, vertical_middle) = footprint_ratios(lattice, horizontal_bounds, vertical_bounds, depth_bounds)
    roof_count = 0
    walls = 0
    floor_count = 0
    for (x, y, z) in value_range:
        if lattice[x][y][z] == 1:
            interior_count += 1
        elif lattice[x][y][z] == 2:
            walls += 1
        elif lattice[x][y][z] == 4:
            roof_count += 1
        elif lattice[x][y][z] == 3:
            floor_count += 1
    """
    interior_count = 0
    total_count = 0

    # lattice = fill_tiny_gaps(lattice)

    try:

        for (x, y, z) in value_range:
            if lattice[x][y][z] == 0:
                continue
            total_count += 1
            if lattice[x][y][z] == 1:
                interior_count += 1

        if total_count == 0:
            raise InfeasibleVoxelCount

        # if interior_count / total_count < 0.3:
            # raise InfeasibleInteriorVolume

        lattice, entrance = place_entrance(lattice)
        if not entrance:
            raise InfeasibleEntrance

        """horizontal_bounds, depth_bounds, vertical_bounds = bounding_box(lattice)
        width = (horizontal_bounds[1] - horizontal_bounds[0])
        height = vertical_bounds[1]
        depth = (depth_bounds[1] - depth_bounds[0])
        if width < 10 or height < 10 or depth < 10:
            raise InfeasibleBoundingBox"""

        # iterative_flood_interior(lattice)

        return lattice

        """
        lattice_stability, floor_stability = stability(lattice)
        if floor_stability > 6:
            raise InfeasibleLateralStability
        """

    except InfeasibleVoxelCount:
        raise InfeasibleError
    except InfeasibleRoof:
        raise InfeasibleError
    except InfeasibleEntrance:
        raise InfeasibleError
    except InfeasibleBoundingBox:
        raise InfeasibleError
    except InfeasibleInteriorVolume:
        raise InfeasibleError
    except InfeasibleLateralStability:
        raise InfeasibleError


def place_entrance(lattice):
    """
    Checks whether the given lattice contains a possible entrance at ground level,
    by comparing it to existing door frames and seeing if at least one of the
    templates fit somewhere in the space.
    :param lattice: Lattice being checked.
    :return: lattice with entrance added, boolean result of check.
    """
    for x in range(20):
        for y in range(20):
            if lattice[x][y][0] == 3:
                for frame in door_frames_ns:
                    if np.array_equal(frame, lattice[x:x + 3, y:y + 2, 0:4]):
                        lattice[x + 1][y][1] = 1
                        lattice[x + 1][y][2] = 1
                        lattice[x + 1][y + 1][1] = 1
                        lattice[x + 1][y + 1][2] = 1
                        return lattice, True
                for frame in door_frames_ew:
                    if np.array_equal(frame, lattice[x:x + 2, y:y + 3, 0:4]):
                        lattice[x][y + 1][1] = 1
                        lattice[x][y + 1][2] = 1
                        lattice[x + 1][y + 1][1] = 1
                        lattice[x + 1][y + 1][2] = 1
                        return lattice, True
    return lattice, False


def bounding_box(lattice):
    """
    :param lattice:
    returns:
    """
    left_bound = 20
    right_bound = 0
    near_bound = 0
    far_bound = 20
    bottom_bound = 0
    top_bound = 0
    for (x, y, z) in value_range:
        if lattice[x][y][z] > 1:
            if y > far_bound:
                far_bound = y
            elif y < near_bound:
                near_bound = y
            if x > right_bound:
                right_bound = x
            elif x < left_bound:
                left_bound = x
            if z > top_bound:
                top_bound = z
    return (left_bound, right_bound), (near_bound, far_bound), (bottom_bound, top_bound)


def footprint_ratios(lattice, horizontal_bounds, vertical_bounds, depth_bounds):
    top_half = 1
    bottom_half = 1
    left_half = 1
    right_half = 1
    near_half = 1
    far_half = 1
    middle_x = 1
    outside_x = 1
    middle_y = 1
    outside_y = 1
    middle_z = 1
    outside_z = 1

    width = horizontal_bounds[1] - horizontal_bounds[0]
    depth = depth_bounds[1] - depth_bounds[0]
    height = vertical_bounds[1]

    for x in range(horizontal_bounds[0], horizontal_bounds[1] + 1):
        for y in range(depth_bounds[0], depth_bounds[1] + 1):
            for z in range(vertical_bounds[0], vertical_bounds[1] + 1):
                if lattice[x][y][z] > 1:

                    if horizontal_bounds[0] + width / 4 < x < horizontal_bounds[1] - width / 4:
                        middle_x += 1
                    else:
                        outside_x += 1
                    if depth_bounds[0] + depth / 4 < y < depth_bounds[1] - depth / 4:
                        middle_y += 1
                    else:
                        outside_y += 1
                    if height / 4 < z < vertical_bounds[1] - height / 4:
                        middle_z += 1
                    else:
                        outside_z += 1
                    if x < horizontal_bounds[0] + width / 2:
                        left_half += 1
                    else:
                        right_half += 1
                    if y < depth_bounds[0] + depth / 2:
                        bottom_half += 1
                    else:
                        top_half += 1
                    if z < height / 2:
                        near_half += 1
                    else:
                        far_half += 1

    horizontal_footprint = left_half/right_half
    depth_footprint = near_half/far_half
    vertical_footprint = top_half/bottom_half
    horizontal_middle = middle_x/outside_x
    depth_middle = middle_y/outside_y
    vertical_middle = middle_z/outside_z

    return [horizontal_footprint, depth_footprint, vertical_footprint, horizontal_middle, depth_middle, vertical_middle]


def height_symmetry(lattice, horizontal_bounds, vertical_bounds, depth_bounds):
    symmetry_count = 0
    for x in range(horizontal_bounds[0], horizontal_bounds[1]):
        for y in range(depth_bounds[0], depth_bounds[1]):
            for z in range(vertical_bounds[0], int(vertical_bounds[1] / 2)):
                if lattice[x][y][z] == lattice[x][y][int(vertical_bounds[1] / 2) + z]:
                    symmetry_count += 1
    return symmetry_count


def width_symmetry(lattice, horizontal_bounds, vertical_bounds, depth_bounds):
    symmetry_count = 0
    width = horizontal_bounds[1] - horizontal_bounds[0]
    for x in range(horizontal_bounds[0], int(horizontal_bounds[1] / 2)):
        for y in range(depth_bounds[0], depth_bounds[1]):
            for z in range(vertical_bounds[0], vertical_bounds[1]):
                if lattice[x][y][z] == lattice[int(width / 2) + x][y][z]:
                    symmetry_count += 1
    return symmetry_count


def depth_symmetry(lattice, horizontal_bounds, vertical_bounds, depth_bounds):
    symmetry_count = 0
    depth = depth_bounds[1] - depth_bounds[0]
    for x in range(horizontal_bounds[0], horizontal_bounds[1]):
        for y in range(depth_bounds[0], int(depth_bounds[1] / 2)):
            for z in range(vertical_bounds[0], vertical_bounds[1]):
                if lattice[x][y][z] == lattice[x][int(depth / 2) + y][z]:
                    symmetry_count += 1
    return symmetry_count


def stability(lattice):
    try:
        boolean_lattice = change_to_ones(lattice.copy())
        lattice_com = center_of_mass(boolean_lattice)
        floor_plan = np.zeros((lattice_dimensions[0], lattice_dimensions[1]))
        for i in range(0, lattice.shape[0]):
            for j in range(0, lattice.shape[1]):
                if lattice[i][j][0] == 3:
                    floor_plan[i][j] = 1
        (floor_x, floor_y) = center_of_mass(floor_plan)
        return distance.euclidean((floor_x, floor_y, 0), lattice_com), distance.euclidean((floor_x, floor_y), (lattice_com[0], lattice_com[1]))
    except (ValueError, RuntimeError):
        raise InfeasibleError


def iterative_flood_interior(input_lattice):
    visited = np.zeros(input_lattice.shape, int)
    space_found = False
    for i, j, k in value_range:
        if visited[i][j][k] == 0 and input_lattice[i][j][k] == 1:
            visited, space_found_new = detect_structure(input_lattice, visited, 1, (i, j, k), any_type=False)
            if not space_found:
                space_found = space_found_new
    if not space_found:
        raise InfeasibleInteriorVolume


def fill_tiny_gaps(input_lattice):
    visited = np.zeros(input_lattice.shape, int)
    for i, j, k in value_range:

        # Make sure this space hasn't been visited and is interior air
        if visited[i][j][k] == 0 and input_lattice[i][j][k] == 1:

            # If the first voxel above this space is solid, this isn't traversable.
            if input_lattice[i][j][k+1] != 1:
                input_lattice[i][j][k] = 2
            # Otherwise drill upward till the next solid block is found and count the air voxels.
            else:
                for drill in range(k+1, 20):
                    if input_lattice[i][j][drill] == 1:
                        visited[i][j][drill] = 1
                    else:
                        break
    return input_lattice


def iterative_flood(input_lattice):
    """
    Given an input lattice perform an exhaustive flood-fill algorithm from the bottom of the
    XY plane to identify any floating voxels detached from the building inside lattice.
    Also detects separate structures and labels them accordingly.
    :param input_lattice: original lattice generated by CPPN-NEAT genome.
    :return: original lattice with floating voxels removed.
    """
    visited = np.zeros(input_lattice.shape, int)
    label = 0

    for i in range(0, visited.shape[0]):
        for j in range(0, visited.shape[1]):
            if visited[i][j][0] == 0 and input_lattice[i][j][0] == 1:
                label += 1
                visited, _ = detect_structure(input_lattice, visited, label, (i, j, 0))

    visited = keep_largest_structure(visited, label)
    if label == 0:
        raise InfeasibleError

    return visited


def detect_structure(lattice, visited, label, coordinate, any_type=True):
    """
    :param lattice:
    :param visited:
    :param label:
    :param coordinate:
    :param any_type:
    :return:
    """
    to_fill = set()
    to_fill.add(coordinate)
    counter = 0

    # Keep looping whilst the set of remaining unvisited voxels is empty.
    while len(to_fill) != 0:

        # Pop the first element in the to-fill list.
        voxel = to_fill.pop()
        counter += 1
        # If the voxel is active, mark it as true in the boolean grid and add it's neighbors
        if (lattice[voxel[0]][voxel[1]][voxel[2]] != 0 and any_type) or (not any_type and lattice[voxel[0]][voxel[1]][voxel[2]] == 1):
            visited[voxel[0]][voxel[1]][voxel[2]] = label
            if voxel[0] < 19 and not visited[voxel[0] + 1][voxel[1]][voxel[2]]:
                to_fill.add((voxel[0] + 1, voxel[1], voxel[2]))
            if voxel[0] > 0 and not visited[voxel[0] - 1][voxel[1]][voxel[2]]:
                to_fill.add((voxel[0] - 1, voxel[1], voxel[2]))
            if voxel[1] < 19 and not visited[voxel[0]][voxel[1] + 1][voxel[2]]:
                to_fill.add((voxel[0], voxel[1] + 1, voxel[2]))
            if voxel[1] > 0 and not visited[voxel[0]][voxel[1] - 1][voxel[2]]:
                to_fill.add((voxel[0], voxel[1] - 1, voxel[2]))
            if voxel[2] < 19 and not visited[voxel[0]][voxel[1]][voxel[2] + 1]:
                to_fill.add((voxel[0], voxel[1], voxel[2] + 1))
            if voxel[2] > 0 and not visited[voxel[0]][voxel[1]][voxel[2] - 1]:
                to_fill.add((voxel[0], voxel[1], voxel[2] - 1))

    # Return the boolean grid but converted to integer values to align with the rest of the pipeline
    return np.asarray(visited, dtype=int), counter > 125


def keep_largest_structure(visited, label):
    """
    :param visited:
    :param label:
    :return:
    """
    if label == 0:
        return visited

    # number_of_voxels contains the number of each of the labeled voxels
    number_of_voxels = np.zeros(label, int)

    for i in range(0, number_of_voxels.shape[0]):
        number_of_voxels[i] = np.count_nonzero(visited == i + 1)

    # keeps the pointer of the max voxels from the array number_of_voxels
    most_voxels = np.where(number_of_voxels == np.amax(number_of_voxels))

    # the label that has the most voxels in the visited matrix
    keep_voxel = most_voxels[0][0] + 1

    # set the rest of the voxels to 0 and the keep_voxel elements to 1
    for i in range(0, visited.shape[0]):
        for j in range(0, visited.shape[1]):
            for k in range(0, visited.shape[2]):
                if visited[i][j][k] != 0 and not visited[i][j][k] == keep_voxel:
                    visited[i][j][k] = 0
                elif visited[i][j][k] != 0 and visited[i][j][k] == keep_voxel:
                    visited[i][j][k] = 1

    return visited


def identify_materials(lattice):
    """

    :param lattice:
    :return:
    """
    for x, y, z in value_range:
        if lattice[x][y][z] != 0:

            if z == 0:
                if lattice[x][y][z + 1] == 0:
                    lattice[x][y][z] = 0
                else:
                    lattice[x][y][z] = materials['Floor']
                continue
            elif z == lattice_dimensions[0] - 1:
                lattice[x][y][z] = materials['Roof']
                continue
            else:
                if lattice[x][y][z + 1] == 0:
                    lattice[x][y][z] = materials['Roof']
                    continue

                if lattice[x][y][z - 1] == 0:
                    if lattice[x][y][z + 1] == 0:
                        lattice[x][y][z] = 0
                    else:
                        lattice[x][y][z] = materials['Floor']

            if x == 0 or y == 0:
                lattice[x][y][z] = materials['Wall']
            try:
                if lattice[x + 1][y][z] == 0:
                    lattice[x][y][z] = materials['Wall']
                elif lattice[x - 1][y][z] == 0:
                    lattice[x][y][z] = materials['Wall']
                elif lattice[x][y + 1][z] == 0:
                    lattice[x][y][z] = materials['Wall']
                elif lattice[x][y - 1][z] == 0:
                    lattice[x][y][z] = materials['Wall']
            except IndexError:
                lattice[x][y][z] = materials['Wall']

    return lattice


def change_to_ones(input_lattice, keep_interior=False):
    """

    :param input_lattice:
    :param keep_interior
    :return: 
    """
    for i in range(0, input_lattice.shape[0]):
        for j in range(0, input_lattice.shape[1]):
            for k in range(0, input_lattice.shape[2]):
                if input_lattice[i][j][k] > 1:
                    input_lattice[i][j][k] = 1
                elif not keep_interior:
                    input_lattice[i][j][k] = 0
    return input_lattice

if __name__ == "__main__":
    human_population = np.load("Real-World Datasets/Ahousev5_Buildings_Fixed.npy", allow_pickle=True)
    voxel_plot(identify_materials(human_population[1]), "")