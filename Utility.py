import numpy as np
from Delenox_Config import value_range, lattice_dimensions


def lattice_to_file(lattice, output_counter):
    file = open('Lattice_Dumps/lattice' + output_counter + '.txt', 'w')
    for channel in lattice:
        for row in channel:
            np.ndarray.tofile(row, file, ', ')
            file.write('\n')
        file.write('\n')


def get_min_elements_list(list1, number):
    final_list = []
    for i in range(number):
        min_value = None
        for item in list1:
            if min_value is None:
                min_value = item
            elif item < min_value:
                min_value = item
        list1.remove(min_value)
        final_list.append(min_value)
    return final_list


def get_max_elements_list(list1, number):
    final_list = []
    for i in range(number):
        max_value = None
        for item in list1:
            if max_value is None:
                max_value = item
            elif item > max_value:
                max_value = item
        list1.remove(max_value)
        final_list.append(max_value)
    return final_list


def get_max_elements(list1, number):
    final_list = {}
    for i in range(number):
        max1 = None
        key1 = None
        for key, genome in list1:
            if max1 is None:
                max1 = genome
                key1 = key
            elif genome.fitness > max1.fitness:
                max1 = genome
                key1 = key
        del list1[key1]
        final_list.update({key1: max1})
    return final_list


def calculate_error(original, reconstruction):
    error = 0
    for (x, y, z) in value_range:
        if original[x][y][z] != np.round(reconstruction[x][y][z]):
            error += 1
    return round(error / (lattice_dimensions[0] ** 3) * 100, 2)


def add_noise(lattice):
    noisy_lattice = lattice.copy()
    for (x, y, z) in value_range:
        if np.random.random() < 0.025:
            noisy_lattice[x][y][z] = 0
    return noisy_lattice
