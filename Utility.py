def get_min_elements_list(list1, number):
    """

    """
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
    """

    """
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
    """

    """
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
