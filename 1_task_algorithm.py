import sys

def multiplicate(input_arr):
    """
    Here we have 3 variance
    1) we have no zeros element
    [1, 2, 3, 4] -> [24, 12, 8, 6]
    2) we have only one zeros element in array => on i place
    is the product of all non-zeros elements and the other positions are 0
    [1, 2, 3, 0] -> [0, 0, 0, 6]
    3) we have more than one zeros element => on each place is zero
    [1, 2, 0, 0] -> [0, 0, 0, 0]
    """
    check_first_and_second_zero = 0
    result_prod = 1
    for i in input_arr:
        if i == 0 and check_first_and_second_zero == 0:
            check_first_and_second_zero = 1
        elif i == 0 and check_first_and_second_zero == 1:
            check_first_and_second_zero = 2
            result_prod = 0
            break
        else:
            result_prod *= i

    result_array = []
    if check_first_and_second_zero == 0:
        for elem in input_arr:
            result_array.append(result_prod // elem)
    if check_first_and_second_zero == 1:
        for elem in input_arr:
            if elem == 0:
                result_array.append(result_prod)
            else:
                result_array.append(0)
    if check_first_and_second_zero == 2:
        for elem in input_arr:
            result_array.append(0)
    return result_array

print(multiplicate(eval(sys.argv[1])))
#