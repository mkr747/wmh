import pandas as pd
from test import Test


def main():
    second_part_tests()


def first_part_tests():
    data_names = ['AMZN 1y.csv', 'AMZN 3M.csv', 'AMZN 6M.csv', 'FB 1Y.csv']
    for data_name in data_names:
        run_tests(data_name)


def second_part_tests():
    data_names = ['FB 3M.csv', 'FB 6M.csv',
                  'apple 1Y.csv', 'apple 3M.csv', 'apple 6M.csv']
    for data_name in data_names:
        run_tests(data_name)


def run_tests(data_name):
    test = Test()
    test.linear_search_grid_test(data_name)
    test.poly_search_grid_test(data_name)
    #test.precomputed_search_grid_test(data_name)
    test.rbf_search_grid_test(data_name)
    test.sigmoid_search_grid_test(data_name)

if __name__ == "__main__":
    main()
