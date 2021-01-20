from data_provider import DataProvider
from attribute_builder import AttributeBuilder
import numpy as np
from rbf import rbf
from attribute import Attribute
from guard import Guard


class Benchmark:
    def __init__(self):
        params = DataProvider()
        self.attributes = params.get_linear_params().copy()

    def test_apple(self):
        data_names_apple = ['apple 1Y', 'apple 3M.csv', 'apple 6M.csv']
        linear_kernel = 'linear_kernel'

    def test_amazon(self):
        data_names_amazon = ['AMZN 1Y.csv', 'AMZN 3M.csv', 'AMZ 6M.csv']
        pass

    def test_facebook(self):
        fb_attribute = self.attributes.copy()
        data_names_facebook = ['FB 1Y.csv']

        fb_attribute['data_name'] = data_names_facebook
        self.attributes['data_name'].values = data_names_facebook
        self.attributes['kernels'].values = ['linear']
        fb_train = AttributeBuilder(self.attributes)
        fb_train.create()
        fb_train.execute()
        pass

    def test(self, data_name_apple):
        paramters = AttributeBuilder(self.attributes)
        paramters.create()
        paramters.execute()
