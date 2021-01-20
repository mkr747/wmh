import numpy as np
from rbf import rbf
from attribute import Attribute


class AttributeBuilder:
    def __init__(self, attributs={}):
        self._params_to_execute = []
        self._attributes = attributs.copy()

    def get_attribute(self, name):
        return self._attributes[name]

    def create(self):
        params = []
        for key in self._attributes.keys():
            print(len(params))
            if(self._attributes[key].is_enabled):
                length = len(params)
                if length == 0:
                    row = dict()
                    for value in self._attributes[key].values:
                        row[key] = value
                        params.append(row)
                else:
                    for i in range(length):
                        for value in self._attributes[key].values:
                            if key in params[i]:
                                row = params[i].copy()
                                row[key] = value
                                params.append(row)
                            else:
                                params[i][key] = value

        print('Finished')
        self._params_to_execute = params

    def execute(self):
        [rbf(expression) for expression in self._params_to_execute]
