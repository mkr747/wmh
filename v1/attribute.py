class Attribute:
    def __init__(self, name, is_enabled, values):
        self._name = name
        self._is_enabled = is_enabled
        self._values = values

    @property
    def name(self):
        return self._name

    @property
    def is_enabled(self):
        return self._is_enabled

    @property
    def values(self):
        return self._values

    @name.setter
    def name(self, value):
        self._name = value

    @is_enabled.setter
    def is_enabled(self, value):
        self._is_enabled = value

    @values.setter
    def values(self, value):
        self._values = value
