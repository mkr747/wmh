class SvrParameters:
    @property
    def data_name(self):
        return self._data_name

    @property
    def cv(self):
        return self._cv

    @property
    def kernel(self):
        return self._kernel

    @property
    def degree(self):
        return self._degree

    @property
    def gamma(self):
        self._gamma

    @property
    def coef0(self):
        return self._coef0

    @property
    def tol(self):
        return self._tol

    @property
    def c(self):
        return self._c

    @property
    def epsilon(self):
        return self._epsilon

    @property
    def shrinking(self):
        return self._shrinking
