import numpy as np
# import scipy.linalg as sl
import scipy.sparse as sps
# import scipy.sparse.linalg as spsl


class NoiseGaussianIID:
    def __init__(self, dim, dtype=np.float64):
        '''
        Description: Noise distribution N(mean, var^2 Id)
        Input:
            dim: int or np.int32 or np.int64
        '''
        assert type(dim) in [int, np.int32, np.int64]
        self.dim = dim
        self.dtype = dtype
        self.mean = np.zeros(self.dim, dtype=dtype)
        self.std_dev = np.array(1.0, dtype=dtype)

    def set_parameters(self, mean=None, std_dev=None):
        if mean is None:
            # defalut value is zero
            self.mean = np.zeros(self.dim, dtype=self.dtype)
        else:
            assert len(mean) == self.dim
            self.mean = np.array(mean, dtype=self.dtype)

        if std_dev is None:
            self.std_dev = np.array(1.0, dtype=self.dtype)
        else:
            self.std_dev = np.array(std_dev, dtype=self.dtype)

    def eval_CM_inner(self, u, v=None):
        if v is None:
            v = np.copy(u)
        assert len(u) == self.dim
        uu = u - self.mean
        vv = v - self.mean
        val = np.sum(uu*vv)/(self.std_dev**2)
        return val

    def generate_sample(self):
        val = self.mean + self.generate_sample_zero_mean()
        return np.array(val)

    def generate_sample_zero_mean(self):
        rand_vec = np.random.normal(0, 1, (self.dim,))
        sample = self.std_dev*rand_vec
        return np.array(sample, dtype=self.dtype)

    def precision_times_param(self, vec):
        return np.array(vec/(self.std_dev**2))













