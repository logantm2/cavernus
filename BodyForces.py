import mfem.par as mfem
import numpy as np
import abc

class BodyForce(mfem.VectorPyCoefficient, abc.ABC):
    def __init__(self, dim):
        mfem.VectoyPyCoefficient.__init__(self, dim)

    @abc.abstractmethod
    def EvalValue(self, x):
        pass

class ZeroBodyForce(BodyForce):
    def EvalValue(self, x):
        return np.zeros(self.vdim)
