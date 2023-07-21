import mfem.par as mfem
import numpy as np
import abc

class BodyForce(mfem.VectorPyCoefficientT, abc.ABC):
    def __init__(self):
        pass

    def setVDim(self, vdim):
        mfem.VectorPyCoefficientT.__init__(self, vdim)

    @abc.abstractmethod
    def EvalValue(self, x, t):
        pass

class ZeroBodyForce(BodyForce):
    def EvalValue(self, x, t):
        return np.zeros(self.vdim)
