import mfem.par as mfem
import numpy as np
import abc

class InitialInelasticCreepStrain(mfem.VectorPyCoefficient, abc.ABC):
    def __init__(self):
        pass

    def setVDim(self, vdim):
        mfem.VectorPyCoefficient.__init__(self, vdim)

    @abc.abstractmethod
    def EvalValue(self, x):
        pass

class ZeroInitialInelasticCreepStrain(InitialInelasticCreepStrain):
    def EvalValue(self, x):
        return np.zeros(self.vdim)
