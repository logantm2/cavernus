import mfem.par as mfem
import numpy as np
from scipy.constants import g, proton_mass, Boltzmann
import abc

# Used to specify the displacement, as a vector, on the boundary.
class DirichletBoundaryCondition(mfem.VectorPyCoefficientT, abc.ABC):
    def __init__(
        self,
        boundary_attribute
    ):
        self.boundary_attribute = boundary_attribute
        self.type = "dirichlet"

    def setVDim(self, vdim):
        mfem.VectorPyCoefficientT.__init__(self, vdim)

    @abc.abstractmethod
    def EvalValue(self, x, t):
        pass

class ZeroDirichletBoundaryCondition(DirichletBoundaryCondition):
    def __init__(self, boundary_attribute):
        super().__init__(boundary_attribute)

    def EvalValue(self, x, t):
        return np.zeros(x.size)

# Used to specify the stress, as a matrix, on the boundary.
class NeumannBoundaryCondition(mfem.VectorPyCoefficientBase, abc.ABC):
    def __init__(
        self,
        boundary_attribute
    ):
        self.boundary_attribute = boundary_attribute
        self.type = "neumann"

    def setVDim(self, vdim):
        mfem.VectorPyCoefficientBase.__init__(self, vdim, 1)

    # Return the stress tensor at this point in time.
    @abc.abstractmethod
    def EvalValue(self, x, t):
        pass

    def Eval(self, V, T, ip):
        vdim = self.GetVDim()
        x = T.Transform(ip)
        V.SetSize(vdim)
        V.Assign(0.0)

        stress = self.EvalValue(x, self.GetTime())

        # Compute the normal on this face.
        normal = mfem.Vector(vdim)
        T.SetIntPoint(ip)
        mfem.CalcOrtho(T.Jacobian(), normal)
        # Normal is not unit, so normalize it.
        norm_mag = normal.Norml2()
        for i in range(vdim):
            normal[i] = normal[i] / norm_mag

        # Contract normal with stress tensor.
        for i in range(vdim):
            for j in range(vdim):
                V[i] = V[i] + stress[i,j] * normal[j]

class ZeroNeumannBoundaryCondition(NeumannBoundaryCondition):
    def EvalValue(self, x, t):
        return np.zeros((x.size, x.size))

class KumarMinPressureBC(NeumannBoundaryCondition):
    def __init__(
        self,
        boundary_attribute,
        rho,
        temp,
        cavern_midpoint_depth,
        cavern_height
    ):
        super().__init__(boundary_attribute)
        self.rho = rho
        self.temp = temp
        self.cavern_midpoint_depth = cavern_midpoint_depth
        self.cavern_height = cavern_height

        self.lithostatic_pressure = rho * g
        cavern_bottom_depth = cavern_midpoint_depth + cavern_height/2.0
        # minimum allowable cavern pressure
        self.pc = 0.2 * self.lithostatic_pressure * cavern_bottom_depth
        h2_number_density = self.pc / Boltzmann / temp
        self.rho_h2 = 2.0 * proton_mass * h2_number_density # approximate

    def EvalValue(self, x, t):
        depth_in_cavern = self.cavern_height/2.0 - x[-1]
        pressure = self.pc + self.rho_h2 * g * depth_in_cavern - self.lithostatic_pressure * (self.cavern_midpoint_depth + x[-1])

        return np.diag(pressure * np.ones(x.size))
