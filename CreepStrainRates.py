import Utils
import mfem.par as mfem
import scipy.constants as constants
import numpy as np
import abc

class CreepStrainRate(abc.ABC):
    # Evaluate the elasticity tensor at position x
    # and at index ijkl.
    @abc.abstractmethod
    def evaluate(self, displacement, creep_strain, creep_strain_rate):
        pass

class CarterCreepStrainRate(CreepStrainRate):
    def __init__(
        self,
        constant,
        exponent,
        activation_energy,
        temperature,
        elasticity_tensor
    ):
        self.constant = constant
        self.exponent = exponent
        self.activation_energy = activation_energy
        self.temperature = temperature
        self.elasticity_tensor = elasticity_tensor

    def evaluate(
        self,
        displacement_fe,
        creep_strain_fe,
        element_transformation,
        displacement_mat,
        creep_strain_mat,
        creep_strain_rate
    ):
        u_num_dofs = displacement_fe.GetDof()
        u_num_dims = displacement_fe.GetVDim()
        e_num_dofs = creep_strain_fe.GetDof()
        e_num_dims = creep_strain_fe.GetVDim()

        u_dshape = mfem.DenseMatrix(u_num_dofs, u_num_dims)
        displacement_fe.CalcPhysDShape(element_transformation, u_dshape)
        e_shapef = mfem.Vector(e_num_dofs)
        creep_strain_fe.CalcPhysShape(element_transformation, e_shapef)

        grad_u = mfem.DenseMatrix(u_num_dims)
        mfem.MultAtB(displacement_mat, u_dshape, grad_u)
        grad_u.Symmetrize()
        total_strain = Utils.flattenSymmetricTensor(grad_u)

        creep_strain = mfem.Vector(e_num_dims)
        creep_strain_mat.MultTranspose(e_shapef, creep_strain)

        elastic_strain = total_strain - creep_strain

        x = element_transformation.Transform(element_transformation.GetIntPoint())

        stress = self.elasticity_tensor.calcFlattenedStress(x, elastic_strain)
        stress_deviator, hydrostatic_stress = Utils.calcFlattenedDeviator(stress)
        vM_stress = Utils.calcVonMisesStress(stress_deviator, hydrostatic_stress)
        rate_coefficient = 3./2. * self.constant * np.exp(-self.activation_energy/constants.gas_constant/self.temperature) * np.power(vM_stress, self.exponent)

        creep_strain_rate = stress_deviator * rate_coefficient
