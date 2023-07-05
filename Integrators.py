import mfem.par as mfem
import numpy as np

# This integrates the K matrix in the documentation.
class ElasticIntegrator(mfem.BilinearFormIntegrator):
    def __init__(
        self,
        elasticity_tensor
    ):
        self.elasticity_tensor = elasticity_tensor

    def AssembleElementMatrix(
        self,
        el,
        Trans,
        elmat
    ):
        num_dofs = el.GetDof()
        num_dims = el.GetDim()
        elmat.SetSize(num_dofs * num_dims)
        elmat.Assign(0.0)

        dshape = mfem.DenseMatrix(num_dofs, num_dims)

        # Need to get the max polynomial order for the quadrature rule.
        # This rule assumes that the elasticity tensor is constant over the
        # element.
        # If so, the rule should be exact; otherwise, it is approximate.
        if el.Space() == mfem.FunctionSpace.Pk:
            integration_order = 2*el.GetOrder() - 2 + Trans.OrderW()
        else:
            integration_order = 2*el.GetOrder() + el.GetDim() - 1 + Trans.OrderW()
        int_rule = mfem.IntRules.Get(el.GetGeomType(), integration_order)

        for ip in range(int_rule.GetNPoints()):
            int_point = int_rule.IntPoint(ip)
            Trans.SetIntPoint(int_point)

            weight = Trans.Weight() * int_point.weight

            el.CalcPhysDShape(Trans, dshape)

            x = mfem.Vector(num_dims)
            Trans.Transform(int_point, x)

            for idim in range(num_dims):
                for jdim in range(num_dims):
                    for idof in range(num_dofs):
                        for jdof in range(num_dofs):
                            # The vector FES simply comprises num_dims copies
                            # of a scalar FES,
                            # which has num_dofs DoFs in this cell.

                            strain = mfem.DenseMatrix(num_dims)
                            strain.Assign(0.0)
                            for k in range(num_dims):
                                strain[jdim, k] += 0.5 * dshape[jdof, k]
                                strain[k, jdim] += 0.5 * dshape[jdof, k]

                            # Note that the only nonzero column of
                            # \nabla \hat{\psi_i}
                            # is the idim-th column,
                            # so we only have to compute that column
                            # of the stress tensor.
                            stress_tensor = np.zeros(num_dims)
                            for stress_i in range(num_dims):
                                for k in range(num_dims):
                                    for l in range(num_dims):
                                        stress_tensor[stress_i] += self.elasticity_tensor.evaluate(x, stress_i, idim, k, l) * strain[k, l]

                            ii = idim*num_dofs + idof
                            jj = jdim*num_dofs + jdof
                            for k in range(num_dims):
                                elmat[ii, jj] -= weight * stress_tensor[k] * dshape[idof,k]
