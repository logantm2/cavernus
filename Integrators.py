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

            x = Trans.Transform(int_point)

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

# This integrates the G matrix in the documentation,
# which is generally not square.
class InelasticIntegrator(mfem.BilinearFormIntegrator):
    def __init__(
        self,
        elasticity_tensor
    ):
        self.elasticity_tensor = elasticity_tensor

    # The trial finite element space comprises symmetric matrices
    # whereas the test finite element space comprises vectors.
    # The entries of the trial space are stored in a flattened vector.
    # In 1D, it's just a scalar.
    # In 2D, store as [e_xx, e_yy, e_xy].
    # In 3D, store as [e_xx, e_yy, e_zz, e_xy, e_yz, e_xz]
    def AssembleElementMatrix2(
        self,
        trial_fe,
        test_fe,
        Trans,
        elmat
    ):
        space_dims = trial_fe.GetDim()

        trial_num_dofs = trial_fe.GetDof()
        # trial_num_dims = trial_fe.GetVDim()
        trial_num_dims = space_dims * (space_dims+1) // 2
        test_num_dofs = test_fe.GetDof()
        test_num_dims = space_dims

        elmat.SetSize(test_num_dofs*test_num_dims, trial_num_dofs*trial_num_dims)
        elmat.Assign(0.0)

        # Gradients of the trial space functions
        trial_dshape = mfem.DenseMatrix(trial_num_dofs, space_dims)
        # Test space functions
        test_shapef = mfem.Vector(test_num_dofs)

        # Need to get the max polynomial order for the quadrature rule.
        # This rule assumes that the elasticity tensor is constant over the
        # element.
        # If so, the rule should be exact; otherwise, it is approximate.
        if test_fe.Space() == mfem.FunctionSpace.Pk:
            integration_order = test_fe.GetOrder() + trial_fe.GetOrder() - 1 + Trans.OrderW()
        else:
            integration_order = test_fe.GetOrder() + trial_fe.GetOrder() + space_dims + Trans.OrderW()
        int_rule = mfem.IntRules.Get(trial_fe.GetGeomType(), integration_order)

        nabla_C_e = mfem.Vector(space_dims)

        # Given that we're looking at the i-th component of the trial function,
        # (which, again, is a flattened vector containing symmetric matrices),
        # nonzero_e_entries[i] contains the two nonzero matrix elements.
        # For diagonal elements, repeat the element indices.
        if space_dims == 1:
            nonzero_e_entries = np.array([
                [ [0,0], [0,0] ]
            ])
        elif space_dims == 2:
            nonzero_e_entries = np.array([
                [ [0,0], [0,0] ],
                [ [1,1], [1,1] ],
                [ [0,1], [1,0] ]
            ])
        elif space_dims == 3:
            nonzero_e_entries = np.array([
                [ [0,0], [0,0] ],
                [ [1,1], [1,1] ],
                [ [2,2], [2,2] ],
                [ [0,1], [1,0] ],
                [ [1,2], [2,1] ],
                [ [0,2], [2,0] ]
            ])

        for ip in range(int_rule.GetNPoints()):
            int_point = int_rule.IntPoint(ip)
            Trans.SetIntPoint(int_point)

            weight = Trans.Weight() * int_point.weight

            trial_fe.CalcPhysDShape(Trans, trial_dshape)
            test_fe.CalcPhysShape(Trans, test_shapef)

            x = Trans.Transform(int_point)

            # TODO optimize this.
            for idim in range(test_num_dims):
                for jdim in range(trial_num_dims):
                    for idof in range(test_num_dofs):
                        for jdof in range(trial_num_dofs):
                            ii = idim*test_num_dofs  + idof
                            jj = jdim*trial_num_dofs + jdof

                            nabla_C_e.Assign(0.0)
                            for k in range(space_dims):
                                for l in range(space_dims):
                                    nabla_C_e[k] += 0.5 * trial_dshape[jdof, l] * self.elasticity_tensor.evaluate(
                                        x,
                                        k,
                                        l,
                                        nonzero_e_entries[jdim,0,0],
                                        nonzero_e_entries[jdim,0,1]
                                    )
                                    nabla_C_e[k] += 0.5 * trial_dshape[jdof, l] * self.elasticity_tensor.evaluate(
                                        x,
                                        k,
                                        l,
                                        nonzero_e_entries[jdim,1,0],
                                        nonzero_e_entries[jdim,1,1]
                                    )

                            elmat[ii, jj] += weight * test_shapef[idof] * nabla_C_e[idim]
