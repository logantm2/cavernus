import InitialConditions
import BodyForces
import Integrators
import BoundaryConditions
import ElasticityTensors
import CreepStrainRates
import Utils

import mfem.par as mfem

from mpi4py import MPI

# The built-in PyMFEM wrapper for the ParNonlinearForm doesn't work,
# so I made a reduced-capacity one here that does the job in the one
# place that we use the ParNonlinearForm.
class NFWrapper(mfem.ParNonlinearForm):
    def __init__(
        self,
        pf
    ):
        super().__init__(pf)
        self.dnfi = []

    def AddDomainIntegrator(self, nlfi):
        super().AddDomainIntegrator(nlfi)
        self.dnfi.append(nlfi)

    def Mult(self, x, y):
        P = self.GetProlongation()
        px = mfem.Vector(P.Height())
        P.Mult(x, px)
        py = mfem.Vector(P.Height())
        py.Assign(0.0)

        fes = self.FESpace()
        el_x = mfem.Vector()
        el_y = mfem.Vector()
        for i in range(fes.GetNE()):
            fe = fes.GetFE(i)
            vdofs = mfem.intArray(fes.GetElementVDofs(i))
            T = fes.GetElementTransformation(i)
            px.GetSubVector(vdofs, el_x)
            for k in range(len(self.dnfi)):
                self.dnfi[k].AssembleElementVector(fe, T, el_x, el_y)
                py.AddElementVector(vdofs, el_y)

        P.MultTranspose(py, y)

class CavernusOperator(mfem.PyTimeDependentOperator):
    def __init__(
        self,
        ufespace,
        efespace,
        combined_fespace,
        body_force,
        boundary_conditions,
        elasticity_tensor,
        linear_solver,
        creep_strain_rate
    ):
        mfem.PyTimeDependentOperator.__init__(
            self,
            ufespace.TrueVSize() + efespace.TrueVSize(),
            0.0
        )

        self.ufespace = ufespace
        self.efespace = efespace
        self.boundary_conditions = boundary_conditions
        self.elasticity_tensor = elasticity_tensor
        self.linear_solver = linear_solver
        self.creep_strain_rate = creep_strain_rate

        mesh = ufespace.GetMesh()

        # First set up the linear form for the u problem.
        self.u_linear_form = mfem.ParLinearForm(ufespace)
        # This will handle the integration of the body force...
        self.u_linear_form.AddDomainIntegrator(mfem.VectorDomainLFIntegrator(body_force))
        # ... and the integration of the Neumann boundary conditions.
        # We will also collect Dirichlet boundary Dofs here.
        ess_bdr = mfem.intArray(mesh.bdr_attributes.Max())
        ess_bdr.Assign(0)
        self.bc_marker_arrays = []
        for boundary_condition in boundary_conditions:
            bc_marker_array = mfem.intArray(mesh.bdr_attributes.Max())
            bc_marker_array.Assign(0)
            bc_marker_array[boundary_condition.boundary_attribute-1] = 1
            self.bc_marker_arrays.append(bc_marker_array)
            if boundary_condition.type == "neumann":
                # Need to subtract the boundary conditions from the linear form.
                neg_boundary_condition = mfem.ScalarVectorProductCoefficient(-1.0, boundary_condition)
                zeros = mfem.Vector(2)
                zeros.Assign(0.0)
                self.u_linear_form.AddBoundaryIntegrator(
                    mfem.VectorBoundaryLFIntegrator(neg_boundary_condition),
                    self.bc_marker_arrays[-1]
                )
            elif boundary_condition.type == "dirichlet":
                ess_bdr[boundary_condition.boundary_attribute-1] = 1
        self.ess_tdof_list = mfem.intArray()
        ufespace.GetEssentialTrueDofs(ess_bdr, self.ess_tdof_list)


        # Now set up the K operator.
        self.K_operator = mfem.ParBilinearForm(ufespace)
        self.K_operator.AddDomainIntegrator(Integrators.ElasticIntegrator(self.elasticity_tensor))
        self.K_operator.Assemble()

        # G operator.
        self.G_operator = mfem.ParMixedBilinearForm(efespace, ufespace)
        self.G_operator.AddDomainIntegrator(Integrators.InelasticIntegrator(self.elasticity_tensor))
        self.G_operator.Assemble()

        # F operator.
        self.F_operator = NFWrapper(combined_fespace)
        self.F_operator.AddDomainIntegrator(Integrators.CreepStrainRateIntegrator(self.creep_strain_rate))

        # M operator.
        epsilon_dims = self.efespace.GetVDim()
        epsilon_ones = mfem.Vector(epsilon_dims)
        epsilon_ones.Assign(1.0)
        epsilon_ones_coef = mfem.VectorConstantCoefficient(epsilon_ones)
        self.M_operator = mfem.ParBilinearForm(efespace)
        self.M_operator.AddDomainIntegrator(mfem.VectorMassIntegrator(epsilon_ones_coef))
        self.M_operator.Assemble(0)
        self.M_operator.Finalize(0)
        self.Mmat = self.M_operator.ParallelAssemble()

        # Scratch space
        self.A_ = mfem.OperatorPtr()
        self.X_ = mfem.Vector()
        self.B_ = mfem.Vector()
        self.u_gf_ = mfem.ParGridFunction(ufespace)
        self.e_gf_ = mfem.ParGridFunction(efespace)
        self.w_ = mfem.Vector(ufespace.TrueVSize() + efespace.TrueVSize())
        self.z_ = mfem.Vector(ufespace.TrueVSize() + efespace.TrueVSize())

    # Given the inelastic creep strain (epsilon) at this instant,
    # solve for the displacement.
    def instantaneousDisplacement(self, epsilon_gf, u_gf):
        self.u_linear_form.Assemble()
        self.G_operator.AddMult(epsilon_gf, self.u_linear_form)

        # Project Dirichlet boundary conditions to solution.
        for i, bc in enumerate(self.boundary_conditions):
            if bc.type == "dirichlet":
                u_gf.ProjectBdrCoefficient(bc, self.bc_marker_arrays[i])

        self.K_operator.FormLinearSystem(
            self.ess_tdof_list,
            u_gf,
            self.u_linear_form,
            self.A_,
            self.X_,
            self.B_
        )

        self.linear_solver.SetOperator(self.A_.Ptr())
        self.linear_solver.Mult(self.B_, self.X_)

        self.K_operator.RecoverFEMSolution(
            self.X_,
            self.u_linear_form,
            u_gf
        )

    def Mult(self, u_epsilon, u_epsilon_dt):
        u_size = self.ufespace.TrueVSize()
        e_size = self.efespace.TrueVSize()
        epsilon = mfem.Vector(u_epsilon, u_size, e_size)
        du_dt = mfem.Vector(u_epsilon_dt, 0, u_size)
        depsilon_dt = mfem.Vector(u_epsilon_dt, u_size, e_size)

        # Calculate the displacement due to this creep strain
        self.e_gf_.Distribute(epsilon)
        self.instantaneousDisplacement(self.e_gf_, self.u_gf_)

        # Move this creep strain and displacement to scratch true dof vector.
        w_u = mfem.Vector(self.w_, 0, u_size)
        w_e = mfem.Vector(self.w_, u_size, e_size)
        self.u_gf_.GetTrueDofs(w_u)
        self.e_gf_.GetTrueDofs(w_e)

        # Apply the F operator to this creep strain and displacement.
        self.F_operator.Mult(self.w_, self.z_)

        # Apply inverse of M operator to get creep strain rate
        z_e = mfem.Vector(self.z_, u_size, e_size)
        self.linear_solver.SetOperator(self.Mmat)
        self.linear_solver.Mult(z_e, depsilon_dt)

        # Ensure that displacement "rate" is zero
        du_dt.Assign(0.0)

def main(input):
    num_procs = MPI.COMM_WORLD.size
    myid = MPI.COMM_WORLD.rank

    order = input["order"]
    ode_solver = input["ode_solver"]
    t_final = input["t_final"]
    num_timesteps = input["num_timesteps"]
    mesh_filename = input["mesh_filename"]
    initial_creep_strain = input["initial_creep_strain"]
    boundary_conditions = input["boundary_conditions"]
    linear_solver = input["linear_solver"]
    body_force = input["body_force"]
    elasticity_tensor = input["elasticity_tensor"]
    creep_strain_rate = input["creep_strain_rate"]

    device = mfem.Device('cpu')
    if myid == 0:
        device.Print()

    # 3. Read the serial mesh from the given mesh file on all processors. We can
    #    handle triangular, quadrilateral, tetrahedral and hexahedral meshes
    #    with the same code.
    mesh = mfem.Mesh(mesh_filename, 1, 1)
    dim = mesh.Dimension()

    # 6. Define a parallel mesh by a partitioning of the serial mesh. Once the
    #    parallel mesh is defined, the serial mesh can be deleted.
    pmesh = mfem.ParMesh(MPI.COMM_WORLD, mesh)
    del mesh

    # 7. Define the parallel vector finite element spaces representing the mesh
    #    deformation x_gf, the velocity v_gf, and the initial configuration,
    #    x_ref. Define also the elastic energy density, w_gf, which is in a
    #    discontinuous higher-order space. Since x and v are integrated in time
    #    as a system, we group them together in block vector vx, on the unique
    #    parallel degrees of freedom, with offsets given by array true_offset.

    fec = mfem.H1_FECollection(order, dim)

    ufespace = mfem.ParFiniteElementSpace(pmesh, fec, dim, mfem.Ordering.byNODES)
    uglob_size = ufespace.GlobalTrueVSize()
    if (myid == 0):
        print(f"Number of displacement unknowns: {uglob_size}")

    num_epsilon_components = dim * (dim+1) // 2
    efespace = mfem.ParFiniteElementSpace(pmesh, fec, num_epsilon_components, mfem.Ordering.byNODES)
    eglob_size = efespace.GlobalTrueVSize()
    if (myid == 0):
        print(f"Number of inelastic creep strain unknowns: {eglob_size}")
    combined_fespace = mfem.ParFiniteElementSpace(pmesh, fec, dim + num_epsilon_components, mfem.Ordering.byNODES)

    # Primal dofs are stored in separate MFEM grid functions.
    u_gf = mfem.ParGridFunction(ufespace)
    epsilon_gf = mfem.ParGridFunction(efespace)

    # Store the initial or reference displacements.
    x_ref = mfem.ParGridFunction(ufespace)
    pmesh.GetNodes(x_ref)

    # 8. Set the initial conditions for v_gf, x_gf and vx, and define the
    #    boundary conditions on a beam-like mesh (see description above).
    epsilon_gf.ProjectCoefficient(initial_creep_strain)

    operator = CavernusOperator(
        ufespace,
        efespace,
        combined_fespace,
        body_force,
        boundary_conditions,
        elasticity_tensor,
        linear_solver,
        creep_strain_rate
    )
    operator.instantaneousDisplacement(epsilon_gf, u_gf)

    # A block vector to store the true DoFs
    true_offset = mfem.intArray(3)
    true_offset[0] = 0
    true_offset[1] = ufespace.TrueVSize()
    true_offset[2] = ufespace.TrueVSize() + efespace.TrueVSize()
    u_epsilon = mfem.BlockVector(true_offset)
    # Transfer from initial condition in GridFunctions to Vector of true DoFs
    u_gf.GetTrueDofs(u_epsilon.GetBlock(0))
    epsilon_gf.GetTrueDofs(u_epsilon.GetBlock(1))

    # For data output
    data_collection = mfem.ParaViewDataCollection("cavernus", pmesh)
    data_collection.SetPrefixPath("output")
    data_collection.SetDataFormat(mfem.VTKFormat_BINARY)
    if order > 1:
        data_collection.SetHighOrderOutput(True)
    data_collection.SetLevelsOfDetail(order)
    data_collection.RegisterField("displacement", u_gf)
    data_collection.RegisterField("creep_strain", epsilon_gf)
    data_collection.SetCycle(0)
    data_collection.SetTime(0.0)
    data_collection.SaveMesh()
    data_collection.Save()

    # Time integration.
    time = 0.0
    timestep = 0
    operator.SetTime(time)
    ode_solver.Init(operator)
    last_step = False

    while not last_step:
        Utils.logAndContinue(
            "info",
            f"At timestep {timestep}",
            "main"
        )

        dt = min(t_final - time, t_final/num_timesteps)
        time, dt = ode_solver.Step(u_epsilon, time, dt)

        if (time >= t_final - 1.e-8*dt):
            last_step = True

        timestep = timestep + 1

    u_gf.Distribute(u_epsilon.GetBlock(0))
    epsilon_gf.Distribute(u_epsilon.GetBlock(1))
    operator.instantaneousDisplacement(epsilon_gf, u_gf)
    data_collection.SetCycle(timestep)
    data_collection.SetTime(time)
    data_collection.Save()

if __name__ == "__main__":
    # PCG solver with BoomerAMG preconditioner
    amg = mfem.HypreBoomerAMG()
    amg.SetPrintLevel(0)
    linear_solver = mfem.HyprePCG(MPI.COMM_WORLD)
    linear_solver.SetTol(1.e-12)
    linear_solver.SetMaxIter(200)
    linear_solver.SetPrintLevel(0)
    linear_solver.SetPreconditioner(amg)

    youngs_modulus = 44.e9
    poisson_ratio = 0.3
    l = youngs_modulus*poisson_ratio/(1.+poisson_ratio)/(1.-poisson_ratio)
    mu = youngs_modulus/2./(1.+poisson_ratio)
    elasticity_tensor = ElasticityTensors.ConstantIsotropicElasticityTensor(l, mu)

    inputs = {
        "order" : 1,
        "ode_solver" : mfem.ForwardEulerSolver(),
        "t_final" : 275.0 * 24.0 * 3600.0, # Seconds in 275 days
        "num_timesteps" : 185,
        "mesh_filename" : "test.msh",
        "initial_creep_strain" : InitialConditions.ZeroInitialInelasticCreepStrain(3),
        "boundary_conditions" : [
            BoundaryConditions.ZeroBoundaryCondition(2, 11, "neumann"), # cavern
            BoundaryConditions.ZeroBoundaryCondition(2, 21, "dirichlet"), # top
            BoundaryConditions.ZeroBoundaryCondition(2, 22, "dirichlet")  # right
        ],
        "linear_solver" : linear_solver,
        "body_force" : BodyForces.ZeroBodyForce(2),
        "elasticity_tensor" : elasticity_tensor,
        "creep_strain_rate" : CreepStrainRates.CarterCreepStrainRate(
            8.1e-28,
            3.5,
            51600.,
            298.0,
            elasticity_tensor
        )
    }

    main(inputs)
