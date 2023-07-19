import InitialConditions
import BodyForces
import Integrators
import BoundaryConditions
import ElasticityTensors
import CreepStrainRates

import mfem.par as mfem

from mpi4py import MPI

class CavernusOperator(mfem.PyTimeDependentOperator):
    def __init__(
        self,
        ufespace,
        efespace,
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

        # G operator.
        self.G_operator = mfem.ParMixedBilinearForm(efespace, ufespace)
        self.G_operator.AddDomainIntegrator(Integrators.InelasticIntegrator(self.elasticity_tensor))

        # F operator.
        self.F_operator = mfem.ParBlockNonlinearForm([ufespace, efespace])
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
        self.G_operator.Assemble() # LTM is this needed?
        self.G_operator.AddMult(epsilon_gf, self.u_linear_form)
        self.K_operator.Assemble() # LTM is this needed?

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
    dt = input["dt"]
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
    exit(0)

    def visualize(out, pmesh, deformed_nodes, field,
                field_name='', init_vis=False):
        nodes = deformed_nodes
        owns_nodes = 0

        nodes, owns_nodes = pmesh.SwapNodes(nodes, owns_nodes)

        out.send_text("parallel " + str(num_procs) + " " + str(myid))
        out.send_solution(pmesh, field)

        nodes, owns_nodes = pmesh.SwapNodes(nodes, owns_nodes)

        if (init_vis):
            out.send_text("window_size 400 400")
            out.send_text("window_title '" + field_name)
            if (pmesh.SpaceDimension() == 2):
                out.send_text("view 0 0")
                out.send_text("keys jl")
            out.send_text("keys cm")         # show colorbar and mesh
            # update value-range; keep mesh-extents fixed
            out.send_text("autoscale value")
            out.send_text("pause")
        out.flush()


    oper = HyperelasticOperator(fespace, ess_bdr, visc,  mu, K)
    if (visualization):
        vis_v = mfem.socketstream("localhost", 19916)
        vis_v.precision(8)
        visualize(vis_v, pmesh, x_gf, v_gf, "Velocity", True)

        MPI.COMM_WORLD.Barrier()
        vis_w = mfem.socketstream("localhost", 19916)
        oper.GetElasticEnergyDensity(x_gf, w_gf)
        vis_w.precision(8)
        visualize(vis_w, pmesh, x_gf, w_gf, "Elastic energy density", True)

    # 8. Perform time-integration (looping over the time iterations, ti, with a
    #    time-step dt).
    t = 0.
    ti = 1

    oper.SetTime(t)
    ode_solver.Init(oper)
    last_step = False

    while not last_step:
        dt_real = min(dt, t_final - t)
        t, dt = ode_solver.Step(vx, t, dt_real)

        if (t >= t_final - 1e-8*dt):
            last_step = True

        if (last_step or (ti % vis_steps) == 0):
            v_gf.Distribute(vx.GetBlock(0))
            x_gf.Distribute(vx.GetBlock(1))

            ee = oper.ElasticEnergy(x_gf)
            ke = oper.KineticEnergy(v_gf)

            text = ("step " + str(ti) + ", t = " + str(t) +
                    ", EE = " + "{:g}".format(ee) +
                    ", KE = " + "{:g}".format(ke) +
                    ", dTE = " + "{:g}".format((ee+ke)-(ee0+ke0)))

            if myid == 0:
                print(text)
            if visualization:
                visualize(vis_v, pmesh, x_gf, v_gf)
                oper.GetElasticEnergyDensity(x_gf, w_gf)
                visualize(vis_w, pmesh, x_gf, w_gf)

        ti = ti + 1

    #
    # if i translate c++ line-by-line, ti seems the second swap does not work...
    #

    smyid = '{:0>6d}'.format(myid)
    mesh_name = "deformed."+smyid
    velo_name = "velocity."+smyid
    ee_name = "elastic_energy."+smyid

    nodes = x_gf
    owns_nodes = 0
    nodes, owns_nodes = pmesh.SwapNodes(nodes, owns_nodes)
    pmesh.Print(mesh_name, 8)
    pmesh.SwapNodes(nodes, owns_nodes)

    v_gf.Save(velo_name, 8)
    oper.GetElasticEnergyDensity(x_gf, w_gf)
    w_gf.Save(ee_name,  8)

if __name__ == "__main__":
    # PCG solver with BoomerAMG preconditioner
    amg = mfem.HypreBoomerAMG()
    amg.SetPrintLevel(0)
    linear_solver = mfem.HyprePCG(MPI.COMM_WORLD)
    linear_solver.SetTol(1.e-12)
    linear_solver.SetMaxIter(200)
    linear_solver.SetPrintLevel(0)
    linear_solver.SetPreconditioner(amg)

    youngs_modulus = 35.e9
    poisson_ratio = 0.25
    l = youngs_modulus*poisson_ratio/(1.+poisson_ratio)/(1.-poisson_ratio)
    mu = youngs_modulus/2./(1.+poisson_ratio)
    elasticity_tensor = ElasticityTensors.ConstantIsotropicElasticityTensor(l, mu)

    inputs = {
        "order" : 1,
        "ode_solver" : mfem.ForwardEulerSolver(),
        "t_final" : 275.0 * 24.0 * 3600.0, # Seconds in 275 days
        "dt" : 1.5 * 24.0 * 3600.0, # Seconds in 1.5 days
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
            8.1e-27,
            3.5,
            51600.,
            300.0,
            elasticity_tensor
        )
    }

    main(inputs)
