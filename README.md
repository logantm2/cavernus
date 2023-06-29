# cavernus
Library for FEM analysis of salt caverns

# Theory
Salt cavern rock moves with a velocity that is small compared to
$L/\tau$, where $L$ and $\tau$ are characteristic length and time scales
of the system, respectively.
They are therefore well described quasistatically.
The displacement ${u} = {u}({x}, t)$ of the rock from
its unforced state in a domain $\Omega$ is governed by the elasticity equation,
which can be written as
$$
\begin{align}
\nabla \cdot (C : \nabla^{\mathrm{s}} {u}) = \nabla \cdot (C : \epsilon_{\mathrm{cr}}) + f^{\mathrm{b}} && \text{in } \Omega, \\
{u} = {g} && \text{on } \Gamma_{\mathrm{D}}, \\
\hat{n} \cdot (C : \nabla^{\mathrm{s}} {u}) = {h} && \text{on } \Gamma_{\mathrm{N}},
\end{align}
$$
where $C = C({x}, t)$ is the elasticity or stiffness tensor,
$\nabla^{\mathrm{s}} = \frac{1}{2}(\nabla + \nabla^T)$ is the symmetrized gradient,
$\epsilon_{\mathrm{cr}}$ is the inelastic creep strain,
$f^{\mathrm{b}} = f^{\mathrm{b}}({x}, t)$ is a known body force,
${g} = {g}({x}, t)$ is a Dirichlet boundary condition
imposed on a portion of the boundary $\Gamma_{\mathrm{D}}$,
and
${h} = {h}({x}, t)$ is a Neumann boundary condition imposed
on a portion of the boundary $\Gamma_{\mathrm{N}}$.
We require that $\overline{\Gamma_{\mathrm{D}} \cup \Gamma_{\mathrm{N}}} = \partial \Omega$.

A constitutive relation for the inelastic creep strain is required
in order to close the system.
We are interested in the case where the inelastic creep
strain rate is a function of the displacement,
$\dot{\epsilon}_{\mathrm{cr}} = \dot{\epsilon}_{\mathrm{cr}} ({u})$.
This time dependence drives the evolution of ${u}$.

A continuous finite element method is used to discretize the elasticity equation.
Let $V \subset H^1(\Omega)$ be a subspace of a Sobolev space on $\Omega$.
For each $v \in V$, a weak solution $u$ must satisfy
$$\int_\Omega \nabla \cdot (C : \nabla^{\mathrm{s}}u) v \, \mathrm{d}V = \int_{\Omega} \nabla \cdot (C : \epsilon_{\mathrm{cr}}) v \, \mathrm{d}V + \int_{\Omega} f^{\mathrm{b}} v \, \mathrm{d} V$$
$$\begin{align*}
\Rightarrow - \int_{\Omega} \nabla v \cdot (C : \nabla^{\mathrm{s}} u) \, \mathrm{d}V &+ \int_{\partial \Omega} \hat{n} \cdot (C : \nabla^{\mathrm{s}} u) v \, \mathrm{d} S \\
&= - \int_\Omega \nabla v \cdot (C : \epsilon_{\mathrm{cr}}) \, \mathrm{d}V + \int_{\partial \Omega} \hat{n} \cdot (C : \epsilon_{\mathrm{cr}}) v \, \mathrm{d}S + \int_{\Omega} f^{\mathrm{b}} v \, \mathrm{d}V,
\end{align*}$$
where the second equation arises through integration by parts.
