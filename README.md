# cavernus
Library for FEM analysis of salt caverns

# Theory
Salt cavern rock moves with a velocity that is small compared to
$L/\tau$, where $L$ and $\tau$ are characteristic length and time scales
of the system, respectively.
They are therefore well described quasistatically.
The displacement ${u} = {u}({x}, t)$ of the rock from
its unforced state in a $d$-dimensional
domain $\Omega$ is governed by the elasticity equation,
which can be written as
$$\nabla \cdot (C : \nabla^{\mathrm{s}} {u}) = \nabla \cdot (C : \epsilon_{\mathrm{cr}}) + f^{\mathrm{b}} \text{ in } \Omega,$$
$${u} = {g} \text{ on } \Gamma_{\mathrm{D}}$$
$$\hat{n} \cdot (C : \nabla^{\mathrm{s}} {u}) = {h} \text{ on } \Gamma_{\mathrm{N}},$$
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
strain rate is a function of the displacement
and the inelastic creep strain itself,
$\dot{\epsilon_{\mathrm{cr}}} = F ({u}, \epsilon_{\mathrm{cr}})$.
This time dependence drives the evolution of ${u}$.

A continuous finite element method is used to discretize the elasticity equation
in space.
Let $V \subset H^1(\Omega)$ be a subspace of a Sobolev space on $\Omega$.
For each $v \in V$, weak solutions $u, \epsilon_{\mathrm{cr}}$ must satisfy
$$\int_\Omega \nabla \cdot (C : \nabla^{\mathrm{s}}u) v \, \mathrm{d}V = \int_{\Omega} \nabla \cdot (C : \epsilon_{\mathrm{cr}}) v \, \mathrm{d}V + \int_{\Omega} f^{\mathrm{b}} v \, \mathrm{d} V$$
$$\Rightarrow - \int_{\Omega} \nabla v \cdot (C : \nabla^{\mathrm{s}} u) \, \mathrm{d}V + \int_{\Gamma_{\mathrm{N}}} h v \, \mathrm{d} S = \int_{\Omega} \nabla \cdot (C : \epsilon_{\mathrm{cr}}) v \, \mathrm{d}V + \int_{\Omega} f^{\mathrm{b}} v \, \mathrm{d}V,$$
where the second equation arises through integration by parts.

Let $V_h \subset V$ be a finite-dimensional subspace of $V$
parametrized by a length scale $h$
and let $\{ \psi_1, \ldots, \psi_N\}$ be a basis for $V_h$.
Approximate the exact solutions as
$$u \approx u_h = \sum_{i=1}^N u_i \psi_i,$$
$$\epsilon_{\mathrm{cr}} \approx \epsilon_h = \sum_{i=1}^N \epsilon_i \psi_i.$$
Now $u_h$ and $\epsilon_h$ must satisfy
$$-\sum_{j=1}^N u_j \int_\Omega \nabla \psi_i \cdot (C : \nabla^{\mathrm{s}} \psi_j) \, \mathrm{d}V = - \int_{\Gamma_{\mathrm{N}}} h \psi_i \, \mathrm{d}S + \sum_{j=1}^N \epsilon_j \int_{\Omega} \nabla \cdot (C : \psi_j) \psi_i \, \mathrm{d}V+ \int_\Omega f^{\mathrm{b}} \psi_i \, \mathrm{d} V$$
for each $1 \leq i \leq N$.
Let $K$ be the matrix whose entries are given by
$$K_{ij} = - \int_\Omega \nabla \psi_i \cdot (C : \nabla^{\mathrm{s}} \psi_j) \, \mathrm{d}V,$$
let $G^{\mathrm{int}} = G^{\mathrm{int}}(\epsilon_{\mathrm{cr}})$
and $G^{\mathrm{bdr}} = G^{\mathrm{bdr}}(\epsilon_{\mathrm{cr}})$
be the operators defined by
$$G_i^{\mathrm{int}}(\epsilon_{\mathrm{cr}}) = - \int_\Omega \nabla \psi_i \cdot (C : \epsilon_{\mathrm{cr}}) \, \mathrm{d}V,$$
$$G_i^{\mathrm{bdr}}(\epsilon_{\mathrm{cr}}) = \int_{\partial \Omega} \hat{n} \cdot (C : \epsilon_{\mathrm{cr}}) \psi_i \, \mathrm{d}S,$$
and let $\vec{b}$ and $\vec{h}$ be the vectors whose entries are given by
$$b_i = \int_\Omega f^{\mathrm{b}} \psi_i \, \mathrm{d}V,$$
$$h_i = - \int_{\Gamma_{\mathrm{N}}} h \psi_i \, \mathrm{d}S.$$
Then the weak form of the elasticity equation becomes
$$K \vec{u} = G^{\mathrm{int}} (\epsilon_{\mathrm{cr}}) + G^{\mathrm{bdr}}(\epsilon_{\mathrm{cr}}) + \vec{b} + \vec{h}.$$

Recall that the time evolution of the system is governed by
the inelastic creep strain rate
$$\frac{\mathrm{d} \epsilon_{\mathrm{cr}}}{\mathrm{d}t} = F(u, \epsilon_{\mathrm{cr}}) \approx F(\vec{u}, \epsilon_{\mathrm{cr}}),$$
$$\epsilon_{\mathrm{cr}}(t=0) = \epsilon_0.$$
An interval of time is uniformly partitioned into time steps of size $\Delta t$.
We use the family of Runge-Kutta (RK) ordinary differential equation integrators
to progress from $t=0$ in time steps.
Throughout the remainder of this document,
we use a superscript to denote the time step at which a time-dependent
quantity is evaluated,
as $\xi^n = \xi(n \Delta t)$.
RK methods estimate the value at the next time step as
$$\epsilon_{\mathrm{cr}}^{n+1} = \epsilon_{\mathrm{cr}}^n + \Delta t \sum_{i=1}^s b_i k_i,$$
where
$$k_i = F \left( \vec{y}_i, \epsilon_{\mathrm{cr}}^n + \Delta t \sum_{j=1}^s a_{ij} k_j \right),$$
$\vec{y}_i$ is the solution for $\vec{u}$ given
$\epsilon_{\mathrm{cr}} = \epsilon_{\mathrm{cr}}^n + \Delta t \sum_{j=1}^s a_{ij} k_j$ at time $t = t^n + c_i \Delta t$,
$s$ is known as the number of stages for the particular method,
and the $a_{ij}$, $b_i$, and $c_i$ are constants that depend on the
specific choice of method.

The constants are typically expressed in the form of a Butcher tableau

LTM add an example of a butcher tableau

RK methods are categorized as either explicit or implicit.
Explicit methods are simpler, but require $\Delta t$ to satisfy
certain stability conditions which depend on the method.
Meanwhile, implicit methods are more complex and expensive,
but relax the time step size constraint.
These will be discussed respectively in the following sections.

## Explicit RK methods
For explicit methods, $a_{ij} = 0$ if $j \geq i$.
In other words, the $a$ matrix is strictly lower triangular.
The main consequence of this is that
$k_i$ at some stage depends only on the values of $k_j$ for previous stages,
so that $\vec{y}_i$ is the solution for $\vec{u}$ given
$\epsilon_{\mathrm{cr}} = \epsilon_{\mathrm{cr}}^n + \Delta t \sum_{j=1}^{i-1} a_{ij} k_j$ at time $t=t^n + c_i \Delta t$.
In other words, the estimates at each stage can be explicitly evaluated
from estimates at previous stages.

The simplest example of an explicit RK methods is the forward Euler method,
which is a single-stage method whose Butcher tableau is

LTM add butcher tableau here

Written explicitly,
$$\epsilon_{\mathrm{cr}}^{n+1} = \epsilon_{\mathrm{cr}}^n + \Delta t F(\vec{u}^n, \epsilon_{\mathrm{cr}}^n).$$

## Implicit RK methods
Implicit methods do not have lower triangular $a$ matrices.
Hence $k_i$ at some stage depends implicitly on future stages.

The simplest example of an implicit RK method is the backward Euler method,
which is a single-stage method whose Butcher tableau is

LTM add butcher tableau here

Written explicitly,
$$\epsilon_{\mathrm{cr}}^{n+1} = \epsilon_{\mathrm{cr}}^n + \Delta t F(\vec{u}^{n+1}, \epsilon_{\mathrm{cr}}^{n+1}).$$
