# cavernus
Library for FEM analysis of salt caverns

# Theory
Salt cavern rock moves with a velocity that is small compared to
$L/\tau$, where $L$ and $\tau$ are characteristic length and time scales
of the system, respectively.
They are therefore well described quasistatically.
The displacement ${u} = {u}({x}, t) \rightarrow \mathbb{R}^d$ of the rock from
its unforced state in a $d$-dimensional
domain $\Omega$ is governed by the elasticity equation,
which can be written as
$$\nabla \cdot (C : \nabla^{\mathrm{s}} {u}) = \nabla \cdot (C : \epsilon_{\mathrm{cr}}) + f^{\mathrm{b}} \text{ in } \Omega,$$
$${u} = {g} \text{ on } \Gamma_{\mathrm{D}}$$
$$\hat{n} \cdot (C : \nabla^{\mathrm{s}} {u}) = {h} \text{ on } \Gamma_{\mathrm{N}},$$
where $C = C({x}, t)$ is the elasticity or stiffness tensor,
$\nabla^{\mathrm{s}} = \frac{1}{2}(\nabla + \nabla^t)$ is the symmetrized gradient,
$\epsilon_{\mathrm{cr}} \rightarrow \mathbb{R}^M$ is the inelastic creep strain,
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
For each $v \in V^d$, weak solutions $u, \epsilon_{\mathrm{cr}}$ must satisfy
$$\int_\Omega v \cdot \nabla \cdot (C : \nabla^{\mathrm{s}}u) \, \mathrm{d}V = \int_{\Omega} v \cdot \nabla \cdot (C : \epsilon_{\mathrm{cr}}) \, \mathrm{d}V + \int_{\Omega} f^{\mathrm{b}} \cdot v \, \mathrm{d} V$$
$$\Rightarrow - \int_{\Omega} \langle \nabla v, (C : \nabla^{\mathrm{s}} u) \rangle_{\mathrm{F}} \, \mathrm{d}V + \int_{\Gamma_{\mathrm{N}}} h \cdot v \, \mathrm{d} S = \int_{\Omega} v \cdot \nabla \cdot (C : \epsilon_{\mathrm{cr}}) \, \mathrm{d}V + \int_{\Omega} f^{\mathrm{b}} \cdot v \, \mathrm{d}V,$$
where the second equation arises through integration by parts
and $\langle \cdot, \cdot \rangle_{\mathrm{F}}$ denotes the
Frobenius inner product.

Let $V_h \subset V$ be a finite-dimensional subspace of $V$
parametrized by a length scale $h$
and let $B_h = \{ \psi_1, \ldots, \psi_N\}$ be a basis for $V_h$.
Approximate the exact solutions as
$$u \approx u_h = \sum_{i=1}^{Nd} u_i \hat{\psi_i},$$
$$\epsilon_{\mathrm{cr}} \approx \epsilon_h = \sum_{i=1}^{Nd(d+1)/2} \epsilon_i \bar{\psi_i},$$
where $\hat{\psi_i} \in B_h^d$ and $\bar{\psi_i} \in B_h^M$.
Now $u_h$ and $\epsilon_h$ must satisfy
$$-\sum_{j=1}^{Nd} u_j \int_\Omega \langle \nabla \hat{\psi_i}, (C : \nabla^{\mathrm{s}} \hat{\psi_j}) \rangle_{\mathrm{F}} \, \mathrm{d}V = - \int_{\Gamma_{\mathrm{N}}} h \cdot \hat{\psi_i} \, \mathrm{d}S + \sum_{j=1}^{Nd(d+1)/2} \epsilon_j \int_{\Omega} \hat{\psi_i} \cdot \nabla \cdot (C : \bar{\psi_j}) \, \mathrm{d}V+ \int_\Omega f^{\mathrm{b}} \cdot \hat{\psi_i} \, \mathrm{d} V$$
for each $\hat{\psi_i} \in B_h^d$.
Let $K$ be the $Nd \times Nd$ matrix whose entries are given by
$$K_{ij} = - \int_\Omega \langle \nabla \hat{\psi_i} \cdot (C : \nabla^{\mathrm{s}} \hat{\psi_j}) \rangle_{\mathrm{F}} \, \mathrm{d}V,$$
let $G$ be the $Nd \times Nd(d+1)/2$ matrix whose entries are given by
$$G_{ij} = \int_{\Omega} \hat{\psi_i} \cdot \nabla \cdot (C : \bar{\psi_j}) \, \mathrm{d}V$$
and let $\vec{b}$ and $\vec{h}$ be the vectors whose entries are given by
$$b_i = \int_\Omega f^{\mathrm{b}} \cdot \hat{\psi_i} \, \mathrm{d}V,$$
$$h_i = - \int_{\Gamma_{\mathrm{N}}} h \cdot \hat{\psi_i} \, \mathrm{d}S.$$
Then the weak form of the elasticity equation becomes
$$K \vec{u} = G \vec{\epsilon} + \vec{b} + \vec{h},$$
where $\vec{u} = (u_1, \ldots, u_{Nd})^t$
and $\vec{\epsilon} = (\epsilon_1, \ldots, \epsilon_{Nd(d+1)/2})^t$.

Recall that the time evolution of the system is governed by
the inelastic creep strain rate
$$\frac{\mathrm{d} \epsilon_{\mathrm{cr}}}{\mathrm{d}t} = F(u, \epsilon_{\mathrm{cr}}),$$
$$\epsilon_{\mathrm{cr}}(t=0) = \epsilon^0.$$
The weak solution $\epsilon_h$ must satisfy the weak form
$$\frac{\mathrm{d}}{\mathrm{d}t} \sum_{j=1}^{Nd(d+1)/2} \epsilon_j \int_\Omega \langle \bar{\psi_i}, \bar{\psi_j} \rangle_{\mathrm{F}} \, \mathrm{d}V = \int_\Omega \langle F(u_h, \epsilon_h), \bar{\psi_i} \rangle_{\mathrm{F}} \, \mathrm{d}V$$
for each $\bar{\psi_i} \in B_h^M$.
Let $M$ be the matrix whose entries are given by
$$M_{ij} = \int_\Omega \langle \bar{\psi_i}, \bar{\psi_j} \rangle_{\mathrm{F}} \, \mathrm{d}V,$$
and let $\vec{F}(u, \epsilon_{\mathrm{cr}})$ be the
nonlinear operator whose entries are given by
$$F_i(u, \epsilon_{\mathrm{cr}}) = \int_\Omega \langle F(u, \epsilon_{\mathrm{cr}}), \bar{\psi_i} \rangle_{\mathrm{F}} \, \mathrm{d}V.$$
This document will also occasionally denote
$$\vec{F}(\vec{u}, \vec{\epsilon}) = \vec{F} \left( \sum_{i=1}^{Nd} u_i \hat{\psi_i}, \sum_{i=1}^{Nd(d+1)/2} \epsilon_i \bar{\psi_i}\right) $$
for brevity.
The weak form becomes
$$\frac{\mathrm{d} \vec{\epsilon}}{\mathrm{d}t} = M^{-1} \vec{F} (\vec{u}, \vec{\epsilon}).$$

An interval of time is uniformly partitioned into time steps of size $\Delta t$.
We use the family of Runge-Kutta (RK) ordinary differential equation integrators
to progress from $t=0$ in time steps.
Throughout the remainder of this document,
we use a superscript to denote the time step at which a time-dependent
quantity is evaluated,
as $\xi^n = \xi(n \Delta t)$.
RK methods estimate the value at the next time step as
$$\vec{\epsilon}^{n+1} = \vec{\epsilon}^n + \Delta t \sum_{i=1}^s b_i \vec{k_i},$$
where $\vec{k_i}, \vec{y_i}$ satisfy
$$M \vec{k_i} = \vec{F} \left( \vec{y_i}, \vec{\epsilon}^n + \Delta t \sum_{j=1}^s a_{ij} \vec{k_j} \right),$$
$$K \vec{y_i} = G (\vec{\epsilon}^n + \Delta t \sum_{j=1}^s a_{ij} \vec{k_j}) + \vec{b} + \vec{h}$$
with $t = t^n + c_i \Delta t$,
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
$\vec{k}_i$ at some stage depends only on the values of $\vec{k}_j$ for previous stages.

The simplest example of an explicit RK methods is the forward Euler method,
which is a single-stage method whose Butcher tableau is

LTM add butcher tableau here

Written explicitly,
$$\vec{\epsilon}^{n+1} = \vec{\epsilon}^n + \Delta t M^{-1} \vec{F}(\vec{u}^n, \vec{\epsilon}^n),$$
$$K \vec{u}^{n+1} = G \vec{\epsilon}^{n+1} + \vec{b}^{n+1} + \vec{h}^{n+1}.$$
With this method, the inelastic creep strain at the next time step
can be directly written using quantities from the current time step.
The displacement at the next time step can then be solved as usual.

## Implicit RK methods
Implicit methods do not have lower triangular $a$ matrices.
Hence $k_i$ at some stage depends implicitly on future stages.

The simplest example of an implicit RK method is the backward Euler method,
which is a single-stage method whose Butcher tableau is

LTM add butcher tableau here

Written explicitly,
$$\vec{\epsilon}^{n+1} = \vec{\epsilon}^n + \Delta t M^{-1} \vec{F}(\vec{u}^{n+1}, \vec{\epsilon}^{n+1}).$$
