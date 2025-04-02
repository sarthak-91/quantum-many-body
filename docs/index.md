
---
layout: default
title: Home
---
# Variational Method in Quantum Mechanics 

Solving Time Independent Schrodinger's Equation  $H\psi = E\psi$ exactly in many cases is not easy. There are very few problems that can be exactly solved. For example only the simplest atom Hydrogen has a closed form solution for the Schrodinger's Equation. So what do we do? We come up with methods of approximations.  Perturbation theory is one example of this method. It relies on splitting the Hamiltonian into two: $H = H_0 + H_p$ where $H_0$ has a form that we know the solution to and $H_p$ is treated as a small perturbation of the Hamiltonian that produces perturbation in the wavefunction and the energy eigenvalues. There is another method: the Variational Method which as the name implies relies on finding the closest solution varying some parameters of concern. 

The method goes like this: 
1. Choose a trial wavefunction $\psi_\vec{\theta}$ that depends on a parameters which is written here as a vector $\vec{\theta} = {\theta_1, \theta_2, \theta_3, ... \theta_n}$  
2. Compute the expectation value of the Hamiltonian $E_\vec{\theta} = \frac{<\psi_\vec{\theta}|H|\psi_\vec{\theta}>}{<\psi_\vec{\theta}|\psi_\vec{\theta}>}$  
3. Choose the optimal values of ${\theta_1, \theta_2, \theta_3, ... \theta_n}$ by minimizing $E_\vec{\theta}$ 

Let us try the Hydrogen atom.  
The Schrodinger's equation for the radial part of the hydrogen atom is:
$-\frac{\hbar^2}{2m} \frac{d^2 u(r)}{dr^2} + \frac{\hbar^2}{2m} \frac{l(l+1)}{r^2} u(r) - \frac{u(r)}{4\pi\epsilon_0 r} = E u(r)$   
where $u(r) = rR(r)$ 
 
Writing in atomic units (basically setting the constants to 1) the equation becomes:
$-\frac{1}{2}\frac{d^2 u}{d\rho^2} + \frac{l(l+1)}{2\rho^2} u - \frac{u}{\rho} = \mathcal{E} u$

where, 
$\rho=\frac{r}{r_0}$,
$r_0 = \frac{\hbar^2}{m e^2}$  ,
$E_0 = \frac{\hbar^2}{mr_0^2}$ and 
$\mathcal{E} = \frac{E}{E_0}$
In atomic units, the distances are measured in Bohr's Radius and energy is measured in Hartrees. 

So the Hamiltonian becomes: 
$H =  -\frac{1}{2}\frac{d^2}{d\rho^2} + \frac{l(l+1)}{2\rho^2}  - \frac{1}{\rho}$
Now how do we guess the trial wavefunction?  Well to do that we have to be careful in our motivation to choose the trial wavefunction. One method is to look at what happens in the extremes like $\rho \to 0$ and $\rho \to \infty$. When $\rho$ is large,  the solution is dominated by a negative exponential of the form $e^{-\alpha\rho}$ and when  $\rho$ is small the solution is dominated by $\rho^{l+1}$. So for the ground state (l=0) we can guess our ansatz to be of the form:
$u_\alpha(\rho) = \rho e^{-\alpha \rho}$ where $\alpha$ is the variational parameter of our wavefunction which is restricted to be more than 0. That is step number 1. 

Now for step 2, we have $<u(\rho)|u(\rho)> = \frac{1}{4\alpha^3}$ and $<u(\rho)|H|u(\rho)>$ = $\frac{\alpha-1}{4\alpha^2} - \frac{1}{8\alpha}$ 
Putting it all together, the expectation of the Hamiltonian has the form: 
$E(\alpha)$ = $\frac{\alpha^2}{2} - \alpha$. 
It is easy to see that the minimum value of $E(\alpha)$ is -0.5 at $\alpha=1$. Energies in atomic units is called Hartrees which is 27.2114 eV so the ground state energy we obtained in eV is 13.605 eV. Fortunately for us, our ansatz turned out to have the exact form as the ground state of hydrogen.     

<script type="text/javascript" async
  src="https://polyfill.io/v3/polyfill.min.js?features=es6">
</script>
<script type="text/javascript" async
  src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js">
</script>
