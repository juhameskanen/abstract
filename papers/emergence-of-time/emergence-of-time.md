# Algorithmic Selection of Time and Quantum Structure  
## from a Static Wheeler–DeWitt Universe

**Juha Meskanen**  
December 2025



## Abstract

Canonical quantum gravity predicts a timeless universe governed by the Wheeler–DeWitt equation,
in apparent conflict with the observed flow of time, unitary quantum dynamics, and finite
renormalized physics. We propose an **Algorithmic Selection Principle (ASP)**, according to which
observer-experienced reality is dominated by relational histories admitting minimal description
length. Interpreting the Wheeler–DeWitt wavefunction in an Everettian manner as a static ensemble
of correlated configurations, we show that the choice of relational clock, the arrow of time,
quantum unitarity, the Born rule, and the absence of physical singularities arise as consequences
of algorithmic typicality relative to self-maintaining information-processing structures
(observers). In this framework, spacetime geometry, quantum probabilities, and renormalized
expectation values are emergent features of compression-optimal histories, while divergences and
singularities correspond to vanishing-information configurations that are algorithmically
suppressed and observationally inaccessible.



## 1. Ontological Framework

We adopt the canonical formulation of quantum gravity, in which the universal state
$\Psi[h_{ij}(\mathbf{x}), \phi(\mathbf{x})]$ satisfies the Wheeler–DeWitt (WdW) equation

$$
\hat{H} \Psi = 0,
$$

and contains no fundamental time parameter. We interpret $\Psi$ in an Everettian sense as a static
superposition over all admissible three-geometries and matter field configurations:

$$
\lvert \Psi \rangle = \sum_i c_i \lvert h_i, \phi_i \rangle.
$$

In this view, the universe is a timeless informational structure. Any notion of dynamics must
arise from correlations internal to $\Psi$, rather than from external temporal evolution.



## 2. Algorithmic Selection Principle

**Algorithmic Selection Principle (ASP).**  
*Among all relational histories consistent with the universal state $\Psi$, observer-experienced
reality is dominated by histories whose total description length is minimal.*

Let $S$ denote a relational history induced by a particular factorization and ordering of
configurations. The relative weight of $S$ is

$$
P(S) \propto 2^{-K(S)},
$$

where $K(S)$ is the Kolmogorov complexity (or minimal description length) of $S$.

Description length is evaluated **relative to a self-maintaining information-processing
structure**. Histories incompatible with observer persistence are not experienced, regardless of
their formal existence within $\Psi$.



## 3. Emergence of Quantum Structure

### 3.1 Wavefunction as Optimal Encoding

A finite observer must encode correlations along its experienced history using a representation
that minimizes reconstruction error while preserving predictive power. The complex wavefunction
$\psi$ emerges as the minimal encoding that supports linear superposition, phase coherence, and
stable composition of subsystems.

Highly irregular configurations require longer descriptions and are suppressed under ASP,
leading to an effective low-frequency, smooth structure.



### 3.2 Unitarity from Informational Stability

Persistence of observer identity requires that successive encodings preserve total information
content. In a linear representation space, this restricts admissible transformations to
norm-preserving maps.

Continuous norm preservation uniquely selects unitary evolution, with a Hermitian generator
$\hat{H}$, yielding

$$
i \hbar \frac{\partial}{\partial t} \psi = \hat{H} \psi.
$$



## 4. The Born Rule from Algorithmic Optimality

We now strengthen the derivation of the Born rule by identifying it as the **unique decoding
rule** compatible with algorithmic selection and observer persistence.



### 4.1 Problem Statement

Let $\psi \in \mathcal{H}$ be the observer’s compressed encoding of relational correlations.
A probability rule assigns to each outcome $i$ a weight

$$
P(i) = f(|\psi_i|),
$$

where $f$ is a non-negative function.

We seek the unique $f$ compatible with physically motivated constraints.



### 4.2 Constraints

**(C1) Additivity under coarse-graining**

If outcomes $i$ and $j$ are grouped,

$$
P(i \cup j) = P(i) + P(j).
$$

**(C2) Composition consistency**

For independent subsystems $A$ and $B$,

$$
P(i_A, j_B) = P(i_A) P(j_B).
$$

**(C3) Basis invariance**

Probabilities must be invariant under unitary change of basis.

**(C4) Reconstruction optimality**

The decoding rule must minimize mean squared reconstruction error between encoded and decoded
histories. This is the unique optimal decoder for finite observers using lossy compression.



### 4.3 Uniqueness of the Squared Norm

Constraints (C1) and (C2) imply that $f(x)$ must be quadratic in amplitude magnitude:

$$
f(x) = k x^2.
$$

Constraint (C3) excludes dependence on phase or basis-dependent quantities.

Constraint (C4) selects the $L^2$ norm uniquely: among all $L^p$ norms, only $p = 2$ yields linear
projections, orthogonality preservation, and stable error minimization under compression.

Normalization fixes $k = 1$, yielding

$$
P(i) = |\psi_i|^2.
$$

**Conclusion.**  
The Born rule is not an independent axiom, but the unique probability assignment compatible with
algorithmic compression, compositional consistency, and observer persistence in Hilbert space.



## 5. Singularities and Renormalization

Configurations corresponding to classical singularities or ultraviolet divergences possess
vanishing or ill-defined informational structure. Such states require maximal description length
and are exponentially suppressed under ASP. Their observational probability is zero.

Renormalization in quantum field theory arises as a direct consequence: high-frequency modes are
algorithmically incompressible and do not contribute to dominant observer-compatible histories.
Effective field theories are therefore maximal-compressibility approximations to the static
informational substrate.



## 6. Discussion

Spacetime geometry, time, quantum probabilities, and classicality emerge as features of
algorithmically typical correlations within a timeless universal state. Singularities and
divergences mark the boundaries of informational accessibility, not physical breakdowns.

The Algorithmic Selection Principle shifts the explanatory burden from fundamental dynamics to
informational selection, providing a unified resolution of the problem of time, the origin of
quantum probabilities, and the absence of observable infinities.
