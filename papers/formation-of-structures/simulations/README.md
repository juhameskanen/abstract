## 🌌 Physical Interpretation of Tuning Parameters

In this simulation, the "Laws of Physics" are not hard-coded. Instead, they emerge as the **stationary points** of a global spacetime functional. The behavior of the universe is governed by the balance between two primary forces: **Simplicity** and **Observation**.

### 1. $\kappa$ (Kappa): The Simplicity Prior / Vacuum Viscosity
**Code Variable:** `--kappa` (default: `0.0005`)

$\kappa$ represents the **Energy Cost of Information** in the vacuum. It acts as a 3D spectral regulator in Fourier space.

* **Theory:** In a computational ontology, the "most probable" universe is the one with the **Shortest Description Length** (Solomonoff Induction). $\kappa$ penalizes high-frequency "noise" (complex data), forcing the wavefunction to remain smooth.
* **Effect on Physics:** It is the origin of **Inertia**. A particle continues its state of motion because "changing" or "jittering" that motion would create high-frequency spectral spikes, which are computationally expensive. 
* **Too High ($\kappa \uparrow$):** The universe is "over-compressed." Spacetime becomes too viscous, and structures collapse into trivial, perfectly symmetric geometries (e.g., static disks).
* **Too Low ($\kappa \downarrow$):** The universe is "uncompressed." Without spectral tension, the wavefunction becomes jagged and chaotic, losing all physical coherence.



### 2. $\lambda$ (Lambda): The Observer Stiffness / Coupling Constant
**Code Variable:** `--lmbda` (default: `15.0`)

$\lambda$ represents the **Stiffness of the Partition Function** and the certainty of the observer's measurement.

* **Theory:** Based on the principle that $P(\gamma \mid O) \propto \exp(-\lambda C_O[\gamma])$, $\lambda$ defines how strictly the physical history ($\gamma$) must conform to the observer's filter ($O$). It is the "gravitational pull" of the bit-patterns on the wavefunction.
* **Effect on Physics:** It is the driver of **Structural Emergence**. It forces the smooth probability density to "snap" to the coordinates where the observer's pattern-matching occurs. It is what crystallizes "Stars" out of the vacuum.
* **Too High ($\lambda \uparrow$):** The universe is "Brittle." The observer dominates so much that the physical laws (smoothness) break; structures become noisy and "shredded."
* **Too Low ($\lambda \downarrow$):** The universe is "Haunted." The patterns exist in the underlying bit-string, but the wavefunction is too weak to be influenced by them. No stars are born.




## Theoretical Framework: The Variational Block Universe

The simulation treats the universe as a **Block of Data** where the "Arrow of Time" is defined by a global **Entropy Gradient** ($H_{start} \to H_{end}$).

### The Large-Deviation Principle
We posit that **Observed Physical Laws are the Large-Deviation Minimizers** of the complexity cost over observer-compatible histories. 

By minimizing the total loss function:
$$\mathcal{L} = \kappa \cdot \text{Complexity} + \lambda \cdot \text{Observer Match} + \text{Entropy Error}$$

We are numerically searching for the **Most Compressible History** that satisfies the boundary conditions of the Big Bang and the heat death. In this view:
* **Stars** are not "objects"; they are the most mathematically efficient way to transition from order to disorder.
* **Gravity** is not a "force"; it is the statistical pressure of the observer's selection criteria acting on the simplicity prior.



### Summary Table for Experimentation

| Parameter | Meaning | If Increased... | If Decreased... |
| :--- | :--- | :--- | :--- |
| **$\kappa$ (Kappa)** | Simplicity / Inertia | Rigid, static, symmetric shapes | Chaotic, jittery, noisy static |
| **$\lambda$ (Lambda)** | Observer Coupling | Brittle, pixelated structures | Faint, ghost-like "unformed" blobs |
| **$H_{start/end}$** | Boundary Conditions | Faster/Slower expansion | Static or reversed time-arrows |


