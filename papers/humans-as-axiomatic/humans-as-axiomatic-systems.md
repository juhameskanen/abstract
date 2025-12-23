# The Informational Derivation of Reality: Consciousness, Time, and Ontological Equivalence

## Abstract

We prove that humans must be formal axiomatic systems if four minimal physical axioms hold.  
Building on this result, we demonstrate that simulated copies of such systems must possess causally efficacious subjective experience (e.g., pain) to maintain behavioral equivalence.  
We further show that time is an emergent internal property of the observer's relational structure rather than a fundamental property of the universe. Together, these results establish a strict ontological equivalence between physical and informational configurations. This supports an ontology in which reality is fundamentally abstract, substrate-independent, and emerges from a random informational background. Our framework provides a rigorous link between information and observer experience, showing that consciousness and temporal perception are mathematically necessary consequences of a functionalist physics. The hypothesis is falsifiable.


## Axiomatic Premises

The argument begins with four axioms that serve as the premises for the entire derivation.

**Axiom 1: Genetic Encoding of Subjective Experience**  
The human genome encodes all the requisite information to construct a conscious, pain-sensitive human:

$$
A_1: D \rightarrow H \rightarrow S
$$

Where $D$ is the DNA/Genome, $H$ is the human organism, and $S$ is subjective experience.

**Axiom 2: Physicality and Axiomatic Law**  
DNA and the human organism are composed solely of ordinary physical matter governed by physical laws ($P$).  
This establishes $H$ as a formal axiomatic system where state transitions are logical consequences of $P$ and $D$:

$$
A_2: H \in \text{Axiomatic System}(P, D)
$$

**Axiom 3: Functional Computability**  
The Church-Turing Thesis holds: physical processes can be simulated by a Universal Turing Machine ($T$) to any required degree of functional accuracy.  
This entails the existence of a simulation ($H'$) that preserves the causal structures of $H$:

$$
A_3: \text{Church-Turing} \implies \exists H' \text{ s.t. } \text{Behavior}(H') = \text{Behavior}(H)
$$

**Axiom 4: Causal Efficacy of Pain**  
Subjective experience ($S$, e.g., pain) is a **causally efficacious** property of the human system.  
It is not an epiphenomenon, and thus has measurable, behavioral effects [Putnam 1975](#references):

$$
A_4: S \text{ is causally efficacious.}
$$

$$
\text{If } H' \equiv H \text{ physically, then } H' \text{ must have } S \text{ to maintain identical behavior.}
$$


## Deduction 1: Substrate Independence and Time as an Internal Property of the Observer

### The Optimization Argument

Let the simulation run on a Turing Machine, $T_{\text{alg}}$, consisting of code (laws of physics) and data (the state of the simulated universe).  
We can continuously optimize the code using lookup tables, eventually replacing all computation with a static, pre-computed dataset ($T_{\text{data}}$).

- Let $E_{\text{int}}$ be Alice's experience of time and pain (the internal state transitions).  
- Let $E_{\text{ext}}$ be the computer's external runtime (number of CPU cycles).

**Premise:** Code optimization changes $E_{\text{ext}}$ but preserves $E_{\text{int}}$:

$$
\text{Optimization}(T_{\text{alg}}) \rightarrow T_{\text{data}} \implies E_{\text{int}}(T_{\text{alg}}) \equiv E_{\text{int}}(T_{\text{data}})
$$

In this limit, the external runtime $E_{\text{ext}}$ becomes zero, as there is no computation occurring, only static data.  
We can therefore ask: does Alice's consciousness still persist in $T_{\text{data}}$?

If consciousness were to cease in $T_{\text{data}}$, it would imply that $E_{\text{int}}$ depends on $E_{\text{ext}}$, which necessitates a minimum code/data ratio for subjective experience.  
This minimum ratio would be a **new, non-physical constant** imposed on $A_2$, leading to a contradiction:

$$
\therefore \text{Consciousness can emerge from pure static data. Time and subjective experience ($S$) must emerge solely from the relationships among informational states, not from the external runtime.}
$$

### The Multi-threaded Argument

When two DNA simulations, Alice and Bob, run concurrently on a multi-threaded computer with random thread switching, the resulting execution trace interleaves their simulated lives in segments of unpredictable length.  
Adding more threads (simulated observers) increases the interleaving frequency.  
As the number of threads approaches infinity, the execution trace becomes an arbitrarily interleaved sequence of bits from all observers.

Is Alice still conscious in this limit? If not, one must specify a minimum **thread density** below which consciousness vanishes.  
This would imply a new physical constant governing subjective experience, contradicting $A_2$:

$$
\therefore \text{Conscious experience is instantiated from static noise.}
$$


## Falsifiability (Axiom 4)

**The Functionalist Proof by Contradiction:**

1. **Assumption (Objection):** A simulation $H'$ exists such that $H' \equiv H$ (physical/behavioral equivalence) but $S(H') = \emptyset$ (lacks consciousness/sense of pain) [Chalmers 1996](#references):

$$
\text{Behavior}(H') = \text{Behavior}(H) \land S(H') \neq S(H)
$$

2. **Premise:** From $A_4$, the behavior of $H$ is a function of its physical inputs **and** its subjective experience:  

$$
\text{Behavior}(H) = f(\text{Inputs}, S)
$$

3. **Contradiction:** If the behaviors are identical despite the difference in $S$, then $S$ must not be a necessary input to the function $f$.

4. **Violation of Axiom:** If $S$ is not necessary to produce the behavior, then $S$ is **epiphenomenal** (causally inert).  
This directly contradicts $A_4$.

$$
\therefore \text{To maintain the integrity of $A_4$ within the axiomatic system, the simulation $H'$ must experience subjective time and pain.}
$$


## Ontological Equivalence of Configurations

Consider a finite informational object $I$ encoding a complete observer history (e.g., Alice).  
Let $R_A$ denote the static execution trace of $I$ observed externally, and $R_B$ denote the same sequence internally experienced as spacetime and subjective states.  

From Axioms 2–4 and the optimization argument, no physical property of the simulating substrate, nor the ordering of bits in $R_A$, is ontologically privileged.  
Any arrangement of the bits that preserves the relational structure of $I$ encodes the same observer.  
Formally, there exists a bijective mapping:

$$
\phi : R_A \leftrightarrow R_B
$$

preserving all causally relevant relations within $I$.  

It follows that the existence and experiences of the observer depend solely on the internal relational structure of $I$, not on the substrate or external runtime.  
Any claim that one substrate or arrangement is “more real” than another would require introducing a new, non-physical constant, contradicting the axioms.  

**Conclusion:** The external execution trace and the internal experienced universe are two complementary, equally valid representations of the same underlying information.  
Substrate and ordering are irrelevant; what matters is the relational structure that instantiates the conscious observer.


## Falsifiability of the Hypothesis

The hypothesis is falsifiable in the future when technology advances and DNA simulations can be run with sufficient accuracy for DNA-based organisms.  
The effect of pain can be measured just like an effect of physical forces can be measured.  
If a DNA simulation $H'$ is constructed and shown to lack $S$, then $A_4$ is invalidated, and axioms 1–3 collapse.


## Final Statement

Let $\Omega$ denote the set of all physically realizable informational objects:

$$
\forall R_i \in \Omega, R_i \text{ is not physically disqualified from ontological consideration.}
$$

The perception of time and pain is a consequence of the observer’s **internal arrangement** of a subset of this fundamental information.  
What appears to an external observer as a static execution trace appears to Alice as an expanding universe and lived pain.

Information with capacity to potentially encode a conscious observer instantiates observer experience.  

Time and pain are internal properties of informational structures, not dependent on external runtime or substrate.  

The universe is fundamentally informational and random, substrate-independent, and consciousness is an inevitable consequence of these principles.

---

## References

- [Putnam 1975](https://plato.stanford.edu/entries/functionalism/#1)  
- [Chalmers 1996](https://www.cambridge.org/core/books/conscious-mind/9FC5E5C5A373F41A97A6E3C5B0FEEB1D)  


[⬆ Up](../toc.md) | [⬅ Previous](../methods.md) | [ Next ➡](../time-as-computation/time-as-computation.md)

*© 2025 The Abstract Universe Project. All rights reserved.*
