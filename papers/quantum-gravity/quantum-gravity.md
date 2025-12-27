\documentclass[11pt]{article}

% --- Packages ---
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{amsmath, amssymb, amsthm}
\usepackage{geometry}
\usepackage{graphicx}
\usepackage{hyperref}

% --- Page Setup ---
\geometry{letterpaper, margin=1in}
\setlength{\parindent}{0pt}
\setlength{\parskip}{1em}

% --- Document Metadata ---
\title{\textbf{Toy Model of Quantum-Gravity}}
\author{Juha Meskanen}
\date{January 2019}

\begin{document}

\maketitle

\section*{Abstract}
This paper introduces an informational theory of quantum gravity, proposing that the fundamental laws of physics are not pre-existing but rather emergent properties of a maximally compressed informational universe.
In this view, observers are inherently ``compressed'' information structures, and the smooth, predictable physical laws we experience are a direct consequence of this compression.
To model this, we present a simulation where observers are represented by Gaussian blobs, compressed with complex wavefunctions.
Particles, generated from a probabilistic field, are governed by a deterministic motion derived from probability amplitudes.
The simulation demonstrates that inertia---a cornerstone of classical mechanics---naturally emerges from a selection process that favors the most ``compressible'' and hence the most probable path.

\section{Introduction}
The search for a unified theory of everything has been a central pursuit in modern physics. However, traditional approaches often begin with a set of assumed fundamental laws. This paper proposes an alternative framework where the laws themselves are not fundamental but are emergent from a more primitive, informational substrate.

The key idea is that existence itself is biased toward compression. Observers are not passive; their probability of existing is directly tied to how efficiently their informational structure can be described.

\subsection*{The Compression–Existence Principle}
\begin{itemize}
    \item \textbf{Compression increases multiplicity:} An observer that can be described with fewer bits can be described in vastly more ways within the underlying informational substrate.
    \item \textbf{Multiplicity increases probability:} The more copies of an observer exist, the higher the probability that any given observer will be one of them.
    \item \textbf{Probability selects predictable worlds:} Observers therefore overwhelmingly find themselves in worlds where their own informational structure is maximally compressible.
    \item \textbf{Predictability manifests as physical law:} Smooth, regular, law-like behavior is the kind of information that compresses best. 
\end{itemize}

\section{Theoretical Framework}

\subsection{Wavefunctions as Observer Templates}
An observer is defined as a compressed informational structure. In our 2D model, this structure is a parametric complex wavefunction, $\psi(x,t)$, representing a soft disk or Gaussian shell:
\[
\psi(x,t) = A \exp\big(-2\sigma^2 (r - R)^2\big)\,\exp\big(-i(\omega t - \phi)\big)
\]
Here, $r = \|x - c\|$ is the radial distance from the center $c$, $A$ is the amplitude, $R$ is the radius of the shell, $\sigma$ is its width, $\omega$ is its frequency, and $\phi$ is its phase.

\subsection{Observer Filtering}
An observer emerges from the underlying random noise field by applying a filter. This is modeled by a Gaussian observer window, $O_j(x)$, which smoothly weights the influence of particles based on their proximity to the observer’s center:
\[
O_j(x) = \exp\big(-2\sigma_{\text{obs}}^2 \|x - c_j\|^2\big)
\]

\subsection{Probabilistic Field and Particle Generation}
The random noise is represented by a large number of particles resampled from a total probability density function (PDF), generated from the interference pattern of all observer wavefunctions:
\[
P(x) = \big|\Psi_{\text{total}}(x,t)\big|^2 = \left|\sum_j \psi_j(x,t)\right|^2
\]

\subsection{Compression and Emergent Inertia}
The central prediction is the generation of deterministic motion through a selection process guided by informational compressibility.

\subsubsection*{Soft Assignment}
Particles are ``softly'' assigned to observers using weights proportional to both the filter function and the local probability density:
\[
W_j(x) \propto O_j(x)\cdot |\psi_j(x)|^2
\]

\subsubsection*{Compressibility Cost}
The choice among candidates is determined by an informational cost function derived from the \textbf{principle of minimal description length (MDL)}. We approximate the change in description length using the \textbf{incremental Shannon entropy} of the observer’s trajectory:
\[
\text{Cost}_j(\Delta v) = H(\Delta v)
\]
In practice, this is approximated by quadratic penalties on phase and frequency variation:
\[
\text{Cost}_j(\Delta v) \approx \Delta \phi^2 + \Delta f^2
\]

\section{Simulation Model}
The simulation is implemented in Python and consists of several key classes:
\begin{itemize}
    \item \textbf{Wavefunction}: Represents the parametric Gaussian shell.
    \item \textbf{ObserverWindow}: A Gaussian function used to softly assign particles.
    \item \textbf{WavefunctionGravitySim}: Orchestrates the process, computes total PDF, and chooses next positions based on compressibility cost.
\end{itemize}

\section{Results and Discussion}
The simulation successfully demonstrates the emergence of inertial, law-like behavior. Observers begin to move in smooth, predictable paths as a direct consequence of the system's preference for compressibility.



\begin{figure}[h]
    \centering
    \includegraphics[width=0.8\textwidth]{figures/quantum_gravity.gif}
    \caption{Emergent gravity and Quantum Mechanics.}
    \label{fig:quantum-gravity}
\end{figure}

\section{Conclusion}
This work supports the Compression–Existence Principle: the laws of physics are not axiomatic features of reality, but emergent regularities selected because only such compressed observers dominate the measure of existence.

\section*{Appendix: Simulation Code}
The simulation code is available in \texttt{simulations/quantum\_gravity.py}.

\end{document}
