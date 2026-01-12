# The Abstract Universe Project (AUP)

The **Abstract Universe Project (AUP)** is an ongoing research and simulation framework exploring a unified information-theoretic model of reality. It develops a formal theory of everything (ToE) - a theory in which physical universe emerges from noise as the most compressible and therefore most probable and predictable outcome. The project encompasses both rigorous mathematical exposition and computational simulations that illustrate and validate key theoretical claims.


## Contents

* [Pages](https://juhameskanen.github.io/abstract/) – HTML and PDF versions of all papers
* [Papers](https://github.com/juhameskanen/abstract) – LaTeX and python simulation source code for all papers
* [CHANGELOG ➡](CHANGELOG.md) – Record of updates, improvements, and new simulations


## Purpose

AUP aims to:

1. Formulate a **model of everything**
2. Demonstrate, via simulations, how only two minimal principles give rise to physics.
3. Provide a **rigorous, testable framework** for a Theory of Everything (TOE) that unifies gravity and quantum mechanics as manifestations of information structures

This is a research-in-progress project. Many simulations and book chapters are in draft form and will be refined over time.


## Continuous Integration

AUP uses GitHub Actions for automated builds:

* [![Build and Upload PDFs](https://github.com/juhameskanen/abstract/actions/workflows/build-pdf.yml/badge.svg)](https://github.com/juhameskanen/abstract/actions/workflows/build-pdf.yml) – Builds and uploads the latest PDF versions of the papers
* [![Build and Deploy HTML (GitHub Pages)](https://github.com/juhameskanen/abstract/actions/workflows/build-pages.yml/badge.svg)](https://github.com/juhameskanen/abstract/actions/workflows/build-pages.yml) – Generates and deploys HTML versions of papers to GitHub Pages


## Wiki

The [project wiki](https://github.com/juhameskanen/abstract/wiki) contains:

* Detailed descriptions of simulations
* Implementation notes and algorithms
* Research notes and derivations
* Discussion of ongoing and planned chapters


## Structure

```
.
├── papers/            # LaTeX source and simulations
├── CHANGELOG.md       # Version history
├── README.md          # Project overview (this file)
└── LICENSE.md         # Licensing information
```


## Getting Started

1. **Clone the repository**

   ```bash
   git clone https://github.com/juhameskanen/abstract.git
   cd abstract
   ```

2. **Install dependencies**
   Most simulations require Python ≥ 3.11 with `numpy`, `matplotlib`, and `networkx`. Install via:

   ```bash
   pip install -r requirements.txt
   ```

3. **Run a simulation**

   ```bash
   python simulations/entropy_simulation.py
   ```

4. **Build papers**

   ```bash
   make
   ```

> Note: Many simulations and chapters are in draft stage; behavior may be updated frequently.


## Contributing

Contributions are welcome, especially:

* Additional simulations demonstrating theoretical claims
* Corrections, clarifications, or expansions of draft chapters
* Documentation improvements

Please follow standard GitHub workflow: fork → branch → pull request. Ensure all code additions are well-documented.


## License

*© 2001–2025 The Abstract Universe Project. All rights reserved.*
All project content—including papers, simulations, and documentation—is proprietary. Usage for academic study and non-commercial research is permitted with proper citation.

---

*© 2001 ... 2025 The Abstract Universe Project. All rights reserved.*

