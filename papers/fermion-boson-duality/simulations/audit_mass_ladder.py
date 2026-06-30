"""
audit_mass_ladder.py
IAME Collaboration

FOUNDATION AUDIT: does the C_s formula, as exactly defined in Paper IX,
applied to the (n,m) equal-superposition states defined in Paper VIII,
actually reproduce the claimed lepton mass ladder (0, 8, 12 bits for
electron/muon/tau)?

This is a prerequisite check before using C_s as an energy variable
for E=mc^2. If the existing mass ladder claim doesn't survive contact
with the formula it's supposedly derived from, that needs to be flagged
honestly before building anything more on top of it.

The C_s formula (Paper IX, Definition):

    C_s(Psi) = C_base + sum_i [ C(phi_i) + C(A_i) + omega_i/Delta_omega ]

For an (n,m) equal superposition:
    psi_k = (1/sqrt(n)) * exp(i * 2*pi*m*k/n),  k = 0..n-1

This state, viewed as a function over the ring of n sites, has exactly
ONE nonzero discrete Fourier mode when decomposed in site-index space:
wavenumber = m. All n amplitudes have the same magnitude 1/sqrt(n) and
phases following the single winding m.

We compute C_s honestly under three different reasonable interpretations
of "the spectral decomposition of psi" and report what each gives,
rather than picking the one that happens to match the paper's claim.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ─────────────────────────────────────────────────────────────────────────────
# Exact (n,m) state construction (Paper VIII conventions, unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def make_psi(n, m):
    k = np.arange(n)
    phi = 2 * np.pi * m * k / n
    return np.exp(1j * phi) / np.sqrt(n)


# ─────────────────────────────────────────────────────────────────────────────
# Three honest interpretations of C_s for an (n,m) state
# ─────────────────────────────────────────────────────────────────────────────

def Cs_interpretation_A(n, m, C_base=0.0, C_phi=0.0, C_A=0.0):
    """
    INTERPRETATION A: dominant frequency = winding number m.
    Delta_omega is fixed at 1 (one mode = one unit cost), so the
    frequency term is just m/1 = m.

        C_s = C_base + C_phi + C_A + m

    Predicts: C_s depends on m only, NOT on n.
    """
    return C_base + C_phi + C_A + m


def Cs_interpretation_C(n, m, C_base=0.0, C_phi=0.0, C_A=0.0,
                          delta_omega_fixed=1.0):
    """
    INTERPRETATION C: Delta_omega is a UNIVERSAL fixed resolution
    (does not rescale with n), so larger n spans the SAME physical
    space with more sites, lowering the physical frequency per mode:

        omega_i = 2*pi*m / n
        Delta_omega = delta_omega_fixed
        omega_i / Delta_omega = 2*pi*m / (n * delta_omega_fixed)

    This makes C_s DECREASE with n for fixed m.
    """
    omega = 2 * np.pi * m / n
    return C_base + C_phi + C_A + omega / delta_omega_fixed


def Cs_per_site_amplitude_cost(n, m):
    """
    Candidate EXTRA n-dependent term: cost of specifying which n sites
    out of the universe's bit budget participate. log2(n) as a simple
    proxy for addressing overhead.
    """
    return np.log2(max(n, 1))


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: apply each interpretation to the actual classification table
# ─────────────────────────────────────────────────────────────────────────────

def test1_classification_table():
    print("=" * 78)
    print("TEST 1: C_s of the (n,m) classification table -- three interpretations")
    print("=" * 78)

    table = [
        (1, 0, "vacuum"),
        (2, 0, "neutrino"),
        (2, 1, "charged lepton (generation unspecified)"),
        (3, 0, "scalar sector"),
        (3, 1, "quark/gluon"),
        (4, 2, "spin-2 (open)"),
    ]

    print(f"\n  {'(n,m)':>8}  {'particle':>40}  "
          f"{'C_s(A): =m':>12}  {'C_s(C): 2pi*m/n':>16}  {'log2(n) site-cost':>18}")
    for n, m, label in table:
        Cs_A = Cs_interpretation_A(n, m)
        Cs_C = Cs_interpretation_C(n, m)
        site_cost = Cs_per_site_amplitude_cost(n, m)
        print(f"  ({n},{m})    {label:>40}  {Cs_A:>12.4f}  {Cs_C:>16.4f}  {site_cost:>18.4f}")

    print(f"\n  OBSERVATION:")
    print(f"  Under interpretation A (m as frequency), electron and neutrino")
    print(f"  both have n=2 -- only m differs (0 vs 1). C_s(A) for (2,0)=0,")
    print(f"  (2,1)=1. This distinguishes neutrino from electron by 1 unit,")
    print(f"  but says NOTHING about muon/tau, which Paper VIII claims are")
    print(f"  the SAME (n,m)=(2,1) class at higher 'complexity levels' --")
    print(f"  but (n,m)=(2,1) is a SINGLE point under this formula. There is")
    print(f"  no third index available in the formula to climb a ladder.")

    return table


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: is there a natural "generation" index that COULD produce 0/8/12?
# ─────────────────────────────────────────────────────────────────────────────

def test2_search_for_generation_index():
    print("\n" + "=" * 78)
    print("TEST 2: Is there a derivable generation index giving 0/8/12 bits?")
    print("=" * 78)

    n = 2
    print(f"\n  On an n={n} ring, checking windings m=1,3,5,7 (one candidate")
    print(f"  for a 'generation' index via higher harmonics):")

    for g in range(4):
        m_g = 1 + g * 2
        psi_g = make_psi(n, m_g)
        print(f"    g={g}, m={m_g}:  psi = {np.round(psi_g, 4)}")

    print(f"\n  CONCLUSION: there is no constructive procedure in Papers")
    print(f"  VII-IX that assigns a specific C_s cost of exactly 8 bits to")
    print(f"  'generation 2' and 12 bits to 'generation 3'. The 4-bit unit")
    print(f"  is introduced via a SEPARATE combinatorial argument (four")
    print(f"  binary choices per hop: site, frame, real/imaginary, sign)")
    print(f"  -- not a value computed by applying the C_s formula to a")
    print(f"  specific constructed state for muon or tau.")


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: reverse-engineer -- what WOULD need to be true for 0/8/12 to be
# a C_s calculation rather than a fit?
# ─────────────────────────────────────────────────────────────────────────────

def test3_reverse_engineer_4bit_claim():
    print("\n" + "=" * 78)
    print("TEST 3: Is the 4-bit/generation unit derived or fit?")
    print("=" * 78)

    masses_MeV = {"electron": 0.511, "muon": 105.66, "tau": 1776.86}
    m_e = masses_MeV["electron"]

    print(f"\n  Observed log2(m/m_e):")
    for name, mass in masses_MeV.items():
        log2_ratio = np.log2(mass / m_e)
        print(f"    {name:10s}  mass={mass:>10.3f} MeV   log2(m/m_e) = {log2_ratio:.4f}")

    spacing1 = np.log2(masses_MeV['muon']/m_e)
    spacing2 = np.log2(masses_MeV['tau']/m_e) - np.log2(masses_MeV['muon']/m_e)

    print(f"\n  The paper rounds these to 0, 8, 12 and calls the spacing")
    print(f"  '~4 bits per generation'. Actual spacings:")
    print(f"    electron -> muon: {spacing1:.4f} bits (rounded to 8, error {abs(spacing1-8):.2f})")
    print(f"    muon -> tau:      {spacing2:.4f} bits (rounded to 4, error {abs(spacing2-4):.2f})")
    print(f"\n  These spacings (7.69 and 4.07) are NOT equal to each other,")
    print(f"  and the first does not round cleanly to a multiple of the")
    print(f"  claimed 4-bit unit (7.69 / 4 = 1.92, not an integer).")
    print(f"  The '4-bit unit' and 'near-integer sequence' framing fits")
    print(f"  the OBSERVED masses to a hypothesized integer ladder after")
    print(f"  the fact. The combinatorial 'four binary choices per hop'")
    print(f"  argument is a SEPARATE, unconnected claim that happens to")
    print(f"  also yield the number 4 -- no derivation links the two.")

    print(f"\n  HONEST STATUS: the lepton mass ladder in Paper VIII is a")
    print(f"  numerological observation (masses are roughly log2-spaced)")
    print(f"  combined with a plausible but unconnected combinatorial")
    print(f"  story. It is not a first-principles C_s calculation that")
    print(f"  outputs 0, 8, 12 from the (n,m) classification alone.")

    return masses_MeV


# ─────────────────────────────────────────────────────────────────────────────
# Test 4: what about the (n,m) classification IS solid
# ─────────────────────────────────────────────────────────────────────────────

def test4_what_is_actually_solid():
    print("\n" + "=" * 78)
    print("TEST 4: What about the (n,m) classification IS solid")
    print("=" * 78)

    print(f"\n  CONFIRMED EXACT (machine precision, re-verified in prior session):")
    print(f"    ||F|| = 1/sqrt(n),  ||B|| = sqrt((n-1)/n),  ||F||^2+||B||^2 = 1")
    print(f"    Dominant Fourier mode of B (site-difference space) = m, not n")
    print(f"    ||B||^2 = (n-1)/n, independent of m")
    print(f"\n  These are unaffected by today's audit. They remain clean theorems.")

    print(f"\n  NOT CONFIRMED / NOT DERIVED (flagged by this audit):")
    print(f"    - The claim that C_s(n,m) produces a 'complexity ladder' with")
    print(f"      generations at +4 bits each. No formula in Papers VII-IX")
    print(f"      constructs this; it is asserted without a worked C_s")
    print(f"      calculation for muon or tau as distinct states.")
    print(f"    - (3,0) <-> Higgs sector: already labeled 'conjecture' in")
    print(f"      Paper VIII -- correctly hedged, no bug.")
    print(f"    - (4,2) <-> graviton: already labeled 'open' -- correctly")
    print(f"      hedged, no bug.")

    print(f"\n  RECOMMENDATION:")
    print(f"    Before using C_s as the energy variable for E=mc^2, either:")
    print(f"    (a) construct an explicit, parameter-free formula for C_s of")
    print(f"        a 'generation-g' state that outputs a number, and check")
    print(f"        it against 0/7.69/11.76 honestly (not pre-rounded), or")
    print(f"    (b) use a DIFFERENT, already-solid quantity as the energy")
    print(f"        variable -- ||B||^2 = (n-1)/n is fully proven, though it")
    print(f"        also can't distinguish electron from neutrino mass")
    print(f"        (both have n=2), so it has the same generation problem.")


# ─────────────────────────────────────────────────────────────────────────────
# Figure
# ─────────────────────────────────────────────────────────────────────────────

def make_figure(masses_MeV):
    fig = plt.figure(figsize=(13, 7))
    fig.patch.set_facecolor('#0d1117')
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

    ACCENT  = '#58a6ff'
    ACCENT2 = '#f78166'
    GRID    = '#21262d'
    TEXT    = '#c9d1d9'
    BG      = '#161b22'

    def styled_ax(ax, title):
        ax.set_facecolor(BG)
        for spine in ax.spines.values():
            spine.set_color(GRID)
        ax.tick_params(colors=TEXT, labelsize=8)
        ax.xaxis.label.set_color(TEXT)
        ax.yaxis.label.set_color(TEXT)
        ax.set_title(title, color=TEXT, fontsize=9.5, pad=8)
        ax.grid(True, color=GRID, linewidth=0.5, alpha=0.7)

    ax1 = fig.add_subplot(gs[0])
    m_e = masses_MeV["electron"]
    names = list(masses_MeV.keys())
    actual = [np.log2(masses_MeV[n] / m_e) for n in names]
    claimed = [0, 8, 12]

    x = np.arange(len(names))
    width = 0.35
    ax1.bar(x - width/2, actual, width, color=ACCENT, label='actual log2(m/m_e)')
    ax1.bar(x + width/2, claimed, width, color=ACCENT2, alpha=0.85,
            label='claimed integer ladder')
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, color=TEXT)
    ax1.set_ylabel('log2(mass / m_electron)')
    ax1.legend(fontsize=8, labelcolor=TEXT, facecolor=BG, edgecolor=GRID)
    styled_ax(ax1, 'Actual vs claimed mass ladder spacing')

    for i, (a, c) in enumerate(zip(actual, claimed)):
        ax1.text(i - width/2, a + 0.3, f'{a:.2f}', ha='center', color=TEXT, fontsize=7)
        ax1.text(i + width/2, c + 0.3, f'{c}', ha='center', color=TEXT, fontsize=7)

    ax2 = fig.add_subplot(gs[1])
    ax2.axis('off')
    text = (
        "FOUNDATION AUDIT SUMMARY\n\n"
        "SOLID (machine-verified, unaffected):\n"
        "  ||F|| = 1/\u221an\n"
        "  ||B|| = \u221a((n-1)/n)\n"
        "  ||F||\u00b2 + ||B||\u00b2 = 1\n"
        "  dominant mode of B = m (not n)\n\n"
        "NOT DERIVED (flagged today):\n"
        "  C_s(n,m) generation ladder\n"
        "  (0, 8, 12 bit spacing)\n\n"
        "  Actual spacings: 7.69, 4.07 bits\n"
        "  -- not equal, not exactly matching\n"
        "  the claimed '4 bits per generation'\n\n"
        "  The combinatorial '4 binary choices\n"
        "  per hop' story is UNCONNECTED to\n"
        "  any worked C_s formula output.\n\n"
        "RECOMMENDATION: don't build E=mc\u00b2\n"
        "on the generation ladder. Use the\n"
        "proven ||B||\u00b2 = (n-1)/n instead, or\n"
        "derive C_s(generation) explicitly first."
    )
    ax2.text(0.02, 0.98, text, transform=ax2.transAxes, color=TEXT,
             fontsize=9, va='top', ha='left', family='monospace')
    ax2.set_facecolor(BG)
    for spine in ax2.spines.values():
        spine.set_color(GRID)

    fig.text(0.5, 0.97, 'IAME — Foundation Audit: Mass Ladder Claim vs C_s Formula',
             ha='center', va='top', color=TEXT, fontsize=11.5, fontweight='bold')

    plt.savefig('../figures/audit_mass_ladder.png',
                dpi=160, bbox_inches='tight', facecolor=fig.get_facecolor())
    print("\n  Figure saved.")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("\nIAME Collaboration — FOUNDATION AUDIT")
    print("Question: does the C_s formula actually produce the claimed mass ladder?\n")

    test1_classification_table()
    test2_search_for_generation_index()
    masses_MeV = test3_reverse_engineer_4bit_claim()
    test4_what_is_actually_solid()

    print("\n" + "=" * 78)
    print("AUDIT CONCLUSION")
    print("=" * 78)
    print("""
  The (n,m) classification theorems (||F||, ||B||, conservation law,
  dominant-mode-equals-m) are exact and remain solid.

  The lepton MASS LADDER claim (0/8/12 bits, "4-bit generations") is
  NOT a derived consequence of applying the C_s formula to (n,m)
  states. It is:
    (a) an empirical rounding of the actual log2 mass ratios
        (0, 7.69, 11.76 -> rounded to 0, 8, 12), combined with
    (b) a separate, unconnected combinatorial argument ("4 binary
        choices per hop") that produces the number 4 independently.

  No construction in Papers VII-IX computes C_s for "generation 2"
  or "generation 3" of the (2,1) class and gets 8 or 12 as output.
  There is currently no third parameter in the (n,m) framework that
  a generation index could even attach to.

  THIS IS A REAL GAP, not a minor rounding issue. It means: do not
  use the generation ladder as a foundation for the E=mc^2 derivation.
  The safe foundation to build on is the proven conservation law and
  ||B||^2 = (n-1)/n, with the explicit caveat (already known from the
  prior session) that ||B||^2 alone cannot distinguish electron mass
  from neutrino mass, let alone the three lepton generations.

  This means the C_s -> E=mc^2 path needs either:
    1. A genuine derivation of the generation index's C_s cost, or
    2. Abandoning the generation ladder and finding a different
       bridge from (n,m) classification to a continuous mass variable.
    """)

    make_figure(masses_MeV)
