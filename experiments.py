"""
Companion experiments for the LinkedIn article
"How many qubits does it take to simulate a holographic universe?"

Everything here runs on Qiskit 2.4.x (tested with 2.4.2), NumPy and Matplotlib.

Sections
  A. Resource estimate from Maldacena, arXiv:2303.11534, Sec. 3.1 (rough formula)
  B. Jordan-Wigner Majorana encoding: 2n Majoranas <-> n qubits (verified)
  C. Trotter-cost proxy: quartic Majorana Hamiltonian, term and CX scaling
  D. Page curve: entanglement entropy of Haar-random states, with statistics
  E. Scrambling: entanglement growth in a brickwork of Haar-random 2-qubit gates
  F. Bosonic truncation: qubits per bosonic mode vs ground-state error
"""
import itertools
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import qiskit
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import (SparsePauliOp, Statevector, entropy,
                                 partial_trace, random_statevector,
                                 random_unitary)
from qiskit.synthesis import LieTrotter

assert qiskit.__version__.startswith("2.4"), qiskit.__version__

OUT = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(OUT, "fig")
os.makedirs(FIG, exist_ok=True)

# ---- plotting style (light surface, validated default palette, slots 1-3) ----
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "axes.edgecolor": AXIS,
    "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "font.family": "DejaVu Sans", "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "legend.frameon": False,
})


def finish(ax, title, subtitle=None):
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold", color=INK,
                 pad=22 if subtitle else 10)
    if subtitle:
        ax.text(0, 1.03, subtitle, transform=ax.transAxes, fontsize=10,
                color=INK2, va="bottom")


def save(fig, name):
    path = os.path.join(FIG, name)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return os.path.relpath(path, OUT)      # relative, so results.json is portable


results = {"qiskit_version": qiskit.__version__}

# =============================================================================
# A. Resource estimate (rough formula quoted from the paper, Sec. 3.1)
# =============================================================================
def n_qubits(N, lam):
    """n_q ~ N^2 [ 8 + 9 log2(lam) ],  lam = lambda^(1/3) / omega."""
    return N**2 * (8 + 9 * np.log2(lam))


def n_terms(N):
    """Hamiltonian terms: quartic bosonic 9^2 N^4 + cubic Yukawa 16^2 * 9 * N^3."""
    return 9**2 * N**4 + 16**2 * 9 * N**3


def rough_gates(N, lam):
    """Order-of-magnitude gate count: terms x log2(16 N^2) x lam (all O(1) dropped)."""
    return n_terms(N) * np.log2(16 * N**2) * lam


N0, L0 = 16, 4
results["A"] = {
    "N": N0, "lambda13_over_omega": L0,
    "fermion_qubits": 8 * N0**2,
    "boson_qubits": 9 * N0**2 * int(np.log2(L0)),
    "total_qubits": float(n_qubits(N0, L0)),
    "hamiltonian_terms": float(n_terms(N0)),
    "rough_gate_count": float(rough_gates(N0, L0)),
    # Sensitivity, not a claim about the paper: if levels 0..Lambda must all be
    # representable, ceil(log2(Lambda+1)) qubits per mode are needed, not log2(Lambda).
    "total_qubits_if_levels_0_to_lambda": float(N0**2 * (8 + 9 * np.ceil(np.log2(L0 + 1)))),
}

Ns = np.arange(2, 25)
fig, ax = plt.subplots(figsize=(8, 4.8))
for lam, col in [(2, AQUA), (4, BLUE), (8, ORANGE)]:
    ax.plot(Ns, n_qubits(Ns, lam), color=col, lw=2, label=f"λ¹ᐟ³/ω = {lam}")
    ax.text(Ns[-1] + 0.3, n_qubits(Ns[-1], lam), f"{lam}", color=INK2,
            va="center", fontsize=10)
ax.scatter([N0], [n_qubits(N0, L0)], s=70, color=BLUE, edgecolor=SURFACE,
           linewidth=2, zorder=5)
ax.annotate(f"N = 16, λ¹ᐟ³/ω = 4\n≈ {n_qubits(N0, L0):,.0f} qubits",
            xy=(N0, n_qubits(N0, L0)), xytext=(4.5, 14500), color=INK,
            arrowprops=dict(arrowstyle="-", color=MUTED), fontsize=10)
ax.set_xlabel("Matrix size N (SU(N) gauge group)")
ax.set_ylabel("Logical qubits  n_q")
ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.62))
ax.set_xlim(2, 25.5)
finish(ax, "Qubit count grows as N², and only logarithmically with the cutoff",
       "n_q ≈ N² [ 8 + 9 log₂(λ¹ᐟ³/ω) ]   (rough estimate, Maldacena 2023, Sec. 3.1)")
results["A"]["fig"] = save(fig, "fig1_resource_estimate.png")

# =============================================================================
# B. Jordan-Wigner Majoranas on n qubits
# =============================================================================
def pauli_label(n, ops):
    s = ["I"] * n
    for q, p in ops.items():
        s[n - 1 - q] = p          # Qiskit labels are little-endian
    return "".join(s)


def majorana(n, k):
    """gamma_k on n qubits, k = 0..2n-1 (Jordan-Wigner)."""
    j, r = divmod(k, 2)
    ops = {q: "Z" for q in range(j)}
    ops[j] = "X" if r == 0 else "Y"
    return SparsePauliOp(pauli_label(n, ops))


n_chk = 4
gam = [majorana(n_chk, k) for k in range(2 * n_chk)]
dim = 2**n_chk
worst = 0.0
for i, j in itertools.product(range(2 * n_chk), repeat=2):
    anti = (gam[i].dot(gam[j]) + gam[j].dot(gam[i])).to_matrix()
    target = 2 * (i == j) * np.eye(dim)
    worst = max(worst, np.abs(anti - target).max())
results["B"] = {"n_qubits": n_chk, "n_majoranas": 2 * n_chk,
                "max_deviation_from_clifford_algebra": float(worst)}
assert worst < 1e-12

# =============================================================================
# C. Trotter-cost proxy: quartic Majorana Hamiltonian (NOT the paper's model)
# =============================================================================
rng = np.random.default_rng(2611)


def quartic_majorana_H(n):
    nm = 2 * n
    g = [majorana(n, k) for k in range(nm)]
    terms = []
    for idx in itertools.combinations(range(nm), 4):
        op = g[idx[0]].dot(g[idx[1]]).dot(g[idx[2]]).dot(g[idx[3]])
        assert np.abs(op.coeffs.imag).max() < 1e-12   # Hermitian product
        terms.append(rng.normal() * SparsePauliOp(op.paulis, op.coeffs.real))
    return sum(terms[1:], terms[0]).simplify()


rows = []
for n in range(2, 7):
    H = quartic_majorana_H(n)
    qc = QuantumCircuit(n)
    qc.append(PauliEvolutionGate(H, time=0.1, synthesis=LieTrotter(reps=1)),
              range(n))
    tq = transpile(qc, basis_gates=["cx", "rz", "sx", "x"],
                   optimization_level=1, seed_transpiler=7)
    ops = tq.count_ops()
    rows.append({"majoranas": 2 * n, "qubits": n,
                 "pauli_terms": len(H), "cx": int(ops.get("cx", 0)),
                 "depth": tq.depth()})
results["C"] = {"rows": rows}
# With only five tiny sizes a log-log exponent would be meaningless; report the
# CX cost per Pauli term instead (Jordan-Wigner strings make it grow with n).
for r in rows:
    r["cx_per_term"] = r["cx"] / r["pauli_terms"]
    assert r["pauli_terms"] == len(list(itertools.combinations(range(r["majoranas"]), 4)))

fig, ax = plt.subplots(figsize=(8, 4.6))
mj = [r["majoranas"] for r in rows]
ax.plot(mj, [r["pauli_terms"] for r in rows], "-o", color=BLUE, lw=2, ms=7,
        mec=SURFACE, mew=2, label="Pauli terms in H")
ax.plot(mj, [r["cx"] for r in rows], "-o", color=ORANGE, lw=2, ms=7,
        mec=SURFACE, mew=2, label="CX gates, one Trotter step")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xticks(mj); ax.set_xticklabels(mj)
ax.set_xlabel("Number of Majorana modes (2 per qubit)")
ax.set_ylabel("Count")
ax.legend(loc="upper left")
finish(ax, "Gate cost tracks the number of Pauli terms in the Hamiltonian",
       "Quartic Majorana toy Hamiltonian, Lie–Trotter, 1 step, transpiled to {CX, RZ, SX, X}")
results["C"]["fig"] = save(fig, "fig5_trotter_proxy.png")

# One Majorana quartic term on 4 qubits: the building block of the step above
n4, best = 4, None
g4 = [majorana(n4, k) for k in range(2 * n4)]
for idx in itertools.combinations(range(2 * n4), 4):
    op = g4[idx[0]].dot(g4[idx[1]]).dot(g4[idx[2]]).dot(g4[idx[3]])
    lab = op.paulis[0].to_label()
    w = sum(c != "I" for c in lab)
    if best is None or w > best[0]:
        best = (w, idx, lab)
w, idx, lab = best
term = QuantumCircuit(n4)
term.append(PauliEvolutionGate(SparsePauliOp(lab), time=0.2), range(n4))
term_t = transpile(term, basis_gates=["cx", "h", "rz", "s", "sdg"],
                   optimization_level=0)
fig = term_t.draw("mpl", fold=-1, scale=0.9)
fig.suptitle(f"exp(−i·0.2·{lab}) — Pauli image of γ{idx[0]}γ{idx[1]}γ{idx[2]}γ{idx[3]} "
             f"(Jordan–Wigner, 4 qubits)", x=0.02, ha="left", fontsize=11)
results["C"]["term_circuit"] = {"majorana_indices": list(idx), "pauli": lab,
                                "ops": dict(term_t.count_ops())}
results["C"]["term_fig"] = save(fig, "fig4a_majorana_term_circuit.png")

# =============================================================================
# D. Page curve: Haar-random states, statistics over samples
# =============================================================================
def ent_bits(vec, m, n):
    """Entropy (bits) of qubits 0..m-1 for pure state `vec`, via SVD."""
    s = np.linalg.svd(vec.reshape(2 ** (n - m), 2 ** m), compute_uv=False)
    p = s**2
    p = p[p > 1e-14]
    return float(-(p * np.log2(p)).sum())


def page_bits(dA, dB):
    """Exact Page average entanglement entropy (bits)."""
    m, n = sorted((dA, dB))
    k = np.arange(n + 1, m * n + 1)
    return float((np.sum(1.0 / k) - (m - 1) / (2 * n)) / np.log(2))


# Cross-check the SVD route against Qiskit's partial_trace + entropy
chk = []
for s in range(5):
    sv = random_statevector(2**8, seed=100 + s)
    for m in (2, 4, 6):
        a = ent_bits(np.asarray(sv.data), m, 8)
        b = float(entropy(partial_trace(sv, list(range(m, 8))), base=2))
        chk.append(abs(a - b))
results["D_crosscheck_max_abs_diff"] = float(max(chk))
assert max(chk) < 1e-9

n, S = 12, 400
ms = np.arange(1, n)
E = np.empty((S, len(ms)))
for i in range(S):
    v = np.asarray(random_statevector(2**n, seed=5000 + i).data)
    for j, m in enumerate(ms):
        E[i, j] = ent_bits(v, int(m), n)
mean, std = E.mean(0), E.std(0, ddof=1)
sem = std / np.sqrt(S)
exact = np.array([page_bits(2**int(m), 2**(n - int(m))) for m in ms])
results["D"] = {
    "n_qubits": n, "samples": S,
    "m": ms.tolist(), "mean_bits": mean.tolist(), "std_bits": std.tolist(),
    "ci95_halfwidth_bits": (1.96 * sem).tolist(), "page_exact_bits": exact.tolist(),
    "max_abs_mean_minus_page": float(np.abs(mean - exact).max()),
    "deficit_at_half_bits": float(n / 2 - mean[n // 2 - 1]),
    "deficit_at_half_page_theory_bits": float(n / 2 - exact[n // 2 - 1]),
}

fig, ax = plt.subplots(figsize=(8, 4.8))
ax.plot(ms, np.minimum(ms, n - ms), color=MUTED, lw=1.2, ls=":",
        label="max entropy min(m, n−m)")
ax.fill_between(ms, mean - std, mean + std, color=BLUE, alpha=0.18, lw=0)
ax.plot(ms, mean, "-o", color=BLUE, lw=2, ms=6, mec=SURFACE, mew=1.5,
        label=f"mean over {S} Haar states (±1σ)")
ax.plot(ms, exact, "--", color=ORANGE, lw=2, label="exact Page formula")
ax.set_xlabel("Size of subsystem m (qubits), total n = 12")
ax.set_ylabel("Entanglement entropy S (bits)")
ax.legend(loc="lower center")
finish(ax, "Page curve for a random pure state of 12 qubits",
       "Entropy rises, peaks at m = n/2 and falls: the pure global state is recovered")
results["D"]["fig"] = save(fig, "fig2_page_curve.png")

# =============================================================================
# E. Scrambling: brickwork of Haar-random 2-qubit gates
# =============================================================================
nq, cut, layers, S2 = 8, 4, 14, 200
ent = np.zeros((S2, layers + 1))
for i in range(S2):
    sv = Statevector.from_label("0" * nq)
    ent[i, 0] = ent_bits(np.asarray(sv.data), cut, nq)
    for L in range(1, layers + 1):
        start = 0 if L % 2 == 1 else 1
        for q in range(start, nq - 1, 2):
            sv = sv.evolve(random_unitary(4, seed=int(rng.integers(2**31))), [q, q + 1])
        ent[i, L] = ent_bits(np.asarray(sv.data), cut, nq)
m2, s2 = ent.mean(0), ent.std(0, ddof=1)
page8 = page_bits(2**cut, 2**(nq - cut))
sat = int(np.argmax(m2 >= 0.95 * page8))
results["E"] = {"n_qubits": nq, "cut": cut, "samples": S2,
                "mean_bits": m2.tolist(), "std_bits": s2.tolist(),
                "page_bits": page8, "layers_to_95pct_of_page": sat,
                "max_possible_bits": cut}

fig, ax = plt.subplots(figsize=(8, 4.6))
lay = np.arange(layers + 1)
ax.axhline(page8, color=ORANGE, ls="--", lw=2, label=f"Page value ({page8:.2f} bits)")
ax.fill_between(lay, m2 - s2, m2 + s2, color=BLUE, alpha=0.18, lw=0)
ax.plot(lay, m2, "-o", color=BLUE, lw=2, ms=6, mec=SURFACE, mew=1.5,
        label=f"mean over {S2} circuits (±1σ)")
ax.set_xlabel("Brickwork layers")
ax.set_ylabel("Entropy across the middle cut (bits)")
ax.set_ylim(0, cut + 0.2)
ax.legend(loc="lower right")
finish(ax, "Entanglement across a cut saturates near the Page value",
       "8 qubits, Haar-random 2-qubit gates in a brickwork, cut 4 | 4")
results["E"]["fig"] = save(fig, "fig3_scrambling.png")

# Circuit picture: 6 qubits, 4 layers
qc = QuantumCircuit(6)
for L in range(1, 5):
    for q in range(0 if L % 2 else 1, 5, 2):
        qc.unitary(random_unitary(4, seed=int(rng.integers(2**31))).data,
                   [q, q + 1], label="U")
fig = qc.draw("mpl", fold=-1, scale=0.9)
fig.suptitle("Brickwork scrambler: 6 qubits, 4 layers of Haar-random 2-qubit gates",
             x=0.02, ha="left", fontsize=11)
results["E"]["circuit_fig"] = save(fig, "fig4b_brickwork_circuit.png")

# =============================================================================
# F. Bosonic truncation: anharmonic oscillator H = p²/2 + x²/2 + g x⁴
# =============================================================================
Dbig = 700
a = np.diag(np.sqrt(np.arange(1, Dbig)), 1)
X = (a + a.T) / np.sqrt(2)
P2 = -((a - a.T) @ (a - a.T)) / 2          # p² = -(a - a†)²/2
X2, X4 = X @ X, X @ X @ X @ X


def e0(g, D):
    H = 0.5 * P2 + 0.5 * X2 + g * X4
    return float(np.linalg.eigvalsh(H[:D, :D])[0])


results["F"] = {"couplings": {}}
fig, ax = plt.subplots(figsize=(8, 4.6))
ks = np.arange(1, 7)
for g, col in [(0.1, AQUA), (0.5, BLUE), (2.0, ORANGE)]:
    ref300, ref200 = e0(g, 300), e0(g, 200)
    err = [max(abs(e0(g, 2**int(k)) - ref300), 1e-16) for k in ks]
    results["F"]["couplings"][str(g)] = {
        "E0_reference": ref300, "reference_drift_200_vs_300": abs(ref300 - ref200),
        "abs_error_by_k": dict(zip(map(int, ks), err))}
    ax.plot(ks, err, "-o", color=col, lw=2, ms=6, mec=SURFACE, mew=1.5,
            label=f"g = {g}")
ax.set_yscale("log")
ax.set_xlabel("Qubits per bosonic mode  k   (cutoff: 2ᵏ oscillator levels)")
ax.set_ylabel("|E₀(k) − E₀| (oscillator units)")
ax.legend(loc="upper right")
finish(ax, "Each extra qubit per mode doubles the cutoff; the error falls fast",
       "Ground-state energy of H = p²/2 + x²/2 + g x⁴ in a truncated Fock basis")
results["F"]["fig"] = save(fig, "fig6_boson_truncation.png")

# -----------------------------------------------------------------------------
with open(os.path.join(OUT, "results.json"), "w") as f:
    json.dump(results, f, indent=2)
print(json.dumps(results, indent=2)[:6000])
