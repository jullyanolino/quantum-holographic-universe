# Code blocks quoted in the article, kept in one runnable file (Qiskit 2.4.x).
import itertools

import numpy as np
import qiskit
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import (SparsePauliOp, Statevector, entropy,
                                 partial_trace, random_statevector,
                                 random_unitary)

print("qiskit", qiskit.__version__)

# ---- 0. The resource formula ------------------------------------------------
N, lam = 16, 4                              # lam = λ^(1/3) / ω
n_q = N**2 * (8 + 9 * np.log2(lam))         # fermions + bosons
terms = 9**2 * N**4 + 16**2 * 9 * N**3      # quartic bosonic + cubic Yukawa
print(f"n_q = {n_q:,.0f}   Hamiltonian terms = {terms:,}")

# ---- 1. Jordan–Wigner Majoranas obey the Clifford algebra -------------------
def majorana(n, k):
    j, r = divmod(k, 2)
    s = ["I"] * n
    for q in range(j):
        s[n - 1 - q] = "Z"                  # Qiskit labels are little-endian
    s[n - 1 - j] = "X" if r == 0 else "Y"
    return SparsePauliOp("".join(s))

n = 4
g = [majorana(n, k) for k in range(2 * n)]
err = max(
    np.abs((g[a].dot(g[b]) + g[b].dot(g[a])).to_matrix()
           - 2 * (a == b) * np.eye(2**n)).max()
    for a, b in itertools.product(range(2 * n), repeat=2)
)
print(f"max |{{γa,γb}} - 2δab| = {err:.1e}")

# ---- 2. One quartic Majorana term as a circuit ------------------------------
term = QuantumCircuit(4)
term.append(PauliEvolutionGate(SparsePauliOp("XZYZ"), time=0.2), range(4))
tq = transpile(term, basis_gates=["cx", "h", "rz", "s", "sdg"], optimization_level=0)
print(dict(tq.count_ops()))                 # tq.draw("mpl") gives the figure

# ---- 3. Page curve from Haar-random states ----------------------------------
def S_bits(sv, m, n):
    """Entropy, in bits, of qubits 0..m-1 of the pure state sv."""
    M = np.asarray(sv.data).reshape(2 ** (n - m), 2 ** m)
    p = np.linalg.svd(M, compute_uv=False) ** 2
    p = p[p > 1e-14]
    return float(-(p * np.log2(p)).sum())

def page_bits(dA, dB):
    """Exact Page average entropy for subsystem dimensions dA, dB."""
    m, n = sorted((dA, dB))
    return float((np.sum(1.0 / np.arange(n + 1, m * n + 1)) - (m - 1) / (2 * n))
                 / np.log(2))

n, samples = 12, 400
E = np.empty((samples, n - 1))
for i in range(samples):
    sv = random_statevector(2**n, seed=5000 + i)
    E[i] = [S_bits(sv, m, n) for m in range(1, n)]
mean = E.mean(0)
ci95 = 1.96 * E.std(0, ddof=1) / np.sqrt(samples)
exact = np.array([page_bits(2**m, 2**(n - m)) for m in range(1, n)])
print(f"m = 6: mean {mean[5]:.4f} ± {ci95[5]:.4f}   Page {exact[5]:.4f}")

sv8 = random_statevector(2**8, seed=1)      # cross-check against Qiskit's own route
ref = float(entropy(partial_trace(sv8, list(range(3, 8))), base=2))
assert abs(S_bits(sv8, 3, 8) - ref) < 1e-9

# ---- 4. Brickwork scrambler -------------------------------------------------
rng = np.random.default_rng(2611)

def brickwork_entropy(nq=8, cut=4, layers=14):
    sv = Statevector.from_label("0" * nq)
    out = [S_bits(sv, cut, nq)]
    for L in range(1, layers + 1):
        for q in range(0 if L % 2 else 1, nq - 1, 2):
            U = random_unitary(4, seed=int(rng.integers(2**31)))
            sv = sv.evolve(U, [q, q + 1])
        out.append(S_bits(sv, cut, nq))
    return out

curves = np.array([brickwork_entropy() for _ in range(200)])
print(f"layer 14: {curves.mean(0)[-1]:.2f} ± {curves.std(0, ddof=1)[-1]:.2f} bits"
      f"  (Page {page_bits(2**4, 2**4):.2f})")

# ---- 5. Truncating a bosonic mode -------------------------------------------
D = 700
a = np.diag(np.sqrt(np.arange(1, D)), 1)
x = (a + a.T) / np.sqrt(2)
p2 = -((a - a.T) @ (a - a.T)) / 2

def E0(g, levels):
    H = 0.5 * p2 + 0.5 * x @ x + g * np.linalg.matrix_power(x, 4)
    return np.linalg.eigvalsh(H[:levels, :levels])[0]   # project, then diagonalise

for g in (0.1, 0.5, 2.0):
    ref = E0(g, 300)
    print(g, [f"{abs(E0(g, 2**k) - ref):.1e}" for k in range(1, 7)])
