"""
Spike: Exact (CP-SAT) Scored-Rule-Set solver as certified-compact backend.

Reproduces the feasibility evidence cited in the AutoML paper discussion
(`articles/2027/automl-scoredrulesets/`, Section 5): exact DNF induction with
a global atom budget returns *certified-optimal* rule sets at the compact end
of the Pareto front — something no heuristic or gradient-based backend can
guarantee — but its runtime explodes in literal count / budget, not in n.

Formulation (binary target, one class = "positive"):
  Literal pool:  per feature/threshold the propositions (x_j <= t), (x_j > t)
                 or, for binary features, x_j and NOT x_j.
  Variables:     sel[r][d]  - rule r uses literal d
                 fire[r][i] - rule r fires on sample i
                 pred[i]    - OR over rules (disjunction = DNF classifier)
  Constraints:   sum_d sel[r][d] <= k              (literals per rule)
                 sum_{r,d} sel[r][d] <= B          (global atom budget)
                 fire[r][i] <=> all selected literals true on i   (AND,
                   both directions required: one-sided implications allow
                   memorization via vacuous rules)
                 pred[i] <=> OR_r fire[r][i]                     (OR)
  Objective:     exact F1 via parametric binary search on the threshold t.
                 With P = #positives (constant) and PH = #predicted:
                 F1 >= t  <=>  2*Q*TP >= q*(PH + P),  t = q/Q (Q = 10000).
                 All coefficients are integral (CP-SAT rejects floats).
                 The UNSAT branches (t' > t proven infeasible) form the
                 optimality certificate. Caveat: on timeout CP-SAT returns
                 UNKNOWN, which this spike treats as infeasible — certificates
                 are only as good as the timeout-free proofs.

Scenarios:
  S1  DNF 3x2 (known optimum: F1 = 1.0 at B = 6, three 2-literal rules)
  S2  Breast Cancer (quantile literals over 10 features)
  S3  Scaling probe: n-ladder on DNF (near-linear in n)

Run:
    pip install ortools scikit-learn numpy
    python3 examples/benchmarks/spike_exact_cpusat.py
"""
from __future__ import annotations

import time

import numpy as np
from ortools.sat.python import cp_model
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Literal pools
# ---------------------------------------------------------------------------

def binary_literal_pool(X: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Binary features -> literals x_j and NOT x_j."""
    cols, names = [], []
    for j in range(X.shape[1]):
        cols.append(X[:, j].astype(bool))
        names.append(f"x{j}")
        cols.append(~X[:, j].astype(bool))
        names.append(f"~x{j}")
    return np.column_stack(cols), names


def quantile_literal_pool(
    X_tr: np.ndarray, X_all: np.ndarray, n_bins: int = 3,
    max_features: int | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Threshold literals from train quantiles; transformed to X_all."""
    cols, names = [], []
    feats = range(X_tr.shape[1]) if max_features is None else range(max_features)
    for j in feats:
        col = X_tr[:, j]
        uniq = np.unique(col)
        if len(uniq) <= 2:
            thr = [0.5] if set(uniq) <= {0.0, 1.0} else list((uniq[:-1] + uniq[1:]) / 2)
        else:
            qs = np.linspace(0, 100, n_bins + 1)[1:-1]
            thr = list(np.unique(np.percentile(col, qs)))
        for t in thr:
            v = X_all[:, j] <= t
            cols.append(v); names.append(f"x{j}<={t:.2f}")
            cols.append(~v); names.append(f"x{j}>{t:.2f}")
    return np.column_stack(cols), names


# ---------------------------------------------------------------------------
# Exact solver
# ---------------------------------------------------------------------------

def solve_exact_f1(
    L: np.ndarray, y: np.ndarray, *, R: int, k: int, B: int,
    time_limit: float = 60.0,
) -> dict:
    """Maximize F1 under atom budget B (R rules, each <= k literals).

    Returns dict with f1, rules (literal index lists), status, total_seconds.
    """
    n, D = L.shape
    P = int(y.sum())
    if P == 0 or P == n:
        return {"f1": 0.0, "status": "degenerate", "total_seconds": 0.0}

    lo, hi = 0.0, 1.0
    best = None
    t0 = time.time()

    def feasible(t: float) -> dict | None:
        """CP-SAT feasibility for F1 >= t; solution dict or None."""
        m = cp_model.CpModel()
        sel = [[m.NewBoolVar(f"s{r}_{d}") for d in range(D)] for r in range(R)]
        fire = [[m.NewBoolVar(f"f{r}_{i}") for i in range(n)] for r in range(R)]
        pred = [m.NewBoolVar(f"p_{i}") for i in range(n)]

        for r in range(R):
            m.Add(sum(sel[r]) <= k)
        m.Add(sum(sum(s) for s in sel) <= B)

        # AND: fire[r][i] <=> all selected literals true on sample i.
        false_mask = ~L.astype(bool)   # (n, D)
        never_true_cols = np.where(L.astype(bool).sum(axis=0) == 0)[0]
        for d in never_true_cols:      # literal never true -> never selectable
            for r in range(R):
                m.Add(sel[r][d] == 0)
        for r in range(R):
            for i in range(n):
                bad = np.where(false_mask[i])[0]
                if len(bad) == D:
                    m.Add(fire[r][i] == 0)
                elif len(bad) > 0:
                    # => direction: a false selected literal suppresses fire
                    for d in bad:
                        m.Add(fire[r][i] + sel[r][d] <= 1)
                    # <= direction: all selected literals true -> fire MUST be 1
                    # (prevents memorization via vacuous/disabled rules)
                    m.Add(fire[r][i] >= 1 - sum(sel[r][d] for d in bad))

        # OR: pred[i] iff at least one rule fires
        for i in range(n):
            for r in range(R):
                m.AddImplication(fire[r][i], pred[i])
            m.Add(pred[i] <= sum(fire[r][i] for r in range(R)))

        # Integral F1 threshold: 2Q*TP >= q*(PH + P), t = q/Q, Q = 10000.
        q = int(round(t * 10000))
        tp_expr = sum(2 * 10000 * int(y[i]) * pred[i] for i in range(n))
        ph_expr = sum(q * pred[i] for i in range(n))
        m.Add(tp_expr - ph_expr >= q * P)

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = time_limit
        st = solver.Solve(m)
        if st in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            chosen = []
            for r in range(R):
                lits = [d for d in range(D) if solver.Value(sel[r][d])]
                if lits:
                    chosen.append(lits)
            pv = np.array([solver.Value(pred[i]) for i in range(n)], dtype=bool)
            tp = int((pv & y.astype(bool)).sum())
            ph = int(pv.sum())
            f1 = 2 * tp / (ph + P)
            return {"f1": f1, "rules": chosen,
                    "status": "optimal" if st == cp_model.OPTIMAL else "feasible",
                    "solver_time": solver.WallTime()}
        return None

    # Binary search for the largest feasible t. F1 is discrete, so the search
    # converges to the nearest attainable value; the UNSAT branches above
    # constitute the optimality certificate.
    for _ in range(15):
        mid = (lo + hi) / 2
        res = feasible(mid)
        if res is not None:
            best = res
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-4:
            break
    total = time.time() - t0
    if best is not None:
        best["total_seconds"] = total
        best["budget"] = B
        best["R"] = R
        best["k"] = k
    return best or {"f1": 0.0, "status": "infeasible", "total_seconds": total}


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

def gen_dnf(n: int, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 2, size=(n, 16))
    y = ((X[:, 0] & X[:, 1]) | (X[:, 2] & X[:, 3]) | (X[:, 4] & X[:, 5])).astype(int)
    return X.astype(np.float64), y


def scenario_dnf() -> None:
    print("\n=== S1: DNF 3x2 (known optimum: F1=1.0 at B=6) ===")
    X, y = gen_dnf(400)
    L, _ = binary_literal_pool(X)
    print(f"  n={len(y)}, D={L.shape[1]} literals, P={int(y.sum())}")
    for B in (2, 4, 6, 8):
        r = solve_exact_f1(L, y, R=3, k=2, B=B, time_limit=30)
        n_atoms = sum(len(x) for x in r.get("rules", []))
        print(f"  B={B}: F1={r['f1']:.4f} ({r['status']}), atoms={n_atoms}, "
              f"solve {r.get('total_seconds', 0):.1f}s, rules={r.get('rules')}")


def scenario_breast() -> None:
    print("\n=== S2: Breast Cancer (quantile literals over 10 features) ===")
    data = load_breast_cancer()
    X, y = data.data, 1 - data.target  # positive = malignant (minority)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y)
    sc = StandardScaler().fit(X_tr)
    X_tr_s, X_te_s = sc.transform(X_tr), sc.transform(X_te)
    Xs = np.vstack([X_tr_s, X_te_s])
    ys = np.concatenate([y_tr, y_te])
    L, _ = quantile_literal_pool(X_tr_s, Xs, n_bins=2, max_features=10)
    print(f"  n={len(ys)}, D={L.shape[1]} literals (10 features x 2 bins x 2 ops), "
          f"P={int(ys.sum())}")
    for B in (2, 4, 6):
        r = solve_exact_f1(L, ys, R=3, k=2, B=B, time_limit=60)
        n_atoms = sum(len(x) for x in r.get("rules", []))
        print(f"  B={B}: F1={r['f1']:.4f} ({r['status']}), atoms={n_atoms}, "
              f"solve {r.get('total_seconds', 0):.1f}s")


def scenario_scaling() -> None:
    print("\n=== S3: Scaling probe (DNF, B=6, n-ladder) ===")
    for n in (100, 200, 400, 800):
        X, y = gen_dnf(n)
        L, _ = binary_literal_pool(X)
        r = solve_exact_f1(L, y, R=3, k=2, B=6, time_limit=45)
        print(f"  n={n}: F1={r['f1']:.4f} ({r['status']}), "
              f"solve {r.get('total_seconds', 0):.1f}s")


if __name__ == "__main__":
    print("=" * 62)
    print("  Spike: Exact CP-SAT Scored-Rule-Set solver")
    print("=" * 62)
    scenario_dnf()
    scenario_breast()
    scenario_scaling()
