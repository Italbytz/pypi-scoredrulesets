"""
Neural Logic Network (NLN) backend for Scored Rule Sets.

Implements the core ideas from:
  Payani & Fekri, "Learning Algorithms via Neural Logic Networks" (2019/2020).

Key concepts:
  - Continuous features are discretised into binary propositions via quantile
    thresholds (feature <= t  and  feature > t).
  - A *conjunction layer* learns which propositions to include in each rule
    using a log-linear evidence model with sigmoid activation (differentiable
    AND).  Each rule's weight vector selects relevant propositions; L1
    regularisation induces sparsity so only few propositions participate.
  - A *score layer* assigns per-class weights to each rule.
  - Training uses mini-batch SGD with L1 regularisation on both layers so
    that the final model is sparse.
  - After training, near-zero conjunction weights are pruned to yield crisp
    rules that map directly to a ScoredRuleSet.

Only numpy is required (no PyTorch / TensorFlow dependency).
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from ..runtime import predict as predict_from_ruleset
from ..runtime import predict_proba as predict_proba_from_ruleset
from ..schema import AggregationSpec, Atom, Rule, ScoredRuleSet
from .base import BaseRuleSetEstimator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EPS = 1e-7


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    out = np.empty_like(x)
    pos = x >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[neg])
    out[neg] = exp_x / (1.0 + exp_x)
    return out


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Row-wise softmax (2-D)."""
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def _cross_entropy(proba: np.ndarray, targets: np.ndarray) -> float:
    """Mean cross-entropy loss.  *targets* is a one-hot matrix."""
    clipped = np.clip(proba, _EPS, 1.0 - _EPS)
    return -float(np.mean(np.sum(targets * np.log(clipped), axis=1)))


# ---------------------------------------------------------------------------
# Main estimator
# ---------------------------------------------------------------------------


class NeuralLogicNetClassifier(BaseRuleSetEstimator):
    """Differentiable-logic rule learner inspired by Neural Logic Networks.

    The model architecture has two differentiable layers:

    1. **Conjunction layer** – For each rule *r*, a weight vector
       ``W_conj[r, :]`` is learned.  The rule activation for a sample is
       ``sigmoid( P @ W_conj[r, :] + b_conj[r] )`` where *P* is the binary
       proposition matrix.  Positive weights select propositions that should
       be **true**, negative weights select propositions that should be
       **false** (logical NOT).  L1 regularisation drives most weights to
       zero, yielding sparse conjunctions.

    2. **Score layer** – Each rule's activation is multiplied by a per-class
       weight ``W_score[r, c]`` and summed over rules to produce class logits.

    Parameters
    ----------
    n_rules : int
        Number of candidate rules (conjunction slots) to learn.
    n_bins : int
        Number of quantile thresholds per feature for discretisation.
    learning_rate : float
        SGD step size.
    l1_conj : float
        L1 penalty on the conjunction weights (encourages sparse rules).
    l1_score : float
        L1 penalty on the rule-score matrix (encourages few active rules).
    epochs : int
        Maximum number of training epochs.
    batch_size : int
        Mini-batch size.  ``0`` means full-batch.
    atom_threshold : float
        After training, conjunction weights with absolute value below this
        threshold are pruned (treated as "don't care").
    early_stopping_rounds : int
        Stop if validation F1 has not improved for this many epochs.
    validation_fraction : float
        Fraction of training data used for early-stopping evaluation.
    temperature : float
        Scaling factor for class logits during training (lower → sharper).
    random_state : int | None
        Seed for reproducibility.
    """

    def __init__(
        self,
        n_rules: int = 12,
        n_bins: int = 5,
        learning_rate: float = 0.3,
        l1_conj: float = 0.002,
        l1_score: float = 0.001,
        epochs: int = 300,
        batch_size: int = 0,
        atom_threshold: float = 0.1,
        early_stopping_rounds: int = 30,
        validation_fraction: float = 0.2,
        temperature: float = 1.0,
        random_state: int | None = None,
        max_thresholds_per_feature: int | None = None,
    ):
        self.n_rules = n_rules
        self.n_bins = n_bins
        self.learning_rate = learning_rate
        self.l1_conj = l1_conj
        self.l1_score = l1_score
        self.epochs = epochs
        self.batch_size = batch_size
        self.atom_threshold = atom_threshold
        self.early_stopping_rounds = early_stopping_rounds
        self.validation_fraction = validation_fraction
        self.temperature = temperature
        self.random_state = random_state
        self.max_thresholds_per_feature = max_thresholds_per_feature

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self, X, y):
        X_arr, y_arr = check_X_y(X, y, dtype="numeric")
        self.n_features_in_ = X_arr.shape[1]
        self.classes_ = unique_labels(y_arr)
        n_classes = len(self.classes_)
        rng = np.random.default_rng(self.random_state)

        # --- class mapping --------------------------------------------------
        class_to_idx = {label: idx for idx, label in enumerate(self.classes_)}
        y_idx = np.array([class_to_idx[v] for v in y_arr], dtype=int)

        # --- discretisation --------------------------------------------------
        self._thresholds_ = self._compute_thresholds(X_arr)
        P_full = self._binarise(X_arr)  # (N, n_props)
        n_props = P_full.shape[1]

        # --- train / val split -----------------------------------------------
        if self.validation_fraction > 0:
            tr_idx, val_idx = train_test_split(
                np.arange(len(y_idx)),
                test_size=self.validation_fraction,
                random_state=self.random_state,
                stratify=y_idx,
            )
        else:
            tr_idx = np.arange(len(y_idx))
            val_idx = tr_idx

        P_train, y_train = P_full[tr_idx], y_idx[tr_idx]
        P_val, y_val = P_full[val_idx], y_idx[val_idx]

        # one-hot targets
        Y_train = np.eye(n_classes, dtype=float)[y_train]

        # --- initialise weights ----------------------------------------------
        # W_conj: (n_rules, n_props) – conjunction weights
        # Each rule starts with a random sparse pattern: pick a few features
        # and set their weights higher to bootstrap learning.
        W_conj = rng.normal(0.0, 0.01, size=(self.n_rules, n_props))

        # Build a mapping: feature_index → list of proposition indices
        feat_to_props: dict[int, list[int]] = {}
        pidx = 0
        for j, thr in enumerate(self._thresholds_):
            n_thr = len(thr)
            feat_to_props[j] = list(range(pidx, pidx + n_thr))
            pidx += n_thr

        n_features = self.n_features_in_
        for r in range(self.n_rules):
            # Each rule starts with 2-4 random features activated
            n_init = rng.integers(2, min(5, n_features + 1))
            chosen_feats = rng.choice(n_features, size=n_init, replace=False)
            for fj in chosen_feats:
                props_for_f = feat_to_props.get(int(fj), [])
                if props_for_f:
                    # Pick 1-2 random propositions for this feature
                    n_pick = min(2, len(props_for_f))
                    selected = rng.choice(props_for_f, size=n_pick, replace=False)
                    for si in selected:
                        W_conj[r, si] = rng.uniform(0.3, 0.8)

        # b_conj: (n_rules,) – conjunction bias
        b_conj = rng.uniform(-0.5, 0.0, size=self.n_rules)
        # W_score: (n_rules, n_classes) – rule weights per class
        W_score = rng.normal(loc=0.0, scale=0.3, size=(self.n_rules, n_classes))
        # b_score: (n_classes,) – class bias
        b_score = np.zeros(n_classes, dtype=float)

        # --- Adam optimizer state -------------------------------------------
        adam_beta1 = 0.9
        adam_beta2 = 0.999
        adam_eps = 1e-8
        params_list = [W_conj, b_conj, W_score, b_score]
        m = [np.zeros_like(p) for p in params_list]  # first moment
        v = [np.zeros_like(p) for p in params_list]  # second moment
        t_step = 0

        # --- training loop ---------------------------------------------------
        best_f1 = -1.0
        patience_counter = 0
        best_params = (W_conj.copy(), b_conj.copy(), W_score.copy(), b_score.copy())

        N_train = len(y_train)
        bs = self.batch_size if self.batch_size > 0 else N_train

        for epoch in range(self.epochs):
            perm = rng.permutation(N_train)

            for start in range(0, N_train, bs):
                idx = perm[start : start + bs]
                P_b = P_train[idx]  # (B, D)
                Y_b = Y_train[idx]  # (B, C)

                # --- forward --------------------------------------------------
                logits, cache = self._forward(P_b, W_conj, b_conj, W_score, b_score)
                proba = _softmax(logits / max(self.temperature, _EPS))

                # --- backward -------------------------------------------------
                dW_conj, db_conj, dW_score, db_score = self._backward(
                    P_b, Y_b, proba, W_conj, b_conj, W_score, cache
                )

                # L1 gradients (proximal: only on conjunction and score weights)
                dW_conj += self.l1_conj * np.sign(W_conj)
                dW_score += self.l1_score * np.sign(W_score)

                # Adam update
                grads = [dW_conj, db_conj, dW_score, db_score]
                t_step += 1
                lr_t = self.learning_rate * np.sqrt(1.0 - adam_beta2 ** t_step) / (1.0 - adam_beta1 ** t_step)

                for i, (param, grad) in enumerate(zip(params_list, grads)):
                    m[i] = adam_beta1 * m[i] + (1.0 - adam_beta1) * grad
                    v[i] = adam_beta2 * v[i] + (1.0 - adam_beta2) * (grad ** 2)
                    param -= lr_t * m[i] / (np.sqrt(v[i]) + adam_eps)

                # Ensure params_list references stay current
                W_conj, b_conj, W_score, b_score = params_list

            # --- early stopping on validation ---------------------------------
            val_logits, _ = self._forward(P_val, W_conj, b_conj, W_score, b_score)
            val_pred = np.argmax(val_logits, axis=1)
            val_f1 = float(f1_score(y_val, val_pred, average="macro", zero_division=0))

            if val_f1 > best_f1 + 1e-6:
                best_f1 = val_f1
                patience_counter = 0
                best_params = (W_conj.copy(), b_conj.copy(), W_score.copy(), b_score.copy())
            else:
                patience_counter += 1
                if patience_counter >= self.early_stopping_rounds:
                    break

        # --- store best weights and extract rules ----------------------------
        self._W_conj_, self._b_conj_, self._W_score_, self._b_score_ = best_params
        self._n_props_ = n_props

        self.ruleset_ = self._extract_ruleset(n_classes)
        return self

    # ------------------------------------------------------------------
    # forward / backward (pure numpy, log-linear conjunction)
    # ------------------------------------------------------------------

    @staticmethod
    def _forward(
        P: np.ndarray,
        W_conj: np.ndarray,
        b_conj: np.ndarray,
        W_score: np.ndarray,
        b_score: np.ndarray,
    ) -> tuple[np.ndarray, dict]:
        """
        Forward pass with log-linear conjunction model.

        P        : (B, D)   – binary proposition matrix
        W_conj   : (R, D)   – conjunction weights
        b_conj   : (R,)     – conjunction biases
        W_score  : (R, C)   – per-class rule scores
        b_score  : (C,)     – class bias

        Returns logits (B, C) and a cache dict for backward.
        """
        # Rule evidence: how well each sample matches each rule
        evidence = P @ W_conj.T + b_conj[np.newaxis, :]  # (B, R)
        conj = _sigmoid(evidence)  # (B, R) – soft conjunction activation

        # Class logits: weighted sum of rule activations
        logits = conj @ W_score + b_score  # (B, C)

        cache = {"P": P, "evidence": evidence, "conj": conj}
        return logits, cache

    @staticmethod
    def _backward(
        P: np.ndarray,
        Y: np.ndarray,
        proba: np.ndarray,
        W_conj: np.ndarray,
        b_conj: np.ndarray,
        W_score: np.ndarray,
        cache: dict,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Backward pass (manual gradients).

        Returns (dW_conj, db_conj, dW_score, db_score).
        """
        B = P.shape[0]
        conj = cache["conj"]  # (B, R)

        # d_loss / d_logits  (softmax + cross-entropy)
        d_logits = (proba - Y) / B  # (B, C)

        # d_logits / d_score weights
        dW_score = conj.T @ d_logits  # (R, C)
        db_score = d_logits.sum(axis=0)  # (C,)

        # d_logits / d_conj
        d_conj = d_logits @ W_score.T  # (B, R)

        # d_conj / d_evidence = conj * (1 - conj)  (sigmoid derivative)
        d_evidence = d_conj * conj * (1.0 - conj)  # (B, R)

        # d_evidence / d_W_conj: evidence = P @ W_conj.T + b_conj
        dW_conj = d_evidence.T @ P  # (R, D)
        db_conj = d_evidence.sum(axis=0)  # (R,)

        return dW_conj, db_conj, dW_score, db_score

    # ------------------------------------------------------------------
    # discretisation helpers
    # ------------------------------------------------------------------

    def _compute_thresholds(self, X: np.ndarray) -> list[np.ndarray]:
        """Compute quantile-based thresholds per feature."""
        max_thr = self.max_thresholds_per_feature
        thresholds: list[np.ndarray] = []
        quantiles = np.linspace(0, 1, self.n_bins + 2)[1:-1]  # interior quantiles
        for j in range(X.shape[1]):
            col = X[:, j]
            thr = np.unique(np.quantile(col, quantiles))
            if len(thr) == 0:
                thr = np.array([float(col[0])]) if len(col) > 0 else np.array([0.0])
            # Apply threshold cap
            if max_thr is not None and len(thr) > max_thr:
                idx = np.round(np.linspace(0, len(thr) - 1, max_thr)).astype(int)
                thr = thr[idx]
            thresholds.append(thr)
        return thresholds

    def _binarise(self, X: np.ndarray) -> np.ndarray:
        """Convert X → binary proposition matrix P.

        For each feature j and each threshold t we create one proposition:
          - x_j <= t   →  1.0 if true, 0.0 otherwise

        The complement (x_j > t) is represented implicitly via negative
        conjunction weights (the model learns to negate via w < 0 since
        NOT(P) ≈ 1 − P  and  w·(1−P) = w − w·P).
        """
        parts: list[np.ndarray] = []
        for j, thr in enumerate(self._thresholds_):
            col = X[:, j : j + 1]  # (N, 1)
            parts.append((col <= thr[np.newaxis, :]).astype(float))
        return np.hstack(parts)

    def _proposition_meta(self) -> list[tuple[int, str, float]]:
        """Return (feature_index, op, threshold) for each proposition column."""
        meta: list[tuple[int, str, float]] = []
        for j, thr in enumerate(self._thresholds_):
            for t in thr:
                meta.append((j, "<=", float(t)))
        return meta

    # ------------------------------------------------------------------
    # rule extraction
    # ------------------------------------------------------------------

    def _extract_ruleset(self, n_classes: int) -> ScoredRuleSet:
        """Extract crisp rules from learned weights."""
        prop_meta = self._proposition_meta()
        feature_names = [f"f{j}" for j in range(self.n_features_in_)]

        rules: list[Rule] = []

        # Default rule (class bias → prior-like fallback)
        default_scores = self._b_score_.tolist()
        rules.append(
            Rule(
                atoms=[],
                scores=default_scores,
                rule_id="default",
                metadata={"source": "nln_bias"},
            )
        )

        for r in range(self.n_rules):
            w_r = self._W_conj_[r]  # (D,)
            scores_r = self._W_score_[r].tolist()

            # Skip rule if all scores are near-zero
            if max(abs(s) for s in scores_r) < 1e-4:
                continue

            # Find propositions with significant weight.
            # Use an adaptive threshold: absolute threshold OR top-k per rule.
            abs_w = np.abs(w_r)
            active_mask = abs_w > self.atom_threshold

            # Also limit to at most top-k most important propositions
            # to avoid overly complex rules.
            max_atoms_per_rule = min(8, n_classes * 3)
            if active_mask.sum() > max_atoms_per_rule:
                # Keep only top-k by weight magnitude
                top_k_idx = np.argsort(abs_w)[-max_atoms_per_rule:]
                mask = np.zeros_like(active_mask)
                mask[top_k_idx] = True
                active_mask = active_mask & mask

            active_props = np.where(active_mask)[0]

            if len(active_props) == 0:
                continue  # rule pruned away

            # Build atoms from active propositions.
            # Group by feature: for each feature, determine the single best
            # condition (upper bound or lower bound) to avoid contradictions.
            feat_best: dict[int, tuple[str, float, float]] = {}
            for p_idx in active_props:
                feat_j, _op, thr_val = prop_meta[p_idx]
                weight = float(w_r[p_idx])
                abs_weight = abs(weight)
                # Positive weight on "<=t" → want feature <= t
                # Negative weight on "<=t" → want feature > t
                op = "<=" if weight > 0 else ">"

                key = (feat_j, op)
                prev = feat_best.get(feat_j)
                if prev is None:
                    feat_best[feat_j] = (op, thr_val, abs_weight)
                else:
                    prev_op, prev_val, prev_w = prev
                    if prev_op == op:
                        # Same direction: keep the one with strongest weight
                        if abs_weight > prev_w:
                            feat_best[feat_j] = (op, thr_val, abs_weight)
                    else:
                        # Different directions on same feature → try to form interval
                        # Only valid if lower < upper
                        if op == ">" and prev_op == "<=":
                            lower, upper = thr_val, prev_val
                        elif op == "<=" and prev_op == ">":
                            lower, upper = prev_val, thr_val
                        else:
                            lower, upper = 0, 0

                        if lower < upper:
                            # Valid interval – store as special marker
                            feat_best[feat_j] = ("interval", lower, upper)
                        else:
                            # Contradictory – keep the one with stronger weight
                            if abs_weight > prev_w:
                                feat_best[feat_j] = (op, thr_val, abs_weight)

            atoms: list[Atom] = []
            for feat_j in sorted(feat_best.keys()):
                entry = feat_best[feat_j]
                fname = feature_names[feat_j] if feat_j < len(feature_names) else f"f{feat_j}"
                if entry[0] == "interval":
                    _, lower, upper = entry
                    atoms.append(Atom(feature=fname, op=">", value=lower))
                    atoms.append(Atom(feature=fname, op="<=", value=upper))
                else:
                    op, val, _ = entry
                    atoms.append(Atom(feature=fname, op=op, value=val))

            if not atoms:
                continue

            rules.append(
                Rule(
                    atoms=atoms,
                    scores=scores_r,
                    rule_id=f"nln_{r}",
                    metadata={"source": "nln", "rule_idx": r},
                )
            )

        ruleset = ScoredRuleSet(
            class_labels=self.classes_.tolist(),
            feature_names=feature_names,
            rules=rules,
            aggregation=AggregationSpec(type="argmax_sum", temperature=1.0),
            metadata={
                "backend": "nln",
                "n_rules_learned": self.n_rules,
                "n_rules_extracted": len(rules),
                "epochs": self.epochs,
            },
        )
        ruleset.validate()
        return ruleset

    # ------------------------------------------------------------------
    # predict / sklearn interface
    # ------------------------------------------------------------------

    def predict(self, X):
        check_is_fitted(self, "ruleset_")
        X_arr = check_array(X, dtype="numeric")
        return predict_from_ruleset(self.ruleset_, X_arr)

    def predict_proba(self, X):
        check_is_fitted(self, "ruleset_")
        X_arr = check_array(X, dtype="numeric")
        return predict_proba_from_ruleset(self.ruleset_, X_arr)

    def to_ruleset(self) -> ScoredRuleSet:
        check_is_fitted(self, "ruleset_")
        return self.ruleset_









