"""
HMMR — Hidden Markov Model with Regime Regressions
====================================================
Chaque état caché possède sa propre régression linéaire :
    y_t = β_k · x_t + ε_k    si  s_t = k

Pipeline :
  1. Simulation de rendements 10 min réalistes (multi-régimes)
  2. Feature engineering sur ces rendements
  3. Estimation HMMR par EM (Baum-Welch adapté)
  4. Prédiction : ŷ_{t+1} = Σ_k P(s_{t+1}=k | x_{1:t}) · β_k · x_{t+1}
  5. Backtest walk-forward + métriques + visualisations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
from matplotlib.patches import Patch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.special import logsumexp
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────
# STYLE
# ─────────────────────────────────────────────────────────
COLORS = ["#2ECC71", "#E74C3C", "#3498DB", "#F39C12", "#9B59B6"]
NAMES  = ["Bull", "Bear", "Ranging", "Volatile", "Mixed"]

plt.rcParams.update({
    "figure.facecolor": "#0D1117", "axes.facecolor": "#161B22",
    "axes.edgecolor": "#30363D", "axes.labelcolor": "#C9D1D9",
    "xtick.color": "#8B949E", "ytick.color": "#8B949E",
    "text.color": "#C9D1D9", "grid.color": "#21262D",
    "grid.linestyle": "--", "grid.alpha": 0.5,
    "font.family": "monospace",
    "legend.facecolor": "#161B22", "legend.edgecolor": "#30363D",
})


# ─────────────────────────────────────────────────────────
# 1. SIMULATION DE RENDEMENTS 10 MIN MULTI-RÉGIMES
# ─────────────────────────────────────────────────────────
def simulate_10min_returns(n_bars: int = 2000, n_states: int = 3,
                            seed: int = 42) -> pd.DataFrame:
    """
    Simule directement des rendements 10 min avec régimes cachés.
    Chaque régime a : drift, vol, autocorrélation, et sensibilité volume.
    """
    rng = np.random.default_rng(seed)

    # Paramètres par régime
    _all_params = [
        dict(mu=+0.0008, sigma=0.0035, rho=+0.15, vol_beta=+0.0002),
        dict(mu=-0.0006, sigma=0.0060, rho=-0.10, vol_beta=-0.0003),
        dict(mu=+0.0001, sigma=0.0015, rho=+0.05, vol_beta=+0.0000),
        dict(mu=+0.0000, sigma=0.0120, rho=-0.20, vol_beta=+0.0001),
    ]
    regime_params = {k: _all_params[k] for k in range(min(n_states, len(_all_params)))}

    # Matrice de transition (persistance des régimes)
    A = np.full((n_states, n_states), 0.05 / (n_states - 1))
    np.fill_diagonal(A, 0.95)
    A /= A.sum(axis=1, keepdims=True)

    # Distribution initiale
    pi = np.ones(n_states) / n_states

    # Simulation des états cachés (chaîne de Markov)
    states = np.zeros(n_bars, dtype=int)
    states[0] = rng.choice(n_states, p=pi)
    for t in range(1, n_bars):
        states[t] = rng.choice(n_states, p=A[states[t - 1]])

    # Simulation des rendements conditionnels au régime
    returns = np.zeros(n_bars)
    volumes = np.zeros(n_bars)
    prev_ret = 0.0

    for t in range(n_bars):
        k = states[t]
        p = regime_params[k]
        # Volume stochastique (log-normal)
        volumes[t] = np.exp(rng.normal(4.5, 0.6))
        # Rendement avec autocorrélation et effet volume
        noise = rng.normal(0, p["sigma"])
        returns[t] = p["mu"] + p["rho"] * prev_ret + p["vol_beta"] * np.log(volumes[t]) + noise
        prev_ret = returns[t]

    # Timestamps (barres 10 min sur jours de trading)
    start = pd.Timestamp("2023-01-02 09:30:00")
    bars_per_day = 39  # 6h30 / 10min
    timestamps = []
    day = start
    bar_count = 0
    while len(timestamps) < n_bars:
        bar_time = day + pd.Timedelta(minutes=10 * (bar_count % bars_per_day))
        if bar_count % bars_per_day == 0 and bar_count > 0:
            day += pd.Timedelta(days=1)
            while day.weekday() >= 5:
                day += pd.Timedelta(days=1)
            bar_time = day
        timestamps.append(bar_time)
        bar_count += 1

    df = pd.DataFrame({
        "timestamp": timestamps[:n_bars],
        "return":    returns,
        "volume":    volumes,
        "true_state": states,
    }).set_index("timestamp")

    # Prix reconstruit (pour visualisation)
    df["price"] = 100 * np.exp(df["return"].cumsum())

    print(f"✅  Simulation : {n_bars} barres 10 min | {n_states} régimes")
    for k in range(n_states):
        mask = states == k
        print(f"   Régime {NAMES[k]:8s} : {mask.sum():4d} barres | "
              f"μ={returns[mask].mean():.5f}  σ={returns[mask].std():.5f}")
    return df


# ─────────────────────────────────────────────────────────
# 2. FEATURE ENGINEERING SUR RENDEMENTS
# ─────────────────────────────────────────────────────────
def build_features(df: pd.DataFrame, lags: int = 5) -> pd.DataFrame:
    """Features basées uniquement sur les rendements et le volume."""
    d = df.copy()
    r = d["return"]
    v = np.log(d["volume"] + 1e-9)

    # Lags de rendements
    for i in range(1, lags + 1):
        d[f"ret_lag{i}"] = r.shift(i)

    # Volatilité réalisée (rolling)
    d["vol_5"]  = r.shift(1).rolling(5).std()
    d["vol_10"] = r.shift(1).rolling(10).std()

    # Momentum
    d["mom_3"] = r.shift(1).rolling(3).sum()
    d["mom_6"] = r.shift(1).rolling(6).sum()

    # Volume features
    d["vol_log"]   = v.shift(1)
    d["vol_chg"]   = v.shift(1) - v.shift(2)
    d["vol_ratio"] = v.shift(1) - v.shift(1).rolling(10).mean()

    # Asymétrie récente
    d["skew_10"] = r.shift(1).rolling(10).skew()

    # Cible : rendement courant (prédit à partir des features laggées)
    d["target"] = r

    d = d.dropna()
    return d

FEATURE_COLS = [
    "ret_lag1", "ret_lag2", "ret_lag3", "ret_lag4", "ret_lag5",
    "vol_5", "vol_10", "mom_3", "mom_6",
    "vol_log", "vol_chg", "vol_ratio", "skew_10",
]


# ─────────────────────────────────────────────────────────
# 3. HMMR — ESTIMATION PAR EM
# ─────────────────────────────────────────────────────────
class HMMR:
    """
    Hidden Markov Model with Regime Regressions.

    Chaque état k possède :
      - Un modèle de régression : μ_k(x) = β_k · x
      - Une variance résiduelle  : σ²_k
    L'algorithme EM alterne :
      E-step : calcul des probabilités postérieures γ_t(k) via forward-backward
      M-step : mise à jour de A, π, β_k, σ²_k par régression pondérée
    """

    def __init__(self, n_states: int = 3, n_iter: int = 50,
                 tol: float = 1e-4, alpha: float = 1e-3, seed: int = 42):
        self.n_states = n_states
        self.n_iter   = n_iter
        self.tol      = tol
        self.alpha    = alpha   # régularisation Ridge
        self.seed     = seed
        self.converged = False

    # ── Initialisation ──────────────────────────────────
    def _init_params(self, X: np.ndarray, y: np.ndarray):
        rng = np.random.default_rng(self.seed)
        K, T, D = self.n_states, len(y), X.shape[1]

        # Matrice de transition
        self.A = np.full((K, K), 0.1 / (K - 1))
        np.fill_diagonal(self.A, 0.9)
        self.A /= self.A.sum(axis=1, keepdims=True)

        # Distribution initiale
        self.pi = np.ones(K) / K

        # Initialisation des betas par k-means sur y
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=K, random_state=self.seed, n_init=10)
        labels = km.fit_predict(y.reshape(-1, 1))

        self.betas  = np.zeros((K, D))
        self.sigmas = np.ones(K) * y.std()

        for k in range(K):
            mask = labels == k
            if mask.sum() > D + 1:
                reg = Ridge(alpha=self.alpha)
                reg.fit(X[mask], y[mask])
                self.betas[k]  = reg.coef_
                self.sigmas[k] = max(np.std(y[mask] - reg.predict(X[mask])), 1e-6)

    # ── Densité d'émission (log) ─────────────────────────
    def _log_emission(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """log p(y_t | x_t, s_t=k) pour tout t et k. Shape: (T, K)"""
        T, K = len(y), self.n_states
        log_emit = np.zeros((T, K))
        for k in range(K):
            mu_k = X @ self.betas[k]
            log_emit[:, k] = stats.norm.logpdf(y, loc=mu_k, scale=self.sigmas[k])
        return log_emit

    # ── Forward-Backward ────────────────────────────────
    def _forward(self, log_emit: np.ndarray) -> tuple:
        T, K = log_emit.shape
        log_alpha = np.zeros((T, K))
        log_alpha[0] = np.log(self.pi + 1e-300) + log_emit[0]
        log_A = np.log(self.A + 1e-300)

        for t in range(1, T):
            for k in range(K):
                log_alpha[t, k] = logsumexp(log_alpha[t-1] + log_A[:, k]) + log_emit[t, k]

        log_lik = logsumexp(log_alpha[-1])
        return log_alpha, log_lik

    def _backward(self, log_emit: np.ndarray) -> np.ndarray:
        T, K = log_emit.shape
        log_beta = np.zeros((T, K))
        log_A = np.log(self.A + 1e-300)

        for t in range(T - 2, -1, -1):
            for k in range(K):
                log_beta[t, k] = logsumexp(log_A[k] + log_emit[t+1] + log_beta[t+1])

        return log_beta

    def _e_step(self, X, y):
        log_emit  = self._log_emission(X, y)
        log_alpha, log_lik = self._forward(log_emit)
        log_beta  = self._backward(log_emit)
        log_A     = np.log(self.A + 1e-300)

        T, K = log_emit.shape

        # γ_t(k) = P(s_t=k | observations)
        log_gamma = log_alpha + log_beta
        log_gamma -= logsumexp(log_gamma, axis=1, keepdims=True)
        gamma = np.exp(log_gamma)

        # ξ_t(j,k) = P(s_t=j, s_{t+1}=k | observations)
        log_xi = np.zeros((T-1, K, K))
        for t in range(T - 1):
            for j in range(K):
                for k in range(K):
                    log_xi[t, j, k] = (log_alpha[t, j] + log_A[j, k]
                                       + log_emit[t+1, k] + log_beta[t+1, k])
            log_xi[t] -= logsumexp(log_xi[t].reshape(-1))

        xi = np.exp(log_xi)
        return gamma, xi, log_lik

    def _m_step(self, X, y, gamma, xi):
        K = self.n_states

        # Mise à jour π et A
        self.pi = gamma[0] / gamma[0].sum()
        self.A  = xi.sum(axis=0)
        self.A /= self.A.sum(axis=1, keepdims=True) + 1e-300

        # Mise à jour β_k et σ_k (régression pondérée)
        for k in range(K):
            w = gamma[:, k]
            w_sum = w.sum()
            if w_sum < 1e-6:
                continue

            # Régression Ridge pondérée
            W = np.diag(w)
            XtW  = X.T @ W
            XtWX = XtW @ X + self.alpha * np.eye(X.shape[1])
            XtWy = XtW @ y
            self.betas[k] = np.linalg.solve(XtWX, XtWy)

            resid = y - X @ self.betas[k]
            self.sigmas[k] = max(np.sqrt((w * resid**2).sum() / w_sum), 1e-6)

    # ── Fit (EM) ─────────────────────────────────────────
    def fit(self, X: np.ndarray, y: np.ndarray):
        self._init_params(X, y)
        prev_lik = -np.inf

        for i in range(self.n_iter):
            gamma, xi, log_lik = self._e_step(X, y)
            self._m_step(X, y, gamma, xi)

            delta = log_lik - prev_lik
            if i % 10 == 0:
                print(f"   EM iter {i:3d} | log-lik = {log_lik:.2f} | Δ = {delta:.4f}")
            if abs(delta) < self.tol and i > 5:
                print(f"   ✅ Convergence à l'itération {i}")
                self.converged = True
                break
            prev_lik = log_lik

        self.gamma_train_ = gamma
        self.log_lik_     = log_lik
        return self

    # ── Prédiction ───────────────────────────────────────
    def predict(self, X_hist: np.ndarray, y_hist: np.ndarray,
                X_fut: np.ndarray) -> tuple:
        """
        Prédit y_{t+1} pour chaque point de X_fut.
        Utilise les probabilités forward sur (X_hist, y_hist)
        puis propage via la matrice de transition.

        Retourne : (y_pred, states_pred, state_proba)
        """
        # Forward sur historique complet
        log_emit_hist = self._log_emission(X_hist, y_hist)
        log_alpha, _  = self._forward(log_emit_hist)

        T_fut = len(X_fut)
        log_A = np.log(self.A + 1e-300)
        K     = self.n_states

        preds       = np.zeros(T_fut)
        state_proba = np.zeros((T_fut, K))
        states_pred = np.zeros(T_fut, dtype=int)

        # Probabilité courante = dernière ligne forward normalisée
        log_curr = log_alpha[-1]

        # Ajouter séquentiellement chaque point futur
        log_emit_fut = self._log_emission(X_fut, np.zeros(T_fut))  # dummy y

        for t in range(T_fut):
            # Proba de l'état suivant : P(s_{t+1} | données jusqu'à t)
            log_next = np.array([logsumexp(log_curr + log_A[:, k]) for k in range(K)])
            log_next -= logsumexp(log_next)
            proba_next = np.exp(log_next)

            # Prédiction = Σ_k P(s_{t+1}=k) · β_k · x_{t+1}
            mu_k = X_fut[t] @ self.betas.T   # shape (K,)
            preds[t] = (proba_next * mu_k).sum()

            state_proba[t] = proba_next
            states_pred[t] = np.argmax(proba_next)

            # Mise à jour forward avec la vraie émission
            log_curr = log_next + log_emit_fut[t]
            log_curr -= logsumexp(log_curr)

        return preds, states_pred, state_proba

    def predict_states(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Viterbi simplifié : état le plus probable à chaque t."""
        gamma, _, _ = self._e_step(X, y)
        return np.argmax(gamma, axis=1)


# ─────────────────────────────────────────────────────────
# 4. BACKTEST WALK-FORWARD
# ─────────────────────────────────────────────────────────
def walk_forward(df: pd.DataFrame, n_states: int = 3,
                 train_frac: float = 0.70) -> dict:

    split = int(len(df) * train_frac)
    df_tr = df.iloc[:split].copy()
    df_te = df.iloc[split:].copy()

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(df_tr[FEATURE_COLS])
    X_te = scaler.transform(df_te[FEATURE_COLS])
    y_tr = df_tr["target"].values
    y_te = df_te["target"].values

    print(f"\n📐  Entraînement HMMR ({n_states} états) sur {len(X_tr)} barres...")
    model = HMMR(n_states=n_states, n_iter=80, alpha=1e-3)
    model.fit(X_tr, y_tr)

    print(f"📈  Prédiction sur {len(X_te)} barres test...")
    preds, states_te, proba_te = model.predict(X_tr, y_tr, X_te)
    states_tr = model.predict_states(X_tr, y_tr)

    df_tr["regime"]  = states_tr
    df_te["regime"]  = states_te
    df_te["pred"]    = preds

    # ── Métriques ──────────────────────────────────────
    mask  = ~(np.isnan(y_te) | np.isnan(preds))
    rmse  = np.sqrt(mean_squared_error(y_te[mask], preds[mask]))
    mae   = mean_absolute_error(y_te[mask], preds[mask])
    corr, pval = stats.pearsonr(y_te[mask], preds[mask])
    dir_acc = np.mean(np.sign(y_te[mask]) == np.sign(preds[mask]))

    df_te["strat_ret"] = np.sign(df_te["pred"]) * df_te["target"]
    cum_strat = df_te["strat_ret"].cumsum()
    cum_bh    = df_te["target"].cumsum()

    def sharpe(r):
        mu, s = r.mean(), r.std()
        return mu / s * np.sqrt(252 * 39) if s > 1e-12 else 0.0

    metrics = {
        "rmse": rmse, "mae": mae, "corr": corr, "pval": pval,
        "dir_acc": dir_acc,
        "sharpe_strat": sharpe(df_te["strat_ret"]),
        "sharpe_bh":    sharpe(df_te["target"]),
    }

    return {
        "model": model, "scaler": scaler,
        "df_tr": df_tr, "df_te": df_te,
        "states_tr": states_tr, "states_te": states_te,
        "proba_te": proba_te,
        "X_tr": X_tr, "y_tr": y_tr,
        "X_te": X_te, "y_te": y_te,
        "preds": preds,
        "cum_strat": cum_strat, "cum_bh": cum_bh,
        "metrics": metrics, "n_states": n_states,
    }


# ─────────────────────────────────────────────────────────
# 5. VISUALISATIONS
# ─────────────────────────────────────────────────────────
def plot_report(res: dict, save: str = "hmmr_report.png"):
    df_all = pd.concat([res["df_tr"], res["df_te"]])
    all_st = np.concatenate([res["states_tr"], res["states_te"]])
    df_te  = res["df_te"]
    m      = res["metrics"]
    K      = res["n_states"]

    fig = plt.figure(figsize=(20, 24))
    fig.patch.set_facecolor("#0D1117")
    gs  = gridspec.GridSpec(5, 3, figure=fig,
                            hspace=0.50, wspace=0.35,
                            left=0.06, right=0.97,
                            top=0.94, bottom=0.04)
    fig.suptitle("HMMR · Régression par Régime · Rendements 10 min",
                 fontsize=17, fontweight="bold", color="#E6EDF3", y=0.97)

    # 1. Prix + régimes
    ax = fig.add_subplot(gs[0, :])
    ax.set_title("Prix reconstruit & Régimes HMMR", fontsize=11)
    dates = df_all.index
    price = df_all["price"]
    ax.plot(dates, price, color="#58A6FF", lw=0.8, zorder=3)
    for i in range(len(all_st) - 1):
        ax.axvspan(dates[i], dates[i+1], alpha=0.22,
                   color=COLORS[all_st[i]], linewidth=0)
    ax.axvline(df_te.index[0], color="#F0E68C", lw=1.5, ls="--")
    legend_els = [Patch(facecolor=COLORS[k], alpha=0.6, label=NAMES[k])
                  for k in range(K)]
    legend_els.append(plt.Line2D([0],[0], color="#F0E68C", ls="--", label="Split"))
    ax.legend(handles=legend_els, loc="upper left", fontsize=8)
    ax.set_ylabel("Prix")

    # 2. Probabilités postérieures (test)
    ax = fig.add_subplot(gs[1, :])
    ax.set_title("Probabilités postérieures des régimes (test)", fontsize=11)
    bottom = np.zeros(len(df_te))
    for k in range(K):
        ax.fill_between(df_te.index, bottom, bottom + res["proba_te"][:, k],
                        color=COLORS[k], alpha=0.75, label=NAMES[k])
        bottom += res["proba_te"][:, k]
    ax.set_ylim(0, 1); ax.set_ylabel("P(régime)")
    ax.legend(loc="upper left", fontsize=8, ncol=K)

    # 3. Rendements réels vs prédits
    ax = fig.add_subplot(gs[2, :2])
    ax.set_title("Rendements réels vs prédits HMMR (test)", fontsize=11)
    ax.plot(df_te.index, df_te["target"], color="#8B949E", lw=0.7, alpha=0.7, label="Réel")
    ax.plot(df_te.index, df_te["pred"],   color="#FF7B72", lw=1.2, label="Prédit HMMR")
    ax.axhline(0, color="#30363D", lw=1)
    ax.legend(fontsize=9); ax.set_ylabel("Log-rendement")

    # 4. Scatter réel vs prédit
    ax = fig.add_subplot(gs[2, 2])
    ax.set_title("Scatter réel vs prédit", fontsize=11)
    c_pts = [COLORS[s] for s in res["states_te"]]
    ax.scatter(df_te["pred"], df_te["target"], c=c_pts, alpha=0.3, s=10)
    lim = max(df_te["target"].abs().max(), df_te["pred"].abs().max()) * 1.1
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.axhline(0, color="#30363D"); ax.axvline(0, color="#30363D")
    ax.plot([-lim,lim],[-lim,lim], color="#58A6FF", lw=1, ls="--")
    ax.text(0.05, 0.93, f"r = {m['corr']:.3f}",
            transform=ax.transAxes, color="#E6EDF3", fontsize=10)
    ax.set_xlabel("Prédit"); ax.set_ylabel("Réel")

    # 5. Betas par régime (heatmap)
    ax = fig.add_subplot(gs[3, :2])
    ax.set_title("Coefficients β par régime (HMMR)", fontsize=11)
    betas = res["model"].betas                    # (K, D)
    feat_labels = [c.replace("ret_lag", "r-").replace("vol_", "σ")
                   .replace("mom_", "mom").replace("skew_10","skew") for c in FEATURE_COLS]
    im = ax.imshow(betas, aspect="auto", cmap="RdBu_r",
                   vmin=-np.abs(betas).max(), vmax=np.abs(betas).max())
    ax.set_xticks(range(len(FEATURE_COLS))); ax.set_xticklabels(feat_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(K)); ax.set_yticklabels([NAMES[k] for k in range(K)])
    plt.colorbar(im, ax=ax, label="β")

    # 6. PnL cumulé
    ax = fig.add_subplot(gs[3, 2])
    ax.set_title("PnL cumulé", fontsize=11)
    ax.plot(df_te.index, res["cum_strat"]*100, color="#2ECC71", lw=1.5, label="HMMR")
    ax.plot(df_te.index, res["cum_bh"]*100,    color="#3498DB", lw=1.5, ls="--", label="B&H")
    ax.axhline(0, color="#30363D", lw=1)
    ax.fill_between(df_te.index, res["cum_strat"]*100, 0,
                    where=res["cum_strat"]>0, alpha=0.15, color="#2ECC71")
    ax.legend(fontsize=9); ax.set_ylabel("%")

    # 7. Métriques
    ax = fig.add_subplot(gs[4, :])
    ax.axis("off")
    rows = [
        ["RMSE",            f"{m['rmse']:.6f}"],
        ["MAE",             f"{m['mae']:.6f}"],
        ["Corrélation",     f"{m['corr']:.4f}"],
        ["p-value",         f"{m['pval']:.4f}"],
        ["Direction Acc.",  f"{m['dir_acc']:.1%}"],
        ["Sharpe HMMR",     f"{m['sharpe_strat']:.2f}"],
        ["Sharpe B&H",      f"{m['sharpe_bh']:.2f}"],
    ]
    table = ax.table(cellText=rows, colLabels=["Métrique", "Valeur"],
                     loc="center", cellLoc="center",
                     bbox=[0.3, 0.0, 0.4, 1.0])
    table.auto_set_font_size(False); table.set_fontsize(11)
    for (r, c), cell in table.get_celld().items():
        cell.set_facecolor("#161B22" if r % 2 == 0 else "#21262D")
        cell.set_edgecolor("#30363D")
        cell.set_text_props(color="#C9D1D9")
        if r == 0:
            cell.set_text_props(color="#58A6FF", fontweight="bold")
    table.scale(1.1, 2.0)

    plt.savefig(save, dpi=150, bbox_inches="tight", facecolor="#0D1117")
    print(f"✅  Rapport sauvegardé → {save}")
    plt.close()


def plot_regime_emission(res: dict, save: str = "hmmr_emission.png"):
    """Distributions d'émission réelles vs prédites par régime."""
    model = res["model"]
    K = model.n_states
    df_all = pd.concat([res["df_tr"], res["df_te"]])
    all_st = np.concatenate([res["states_tr"], res["states_te"]])

    fig, axes = plt.subplots(2, K, figsize=(5*K, 9))
    fig.patch.set_facecolor("#0D1117")
    fig.suptitle("HMMR · Émissions par régime", fontsize=14, color="#E6EDF3")

    X_all = res["scaler"].transform(df_all[FEATURE_COLS])
    y_all = df_all["target"].values

    for k in range(K):
        mask = all_st == k
        color = COLORS[k]
        name  = NAMES[k]

        # Distribution réelle
        ax = axes[0, k]
        rets = y_all[mask]
        ax.hist(rets, bins=50, color=color, alpha=0.7, density=True, label="Réel")
        mu_pred = (X_all[mask] @ model.betas[k])
        ax.hist(mu_pred, bins=50, color="white", alpha=0.4, density=True, label="μ prédit")
        ax.set_title(f"{name} — Distribution", color=color, fontweight="bold")
        ax.set_xlabel("Rendement"); ax.legend(fontsize=8)

        # Résidus
        ax = axes[1, k]
        resid = rets - mu_pred
        ax.hist(resid, bins=50, color=color, alpha=0.7, density=True)
        x_range = np.linspace(resid.min(), resid.max(), 200)
        ax.plot(x_range, stats.norm.pdf(x_range, 0, model.sigmas[k]),
                color="white", lw=1.5, label=f"N(0, σ={model.sigmas[k]:.5f})")
        ax.set_title(f"{name} — Résidus", color=color, fontweight="bold")
        ax.legend(fontsize=8)
        info = f"n={mask.sum()}\nμ_β·x={mu_pred.mean():.5f}\nσ={model.sigmas[k]:.5f}"
        ax.text(0.97, 0.95, info, transform=ax.transAxes, ha="right", va="top",
                fontsize=8, color="#C9D1D9",
                bbox=dict(boxstyle="round", facecolor="#21262D", edgecolor="#30363D"))

    plt.tight_layout()
    plt.savefig(save, dpi=150, bbox_inches="tight", facecolor="#0D1117")
    print(f"✅  Émissions sauvegardées → {save}")
    plt.close()


# ─────────────────────────────────────────────────────────
# 6. MAIN
# ─────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  HMMR · Régression par Régime · Rendements 10 min")
    print("=" * 60)

    # Simulation
    print("\n⚙️  Simulation des rendements 10 min...")
    df_raw = simulate_10min_returns(n_bars=2000, n_states=3)

    # Features
    print("\n🔧  Feature engineering...")
    df = build_features(df_raw)
    print(f"   → {len(df)} observations | {len(FEATURE_COLS)} features")

    # Backtest HMMR
    res = walk_forward(df, n_states=3, train_frac=0.70)

    m = res["metrics"]
    print("\n" + "─" * 48)
    print("  MÉTRIQUES HMMR — PÉRIODE TEST")
    print("─" * 48)
    print(f"  RMSE              : {m['rmse']:.7f}")
    print(f"  MAE               : {m['mae']:.7f}")
    print(f"  Corrélation       : {m['corr']:.4f}  (p={m['pval']:.4f})")
    print(f"  Direction Acc.    : {m['dir_acc']:.1%}")
    print(f"  Sharpe HMMR       : {m['sharpe_strat']:.2f}")
    print(f"  Sharpe Buy & Hold : {m['sharpe_bh']:.2f}")

    print("\n  Betas par régime (top features) :")
    betas_df = pd.DataFrame(res["model"].betas,
                            index=[NAMES[k] for k in range(res["n_states"])],
                            columns=FEATURE_COLS)
    print(betas_df.round(6).to_string())
    print("─" * 48)

    # Visualisations
    print("\n🎨  Génération des graphiques...")
    plot_report(res)
    plot_regime_emission(res)

    print("\n✨  Fichiers générés :")
    print("   • hmmr_report.png   (rapport principal)")
    print("   • hmmr_emission.png (émissions par régime)")


if __name__ == "__main__":
    main()
