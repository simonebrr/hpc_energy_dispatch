import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pandas as pd

FIG_DIR = "figure"
TRACE = "csvs/cluster_power_only_nodes_30days.csv"
RESULTS_DIR = "cooling/risultati"

# configurazione di riferimento
THRESHOLD_W = 200_000
BATTERY_WH = 6_400_000
P_WIND_W = 400_000
P_SOLAR_W = 450_000

POLICIES = ["Grid", "Batt", "Batt+Green", "All"]

plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 10,
    "figure.dpi": 150,
    "hatch.linewidth": 0.5,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "savefig.bbox": "tight",
})

# virgola come separatore decimale sugli assi
COMMA_FMT = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ","))


def alpha_label(alpha):
  return rf"$\alpha = {alpha}$".replace(".", "{,}")


def _save(fig, name):
  fig.savefig(f"{FIG_DIR}/{name}.pdf")
  plt.close(fig)
  print(f"  figure/{name}.pdf")


def _normalize_policy(name):
  """
  Uniforma i nomi delle politiche ('BATT+GREEN' -> 'Batt+Green').
  """
  name = str(name).strip().lower()
  return {"grid": "Grid", "batt": "Batt", "batt+green": "Batt+Green",
          "all": "All"}.get(name, name)


def _first_column(df, *candidates):
  """
  Prima colonna presente fra le candidate: i CSV usano nomi diversi.
  """
  for c in candidates:
    if c in df.columns:
      return c
  return None


def load_results():
  """
  Risultati delle quattro politiche (cooling/risultati/*.csv) per la
  configurazione di riferimento.
  """
  parts = []
  for f in sorted(os.listdir(RESULTS_DIR)):
    if not f.endswith(".csv"):
      continue
    d = pd.read_csv(os.path.join(RESULTS_DIR, f))
    if "policy" not in d.columns or "alpha" not in d.columns:
      continue
    d["policy"] = d["policy"].map(_normalize_policy)

    # Grid non ha batteria ne' rinnovabili: colonne vuote
    m = d["threshold"] == THRESHOLD_W
    if "battery_capacity" in d.columns and d["battery_capacity"].notna().any():
      m &= d["battery_capacity"] == BATTERY_WH
    if "pmaxwind" in d.columns and d["pmaxwind"].notna().any():
      m &= (d["pmaxwind"] == P_WIND_W) & (d["pmaxsun"] == P_SOLAR_W)
    d = d[m]

    c_curt = _first_column(d, "sum_curtailment_Wh", "sum_curtailment")
    c_green = _first_column(d, "green_energy_total_Wh", "E_green_total_Wh")

    parts.append(pd.DataFrame({
        "policy": d["policy"],
        "alpha": d["alpha"].round(3),
        "costo_euro": d["costo_euro"],
        "co2_kg": d["co2_kg"],
        "curtailment_Wh": d[c_curt] if c_curt else float("nan"),
        "E_green_Wh": d[c_green] if c_green else float("nan"),
    }))

  return pd.concat(parts, ignore_index=True)


def load_energy_Wh(alpha):
  """
  Energia richiesta dal carico sull'orizzonte, compreso il raffreddamento.
  """
  df = pd.read_csv(TRACE)
  df["time"] = pd.to_datetime(df["time"], utc=True)
  dt = df["time"].diff().dt.total_seconds().fillna(0) / 3600
  return float((df["power"] * dt).sum()) * (1 + alpha)


def figure_1():
  """
  Traccia oraria con e senza raffreddamento, con le due soglie di rete.
  """
  df = pd.read_csv(TRACE)
  df["time"] = pd.to_datetime(df["time"], utc=True)

  # media oraria; il sovraccarico e' moltiplicativo, quindi si puo' scalare
  # direttamente la media
  s = df.set_index("time")["power"].resample("1h").mean() / 1000.0

  fig, ax = plt.subplots(figsize=(7, 3.2))

  ax.plot(s.index, s.values, linewidth=0.8, color="0.15",
          label=r"$\alpha = 0$ (solo nodi)")
  ax.plot(s.index, s.values * 1.5, linewidth=0.8, color="0.55",
          linestyle="--", label=alpha_label(0.5))

  ax.axhline(200, color="black", linestyle=":", linewidth=1.2)
  ax.axhline(400, color="black", linestyle="-.", linewidth=1.2)
  ax.text(s.index[5], 210, "200 kW", fontsize=8)
  ax.text(s.index[5], 410, "400 kW", fontsize=8)

  ax.set_xlabel("Tempo")
  ax.set_ylabel("Potenza [kW]")
  ax.legend(loc="upper right", framealpha=0.9)
  fig.autofmt_xdate()
  _save(fig, "traccia_cooling")


def figure_2(results):
  """
  Costo ed emissioni delle quattro politiche al variare di alpha.
  """
  alphas = sorted(results["alpha"].unique())
  costs, co2 = {}, {}
  for a in alphas:
    sub = results[results["alpha"] == a].set_index("policy")
    costs[a] = [sub.loc[p, "costo_euro"] for p in POLICIES]
    co2[a] = [sub.loc[p, "co2_kg"] for p in POLICIES]

  # stili distinguibili in bianco e nero
  styles = [("white", "///"), ("0.75", "\\\\\\"), ("0.45", ""),
            ("white", "xxx"), ("0.20", "...")]

  fig_width = 7.5 if len(alphas) <= 4 else 9.0
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(fig_width, 3.0))
  width = 0.82 / len(alphas)
  positions = range(len(POLICIES))

  for i, alpha in enumerate(alphas):
    fill, hatch = styles[i % len(styles)]
    offset = (i - (len(alphas) - 1) / 2) * width
    common = dict(color=fill, hatch=hatch, edgecolor="black",
                  linewidth=0.6, label=alpha_label(alpha))
    ax1.bar([p + offset for p in positions], costs[alpha],
            width, **common)
    ax2.bar([p + offset for p in positions], co2[alpha],
            width, **common)

  for ax, y_label in [(ax1, "Costo totale [EUR]"),
                      (ax2, r"Emissioni di CO$_2$ [kg]")]:
    ax.set_xticks(list(positions))
    ax.set_xticklabels(POLICIES, rotation=20, ha="right")
    ax.set_ylabel(y_label)
    ax.yaxis.set_major_formatter(COMMA_FMT)

  ax1.legend(fontsize=8)
  fig.tight_layout()
  _save(fig, "confronto_politiche")


def figure_3(results):
  """
  Curtailment e quota di domanda coperta da rinnovabili al variare di alpha.
  """
  bg = results[(results["policy"] == "Batt+Green")
               & results["curtailment_Wh"].notna()
               & results["E_green_Wh"].notna()].sort_values("alpha")
  alpha = bg["alpha"].tolist()
  curtailment = (bg["curtailment_Wh"] / 1e6).tolist()   # MWh
  # quota rinnovabile = (produzione lorda - curtailment) / energia del carico
  green_share = [
      100 * (g - c) / load_energy_Wh(a)
      for a, g, c in zip(alpha, bg["E_green_Wh"], bg["curtailment_Wh"])
  ]

  # linea di raccordo solo con piu' di cinque punti
  line_style = "-" if len(alpha) > 5 else "none"

  fig, ax1 = plt.subplots(figsize=(5.5, 3.2))

  ax1.plot(alpha, curtailment, marker="o", color="0.15",
           linestyle=line_style, linewidth=0.9, markersize=5,
           label="Energia rinnovabile non utilizzata")
  ax1.set_xlabel(r"Sovraccarico di raffreddamento $\alpha$")
  ax1.set_ylabel("Curtailment [MWh]")
  ax1.set_ylim(0, max(curtailment) * 1.15)
  ax1.xaxis.set_major_formatter(COMMA_FMT)
  ax1.yaxis.set_major_formatter(COMMA_FMT)

  # evidenzia il primo alpha con curtailment nullo
  zeros = [i for i, c in enumerate(curtailment) if c < 1e-9]
  if zeros:
    i_sat = zeros[0]
    ax1.annotate("curtailment nullo",
                 xy=(alpha[i_sat], 0), xytext=(alpha[i_sat] - 0.42, 5.0),
                 fontsize=7.5, color="0.25",
                 arrowprops=dict(arrowstyle="->", color="0.4",
                                 linewidth=0.7))

  ax2 = ax1.twinx()
  ax2.plot(alpha, green_share, marker="s", color="0.55",
           linestyle=line_style, linewidth=0.9, markersize=5,
           label="Quota di domanda da rinnovabili")
  ax2.set_ylabel("Quota rinnovabile [%]")
  ax2.set_ylim(min(green_share) - 8, 100)
  ax2.yaxis.set_major_formatter(COMMA_FMT)
  ax2.grid(False)

  lines = ax1.get_lines() + ax2.get_lines()
  ax1.legend(lines, [l.get_label() for l in lines],
             fontsize=8, loc="upper center")

  fig.tight_layout()
  _save(fig, "curtailment_quota")


if __name__ == "__main__":
  os.makedirs(FIG_DIR, exist_ok=True)
  results = load_results()
  print("Genero le figure:")
  figure_1()
  figure_2(results)
  figure_3(results)
  print("\nFatto.")
