import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch

FIG_DIR = "figure"

plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 9,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
})

# (colore di riempimento, stile del bordo)
BASE = ("white", "solid")
MODIFIED = ("white", (0, (4, 2)))
NEW = ("0.85", "solid")


def box(ax, x, y, w, h, text, style, size=8.0):
  fill, edge = style
  ax.add_patch(FancyBboxPatch(
      (x, y), w, h,
      boxstyle="round,pad=0.004,rounding_size=0.012",
      facecolor=fill, edgecolor="black",
      linewidth=1.0, linestyle=edge, zorder=2))
  ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
          fontsize=size, linespacing=1.35, zorder=3)


def group_box(ax, x, y, w, h, title):
  ax.add_patch(FancyBboxPatch(
      (x, y), w, h,
      boxstyle="round,pad=0.004,rounding_size=0.01",
      facecolor="none", edgecolor="0.55",
      linewidth=0.7, linestyle=(0, (2, 2)), zorder=1))
  ax.text(x + 0.012, y + h - 0.022, title, ha="left", va="center",
          fontsize=8.2, style="italic", color="0.35", zorder=3)


def main():
  fig, ax = plt.subplots(figsize=(7.2, 4.6))
  ax.set_xlim(0, 1)
  ax.set_ylim(0, 1)
  ax.axis("off")

  # riga 1: dati di ingresso
  group_box(ax, 0.02, 0.76, 0.96, 0.20, "Dati di ingresso")
  box(ax, 0.05, 0.795, 0.28, 0.10,
      "traccia PM100\n(nodi di calcolo)", BASE)
  box(ax, 0.36, 0.795, 0.28, 0.10,
      "Open-Meteo\n(vento, irraggiamento)", BASE)
  box(ax, 0.67, 0.795, 0.28, 0.10,
      "ENTSO-E\n(intensità di carbonio)", BASE)

  # riga 2: modelli e politiche
  group_box(ax, 0.02, 0.40, 0.46, 0.32, "Modelli (pacchetto utils/)")
  box(ax, 0.05, 0.575, 0.19, 0.09, "battery\n_model.py", BASE, 7.6)
  box(ax, 0.26, 0.575, 0.19, 0.09, "generator\n_model.py", BASE, 7.6)
  box(ax, 0.05, 0.445, 0.19, 0.09, "renewable\n_real.py", BASE, 7.6)
  box(ax, 0.26, 0.445, 0.19, 0.09, "price_model.py\nco2.py", BASE, 7.6)

  group_box(ax, 0.52, 0.40, 0.46, 0.32, "Politiche di dispatch")
  box(ax, 0.55, 0.575, 0.40, 0.09,
      "Grid   Batt   Batt+Green   All\n"
      r"+ sovraccarico di raffreddamento $\alpha$",
      MODIFIED, 7.6)
  box(ax, 0.55, 0.425, 0.40, 0.115,
      "sim_soglia_dinamica.py\nsim_isteresi.py\nsim_soglie_scalate.py",
      NEW, 7.6)

  # riga 3: vincoli e uscite
  group_box(ax, 0.02, 0.12, 0.46, 0.24, "Generazione dei vincoli")
  box(ax, 0.05, 0.155, 0.40, 0.145,
      "eventi.py\n"
      r"profili di soglia $P^{cap}(t)$:" "\n"
      "costante, singolo, ricorrente, stocastico",
      NEW, 7.6)

  group_box(ax, 0.52, 0.12, 0.46, 0.24, "Uscite")
  box(ax, 0.55, 0.235, 0.40, 0.065,
      r"costo, CO$_2$, curtailment", BASE, 7.6)
  box(ax, 0.55, 0.155, 0.40, 0.065,
      "metriche di violazione, avviamenti", NEW, 7.6)

  # legenda
  handles = [
      Patch(facecolor="white", edgecolor="black", linewidth=1.0,
            label="preesistente"),
      Patch(facecolor="white", edgecolor="black", linewidth=1.0,
            linestyle=(0, (4, 2)), label="preesistente, modificato"),
      Patch(facecolor="0.85", edgecolor="black", linewidth=1.0,
            label="sviluppato in questa tesi"),
  ]
  ax.legend(handles=handles, loc="lower center", ncol=3, fontsize=8,
            frameon=False, bbox_to_anchor=(0.5, -0.02))

  os.makedirs(FIG_DIR, exist_ok=True)
  fig.savefig(f"{FIG_DIR}/architettura.pdf")
  fig.savefig(f"{FIG_DIR}/architettura.png")
  plt.close(fig)
  print(f"  {FIG_DIR}/architettura.pdf")


if __name__ == "__main__":
  main()
