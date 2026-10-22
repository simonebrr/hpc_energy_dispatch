import os

import numpy as np
import pandas as pd

from utils.renewable_real import RenewableModels
from utils.co2 import CarbonIntensityModels
from utils.battery_model import Battery
from utils.generator_model import Generator
from utils.price_model import PriceModel


BATTERY_CAPACITY = 6_400_000   # Wh
P_MAX_WIND = 400_000           # W
P_MAX_SOLAR = 450_000          # W

BASE_THRESHOLDS = [200_000, 400_000]   # T0, in W

ALPHA_LIST = [0.0, 0.3, 0.5, 1.0, 1.3]

SOLAR_COST_EUR_PER_KWH = 0.05
WIND_COST_EUR_PER_KWH = 0.04
SOLAR_CO2_G_PER_KWH = 50.0
WIND_CO2_G_PER_KWH = 34.0

OUTPUT_FILE = "risultati/soglie_scalate.csv"


def load_data():
  """
  Traccia con prezzi, produzione rinnovabile e intensita' di carbonio.
  """
  df = pd.read_csv("csvs/cluster_power_only_nodes_30days.csv")
  df["time"] = pd.to_datetime(df["time"], utc=True)
  df["dt_hours"] = df["time"].diff().dt.total_seconds().fillna(0) / 3600

  price_model = PriceModel()
  df["price_base"], df["price_high"] = price_model.prices_from_df(df)

  rm = RenewableModels()
  df["P_wind"] = rm.wind_from_openmeteo(df, P_MAX_WIND)
  df["P_solar"] = rm.solar_from_openmeteo(df, P_MAX_SOLAR)
  df["P_ren"] = df["P_wind"] + df["P_solar"]

  cm = CarbonIntensityModels(
      csv_file="csvs/carbon_intensity_IT-NORTH-2020.csv")
  df["co2_intensity"] = cm.co2_from_csv(df)

  return df


def to_matrix(df):
  return df[["power", "P_wind", "P_solar", "P_ren", "dt_hours",
             "price_base", "price_high", "co2_intensity"]].to_numpy()


def new_battery():
  return Battery(
      capacity_wh=BATTERY_CAPACITY,
      initial_charge_wh=BATTERY_CAPACITY / 2,
      max_charge_rate_w=BATTERY_CAPACITY,
      max_discharge_rate_w=BATTERY_CAPACITY,
  )


def new_generator():
  return Generator(
      max_power_w=500_000, min_power_w=50_000, efficiency=0.4,
      fuel_cost_per_wh=0.00025, co2_g_per_kwh=450,
  )


def renewable_cost_co2(P_wind, P_solar, dt):
  """
  Costo e CO2 di tutta l'energia rinnovabile prodotta, anche quella non
  utilizzata, come nel simulatore originale.
  """
  E_wind = P_wind * dt
  E_solar = P_solar * dt
  cost = (E_wind / 1000) * WIND_COST_EUR_PER_KWH + \
         (E_solar / 1000) * SOLAR_COST_EUR_PER_KWH
  co2_g = (E_wind / 1000) * WIND_CO2_G_PER_KWH + \
          (E_solar / 1000) * SOLAR_CO2_G_PER_KWH
  return cost, co2_g


def policy_grid(values, threshold, alpha):
  """
  Tutta la domanda dalla rete.
  """
  total_cost = total_co2_g = 0.0
  E_peak_Wh = E_grid_Wh = 0.0

  for i in range(len(values)):
    (power, _, _, _, dt, cost_low, cost_high, grid_co2_intensity) = values[i]
    if dt <= 0:
      continue

    P_load = power * (1 + alpha)

    E_base = min(P_load, threshold) * dt
    E_peak = max(P_load - threshold, 0.0) * dt

    total_cost += E_base * cost_low + E_peak * cost_high
    total_co2_g += ((E_base + E_peak) / 1000) * grid_co2_intensity

    E_grid_Wh += E_base + E_peak
    E_peak_Wh += E_peak

  return {"costo_euro": total_cost, "co2_kg": total_co2_g / 1000,
          "E_rete_Wh": E_grid_Wh, "E_picco_Wh": E_peak_Wh,
          "E_gen_Wh": 0.0, "curtailment_Wh": 0.0, "n_accensioni_gen": 0}


def policy_batt(values, threshold, alpha):
  """
  Peak shaving: sopra soglia scarica la batteria, sotto soglia la ricarica
  con il margine disponibile.
  """
  battery = new_battery()
  total_cost = total_co2_g = 0.0
  E_peak_Wh = E_grid_Wh = 0.0

  for i in range(len(values)):
    (power, _, _, _, dt, cost_low, cost_high, grid_co2_intensity) = values[i]
    if dt <= 0:
      continue

    P_load = power * (1 + alpha)
    E_load = P_load * dt

    if P_load > threshold:
      P_discharge = min(P_load - threshold, threshold)
      E_from_batt = battery.discharge(P_discharge, dt)

      E_peak = max(P_load - threshold, 0.0) * dt
      E_peak_residual = max(E_peak - E_from_batt, 0.0)
      E_base = threshold * dt
      E_grid = E_base + E_peak_residual

      cost = E_base * cost_low + E_peak_residual * cost_high
      co2_g = (E_grid / 1000) * grid_co2_intensity
      cost += battery.step_cost(dt, E_from_batt)
      co2_g += battery.step_co2_g(dt, E_from_batt)

      E_peak_Wh += E_peak_residual
    else:
      P_charge = min(threshold - P_load, threshold)
      E_charged = battery.charge(P_charge, dt)
      E_grid = E_load + E_charged

      cost = E_grid * cost_low
      co2_g = (E_grid / 1000) * grid_co2_intensity
      cost += battery.step_cost(dt, E_charged)
      co2_g += battery.step_co2_g(dt, E_charged)

    total_cost += cost
    total_co2_g += co2_g
    E_grid_Wh += E_grid
    battery.step(dt)

  return {"costo_euro": total_cost, "co2_kg": total_co2_g / 1000,
          "E_rete_Wh": E_grid_Wh, "E_picco_Wh": E_peak_Wh,
          "E_gen_Wh": 0.0, "curtailment_Wh": 0.0, "n_accensioni_gen": 0}


def policy_batt_green(values, threshold, alpha):
  """
  Rinnovabili, poi batteria, poi rete; il surplus carica la batteria.
  """
  battery = new_battery()
  total_cost = total_co2_g = 0.0
  E_peak_Wh = E_grid_Wh = curtail_Wh = 0.0

  for i in range(len(values)):
    (power, P_wind, P_solar, P_ren, dt,
     cost_low, cost_high, grid_co2_intensity) = values[i]
    if dt <= 0:
      continue

    P_load = power * (1 + alpha)
    cost_green, co2_green_g = renewable_cost_co2(P_wind, P_solar, dt)

    E_charged = E_from_batt = 0.0
    cost_grid = co2_grid_g = 0.0

    P_from_ren = min(P_load, P_ren)
    P_excess = max(P_ren - P_from_ren, 0.0)

    if P_excess > 0:
      E_charged = battery.charge(P_excess, dt)
      curtail_Wh += max(P_excess * dt - E_charged, 0.0)
    else:
      P_residual = P_load - P_from_ren
      E_from_batt = battery.discharge(P_residual, dt)
      P_from_grid = max(P_residual * dt - E_from_batt, 0.0) / dt

      E_base = min(P_from_grid, threshold) * dt
      E_peak = max(P_from_grid - threshold, 0.0) * dt

      cost_grid = E_base * cost_low + E_peak * cost_high
      co2_grid_g = ((E_base + E_peak) / 1000) * grid_co2_intensity

      E_grid_Wh += E_base + E_peak
      E_peak_Wh += E_peak

    throughput = E_charged + E_from_batt
    total_cost += cost_green + cost_grid + battery.step_cost(dt, throughput)
    total_co2_g += co2_green_g + co2_grid_g + battery.step_co2_g(dt, throughput)
    battery.step(dt)

  return {"costo_euro": total_cost, "co2_kg": total_co2_g / 1000,
          "E_rete_Wh": E_grid_Wh, "E_picco_Wh": E_peak_Wh,
          "E_gen_Wh": 0.0, "curtailment_Wh": curtail_Wh,
          "n_accensioni_gen": 0}


def policy_all(values, threshold, alpha):
  """
  Come Batt+Green, ma il prelievo oltre soglia e' coperto dal generatore.
  """
  battery = new_battery()
  generator = new_generator()

  total_cost = total_co2_g = 0.0
  E_peak_Wh = E_grid_Wh = E_gen_Wh = curtail_Wh = 0.0
  n_starts = 0
  gen_on_prev = False

  for i in range(len(values)):
    (power, P_wind, P_solar, P_ren, dt,
     cost_low, cost_high, grid_co2_intensity) = values[i]
    if dt <= 0:
      continue

    P_load = power * (1 + alpha)
    cost_green, co2_green_g = renewable_cost_co2(P_wind, P_solar, dt)

    E_charged = E_from_batt = 0.0
    cost_grid = cost_gen = co2_grid_g = co2_gen_g = 0.0
    P_gen = 0.0

    P_from_ren = min(P_load, P_ren)
    P_excess = max(P_ren - P_from_ren, 0.0)

    if P_excess > 0:
      E_charged = battery.charge(P_excess, dt)
      curtail_Wh += max(P_excess * dt - E_charged, 0.0)
    else:
      P_residual = P_load - P_from_ren
      E_from_batt = battery.discharge(P_residual, dt)
      P_after_batt = max(P_residual * dt - E_from_batt, 0.0) / dt

      # sotto soglia la rete, sopra soglia il generatore
      P_base = min(P_after_batt, threshold)
      P_over = max(P_after_batt - threshold, 0.0)
      P_gen = min(P_over, generator.max_power_w)
      P_grid_peak = P_over - P_gen

      E_base = P_base * dt
      E_peak = P_grid_peak * dt
      E_from_gen = P_gen * dt

      cost_grid = E_base * cost_low + E_peak * cost_high
      co2_grid_g = ((E_base + E_peak) / 1000) * grid_co2_intensity
      cost_gen = generator.get_cost(E_from_gen)
      co2_gen_g = generator.get_co2(E_from_gen / 1000)

      E_grid_Wh += E_base + E_peak
      E_peak_Wh += E_peak
      E_gen_Wh += E_from_gen

    # avviamenti: aggiornati a ogni passo, anche durante il surplus
    gen_on = P_gen > 0
    if gen_on and not gen_on_prev:
      n_starts += 1
    gen_on_prev = gen_on

    throughput = E_charged + E_from_batt
    total_cost += (cost_green + cost_grid + cost_gen
                   + battery.step_cost(dt, throughput))
    total_co2_g += (co2_green_g + co2_grid_g + co2_gen_g
                    + battery.step_co2_g(dt, throughput))
    battery.step(dt)

  return {"costo_euro": total_cost, "co2_kg": total_co2_g / 1000,
          "E_rete_Wh": E_grid_Wh, "E_picco_Wh": E_peak_Wh,
          "E_gen_Wh": E_gen_Wh, "curtailment_Wh": curtail_Wh,
          "n_accensioni_gen": n_starts}


POLICIES = {
    "Grid": policy_grid,
    "Batt": policy_batt,
    "Batt+Green": policy_batt_green,
    "All": policy_all,
}


def run_all(values):
  """
  Simula ogni politica per ogni T0 e alpha, con soglia fissa (T0) e
  scalata (T0 * (1 + alpha)).
  """
  rows = []

  for name, func in POLICIES.items():
    for T0 in BASE_THRESHOLDS:
      for alpha in ALPHA_LIST:
        for regime in ["fissa", "scalata"]:
          # per alpha = 0 i due regimi coincidono
          if alpha == 0.0 and regime == "scalata":
            baseline = [r for r in rows
                        if r["policy"] == name and r["T0_W"] == T0
                        and r["alpha"] == 0.0][0]
            row = dict(baseline, regime="scalata")
            rows.append(row)
            continue

          threshold = T0 * (1 + alpha) if regime == "scalata" else T0
          r = func(values, threshold, alpha)
          r.update(policy=name, T0_W=T0, alpha=alpha,
                   regime=regime, soglia_W=threshold)
          rows.append(r)

          print(f"{name:<11} T0 {T0/1000:>3.0f} kW  "
                f"alpha {alpha:<4} {regime:<8} "
                f"soglia {threshold/1000:>5.0f} kW -> "
                f"{r['costo_euro']:>9.0f} EUR  "
                f"{r['co2_kg']:>8.0f} kg")

  return pd.DataFrame(rows)


def decompose(table, quantity="costo_euro"):
  """
  Scompone l'aumento rispetto ad alpha = 0:

    delta_totale  = C(alpha, T0)          - C(0, T0)
    delta_consumo = C(alpha, (1+alpha)T0) - C(0, T0)
    delta_vincolo = C(alpha, T0)          - C(alpha, (1+alpha)T0)
  """
  rows = []

  for name in POLICIES:
    for T0 in BASE_THRESHOLDS:
      sel = table[(table.policy == name) & (table.T0_W == T0)]
      baseline = sel[(sel.alpha == 0.0) & (sel.regime == "fissa")]
      if baseline.empty:
        continue
      C0 = float(baseline[quantity].iloc[0])

      for alpha in ALPHA_LIST:
        if alpha == 0.0:
          continue
        fixed = sel[(sel.alpha == alpha) & (sel.regime == "fissa")]
        scaled = sel[(sel.alpha == alpha) & (sel.regime == "scalata")]
        if fixed.empty or scaled.empty:
          continue

        C_fixed = float(fixed[quantity].iloc[0])
        C_scaled = float(scaled[quantity].iloc[0])

        d_tot = C_fixed - C0
        d_consumption = C_scaled - C0
        d_constraint = C_fixed - C_scaled

        rows.append({
            "policy": name,
            "T0_kW": T0 / 1000,
            "alpha": alpha,
            "grandezza": quantity,
            "base": round(C0, 1),
            "con_alpha": round(C_fixed, 1),
            "delta_totale": round(d_tot, 1),
            "delta_consumo": round(d_consumption, 1),
            "delta_vincolo": round(d_constraint, 1),
            "quota_consumo_%": round(100 * d_consumption / d_tot, 1)
                                if d_tot else np.nan,
            "quota_vincolo_%": round(100 * d_constraint / d_tot, 1)
                                if d_tot else np.nan,
            "residuo": round(d_tot - d_consumption - d_constraint, 6),
        })

  return pd.DataFrame(rows)


if __name__ == "__main__":
  df = load_data()
  values = to_matrix(df)

  print(f"Traccia: {len(df)} campioni, "
        f"{df['dt_hours'].sum():.1f} ore\n")

  table = run_all(values)

  # Grid non ha componenti di taglia fissa: nel regime scalato il costo
  # deve crescere esattamente di (1 + alpha)
  print("\nControllo Grid, regime scalato: C(alpha,(1+alpha)T0) / C(0,T0)")
  grid = table[table.policy == "Grid"]
  for T0 in BASE_THRESHOLDS:
    C0 = float(grid[(grid.T0_W == T0) & (grid.alpha == 0.0)
                    & (grid.regime == "fissa")]["costo_euro"].iloc[0])
    for alpha in ALPHA_LIST:
      if alpha == 0.0:
        continue
      C = float(grid[(grid.T0_W == T0) & (grid.alpha == alpha)
                     & (grid.regime == "scalata")]["costo_euro"].iloc[0])
      deviation = C / C0 - (1 + alpha)
      outcome = "OK" if abs(deviation) < 1e-9 else "ANOMALIA"
      print(f"  T0 {T0/1000:>3.0f} kW  alpha {alpha:<4} "
            f"rapporto {C / C0:.9f}  atteso {1 + alpha:.1f}  [{outcome}]")

  columns = ["policy", "T0_W", "alpha", "regime", "soglia_W",
             "costo_euro", "co2_kg", "E_rete_Wh", "E_picco_Wh",
             "E_gen_Wh", "curtailment_Wh", "n_accensioni_gen"]
  os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
  table[columns].to_csv(OUTPUT_FILE, index=False)
  print(f"\nSalvato in {OUTPUT_FILE}  ({len(table)} righe)")

  for quantity in ["costo_euro", "co2_kg"]:
    decomposition = decompose(table, quantity)
    path = OUTPUT_FILE.replace(".csv", f"_scomposizione_{quantity}.csv")
    decomposition.to_csv(path, index=False)

    print(f"\n=== Scomposizione di {quantity} ===")
    print(decomposition.to_string(index=False))

    max_residual = decomposition["residuo"].abs().max()
    print(f"residuo massimo: {max_residual:.2e}  "
          f"({'OK' if max_residual < 1e-6 else 'ANOMALIA'})")
    print(f"salvato in {path}")
