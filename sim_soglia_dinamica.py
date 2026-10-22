import os

import numpy as np
import pandas as pd

from utils.renewable_real import RenewableModels
from utils.co2 import CarbonIntensityModels
from utils.battery_model import Battery
from utils.generator_model import Generator
from utils.price_model import PriceModel

from eventi import (constant_threshold, single_event,
                    recurring_events, stochastic_events)


P_TARIFF = 400_000            # W
BATTERY_CAPACITY = 6_400_000  # Wh
P_MAX_WIND = 400_000          # W
P_MAX_SOLAR = 450_000         # W

ALPHA_LIST = [0.0, 0.5]      # sovraccarico di raffreddamento
N_SEEDS = 15                   # ripetizioni del processo stocastico

SOLAR_COST_EUR_PER_KWH = 0.05
WIND_COST_EUR_PER_KWH = 0.04
SOLAR_CO2_G_PER_KWH = 50.0
WIND_CO2_G_PER_KWH = 34.0

OUTPUT_FILE = "risultati/soglia_dinamica.csv"


df = pd.read_csv("csvs/cluster_power_only_nodes_30days.csv")
df["time"] = pd.to_datetime(df["time"], utc=True)
df["dt_hours"] = df["time"].diff().dt.total_seconds().fillna(0) / 3600

price_model = PriceModel()
df["price_base"], df["price_high"] = price_model.prices_from_df(df)

rm = RenewableModels()
df["P_wind"] = rm.wind_from_openmeteo(df, P_MAX_WIND)
df["P_solar"] = rm.solar_from_openmeteo(df, P_MAX_SOLAR)
df["P_ren"] = df["P_wind"] + df["P_solar"]

cm = CarbonIntensityModels(csv_file="csvs/carbon_intensity_IT-NORTH-2020.csv")
df["co2_intensity"] = cm.co2_from_csv(df)


SCENARIOS = {
    "costante":
          constant_threshold(df, P_TARIFF),
    "evento_singolo":
          single_event(df, P_TARIFF, "2020-06-05 14:00", 6, 150_000),
    "ricorrente":
          recurring_events(df, P_TARIFF, 18, 21, 200_000),
}

for seed in range(N_SEEDS):
  SCENARIOS[f"stocastico_s{seed:02d}"] = stochastic_events(
      df, P_TARIFF, seed=seed)


def simulate(cap, alpha):
  """
  Politica All con limite operativo cap(t) variabile.

  La tariffa si applica al confine contrattuale P_TARIFF; il generatore copre
  il prelievo oltre cap(t) e, se non basta, l'eccedenza e' contata come
  violazione.
  """
  battery = Battery(
      capacity_wh=BATTERY_CAPACITY,
      initial_charge_wh=BATTERY_CAPACITY / 2,
      max_charge_rate_w=BATTERY_CAPACITY,
      max_discharge_rate_w=BATTERY_CAPACITY,
  )
  generator = Generator(
      max_power_w=500_000, min_power_w=50_000, efficiency=0.4,
      fuel_cost_per_wh=0.00025, co2_g_per_kwh=450,
  )

  total_cost = total_co2_g = 0.0
  sum_gen_Wh = sum_grid_Wh = sum_batt_Wh = sum_curtail_Wh = 0.0

  E_violation_Wh = 0.0
  n_violation_steps = 0
  max_violation_W = 0.0
  violation_hours = 0.0
  n_starts = 0
  gen_on_prev = False

  values = df[["power", "P_ren", "P_wind", "P_solar", "dt_hours",
               "price_base", "price_high", "co2_intensity"]].to_numpy()

  for i in range(len(values)):
    (power, P_ren, P_wind, P_solar, dt,
     cost_low, cost_high, grid_co2_intensity) = values[i]

    if dt <= 0:
      continue

    cap_t = cap[i]
    P_load = power * (1 + alpha)

    E_wind = P_wind * dt
    E_solar = P_solar * dt
    cost_green = (E_wind / 1000) * WIND_COST_EUR_PER_KWH + \
                 (E_solar / 1000) * SOLAR_COST_EUR_PER_KWH
    co2_green_g = (E_wind / 1000) * WIND_CO2_G_PER_KWH + \
                  (E_solar / 1000) * SOLAR_CO2_G_PER_KWH

    E_charged = E_from_batt = 0.0
    cost_grid = cost_gen = co2_grid_g = co2_gen_g = 0.0
    P_gen = 0.0

    P_from_ren = min(P_load, P_ren)
    P_excess_ren = max(P_ren - P_from_ren, 0.0)

    if P_excess_ren > 0:
      # surplus: carica la batteria, il resto e' curtailment
      E_charged = battery.charge(P_excess_ren, dt)
      sum_curtail_Wh += max(P_excess_ren * dt - E_charged, 0.0)

    else:
      # deficit: batteria, poi rete e generatore
      P_residual = P_load - P_from_ren
      E_from_batt = battery.discharge(P_residual, dt)
      P_after_batt = max(P_residual * dt - E_from_batt, 0.0) / dt

      # il generatore copre la parte oltre il limite operativo
      P_over_cap = max(P_after_batt - cap_t, 0.0)
      P_gen = min(P_over_cap, generator.max_power_w)
      P_grid = P_after_batt - P_gen

      P_violation = max(P_grid - cap_t, 0.0)
      if P_violation > 0:
        E_violation_Wh += P_violation * dt
        n_violation_steps += 1
        violation_hours += dt
        max_violation_W = max(max_violation_W, P_violation)

      # la tariffa usa il confine contrattuale, non il limite operativo
      E_base = min(P_grid, P_TARIFF) * dt
      E_peak = max(P_grid - P_TARIFF, 0.0) * dt

      cost_grid = E_base * cost_low + E_peak * cost_high
      co2_grid_g = ((E_base + E_peak) / 1000) * grid_co2_intensity

      E_from_gen = P_gen * dt
      cost_gen = generator.get_cost(E_from_gen)
      co2_gen_g = generator.get_co2(E_from_gen / 1000)

      sum_grid_Wh += E_base + E_peak
      sum_gen_Wh += E_from_gen

    # avviamenti: aggiornati a ogni passo, anche durante il surplus
    gen_on = P_gen > 0
    if gen_on and not gen_on_prev:
      n_starts += 1
    gen_on_prev = gen_on

    throughput = E_charged + E_from_batt
    cost_batt = battery.step_cost(dt, throughput)
    co2_batt_g = battery.step_co2_g(dt, throughput)
    sum_batt_Wh += E_from_batt

    total_cost += cost_green + cost_grid + cost_gen + cost_batt
    total_co2_g += co2_green_g + co2_grid_g + co2_gen_g + co2_batt_g

    battery.step(dt)

  total_hours = df["dt_hours"].sum()

  return {
      "costo_euro": round(total_cost, 3),
      "co2_kg": round(total_co2_g / 1000, 3),
      "E_da_rete_Wh": round(sum_grid_Wh, 1),
      "E_da_generatore_Wh": round(sum_gen_Wh, 1),
      "E_da_batteria_Wh": round(sum_batt_Wh, 1),
      "curtailment_Wh": round(sum_curtail_Wh, 1),
      "E_violazione_Wh": round(E_violation_Wh, 1),
      "n_step_violazione": n_violation_steps,
      "ore_violazione": round(violation_hours, 2),
      "perc_tempo_violazione": round(100 * violation_hours / total_hours, 3),
      "max_violazione_kW": round(max_violation_W / 1000, 1),
      "n_accensioni_generatore": n_starts,
      "SOC_finale": round(battery.info()["SOC"], 3),
  }


if __name__ == "__main__":
  results = []

  for name, cap in SCENARIOS.items():
    for alpha in ALPHA_LIST:
      r = simulate(cap, alpha)
      r["scenario"] = name
      r["alpha"] = alpha
      r["cap_minimo_kW"] = round(float(np.min(cap)) / 1000, 1)
      results.append(r)

      print(f"{name:<16} alpha {alpha:<4} -> "
            f"{r['costo_euro']:>10.0f} EUR  "
            f"{r['co2_kg']:>9.0f} kg  "
            f"violazioni {r['ore_violazione']:>6.1f} h  "
            f"({r['perc_tempo_violazione']:>5.2f}%)  "
            f"gen {r['n_accensioni_generatore']:>4} avvii")

  columns = ["scenario", "alpha", "cap_minimo_kW", "costo_euro", "co2_kg",
             "E_violazione_Wh", "n_step_violazione", "ore_violazione",
             "perc_tempo_violazione", "max_violazione_kW",
             "n_accensioni_generatore", "E_da_rete_Wh",
             "E_da_generatore_Wh", "E_da_batteria_Wh", "curtailment_Wh",
             "SOC_finale"]

  os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
  pd.DataFrame(results)[columns].to_csv(OUTPUT_FILE, index=False)
  print(f"\nSalvato in {OUTPUT_FILE}  ({len(results)} righe)")
