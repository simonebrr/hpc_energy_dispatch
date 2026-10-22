import os

import pandas as pd

from utils.renewable_real import RenewableModels
from utils.co2 import CarbonIntensityModels
from utils.battery_model import Battery
from utils.generator_model import Generator
from utils.price_model import PriceModel

from eventi import constant_threshold, single_event, recurring_events


P_TARIFF = 400_000            # W
BATTERY_CAPACITY = 6_400_000  # Wh
P_MAX_WIND = 400_000          # W
P_MAX_SOLAR = 450_000         # W

GEN_P_MAX = 500_000           # W
GEN_P_MIN = 50_000            # W

ALPHA_LIST = [0.0, 0.5]

SOLAR_COST_EUR_PER_KWH = 0.05
WIND_COST_EUR_PER_KWH = 0.04
SOLAR_CO2_G_PER_KWH = 50.0
WIND_CO2_G_PER_KWH = 34.0

# (label, delta [W], tau_on [h], tau_off [h], enforce_p_min)
CONTROLS = [
    ("reattivo",        0,       0.0,       0.0,       False),
    ("isteresi_25kW",   25_000,  0.0,       0.0,       True),
    ("isteresi_50kW",   50_000,  0.0,       0.0,       True),
    ("min_15min",       0,       0.25,      0.25,      True),
    ("completo",        50_000,  0.25,      0.25,      True),
    ("completo_30min",  50_000,  0.50,      0.50,      True),
    ("reattivo_pmin",   0,       0.0,       0.0,       True),
]

OUTPUT_FILE = "risultati/isteresi.csv"


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
    "costante":       constant_threshold(df, P_TARIFF),
    "evento_singolo": single_event(df, P_TARIFF,
                                   "2020-06-05 14:00", 6, 150_000),
    "ricorrente":     recurring_events(df, P_TARIFF, 18, 21, 200_000),
}


def simulate(cap, alpha, delta=0.0, tau_on=0.0, tau_off=0.0,
             enforce_p_min=True):
  """
  Politica All con controllo vincolato del generatore.

  delta           banda di isteresi [W]: accende sopra cap, spegne sotto cap - delta
  tau_on          tempo minimo di funzionamento [h]
  tau_off         tempo minimo di arresto [h]
  enforce_p_min   se acceso eroga almeno GEN_P_MIN; l'eccedenza carica la batteria
  """
  battery = Battery(
      capacity_wh=BATTERY_CAPACITY,
      initial_charge_wh=BATTERY_CAPACITY / 2,
      max_charge_rate_w=BATTERY_CAPACITY,
      max_discharge_rate_w=BATTERY_CAPACITY,
  )
  generator = Generator(
      max_power_w=GEN_P_MAX, min_power_w=GEN_P_MIN, efficiency=0.4,
      fuel_cost_per_wh=0.00025, co2_g_per_kwh=450,
  )

  total_cost = total_co2_g = 0.0
  sum_gen_Wh = sum_grid_Wh = sum_batt_Wh = sum_curtail_Wh = 0.0

  E_violation_Wh = 0.0
  violation_hours = 0.0
  max_violation_W = 0.0

  # stato del generatore, conservato fra un passo e il successivo
  gen_on = False
  hours_in_state = 0.0
  n_starts = 0
  gen_on_hours = 0.0
  E_gen_forced_Wh = 0.0     # energia erogata oltre il necessario

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

    # 1. potenza scoperta dopo rinnovabili e batteria
    if P_excess_ren > 0:
      E_charged = battery.charge(P_excess_ren, dt)
      sum_curtail_Wh += max(P_excess_ren * dt - E_charged, 0.0)
      P_after_batt = 0.0
    else:
      P_residual = P_load - P_from_ren
      E_from_batt = battery.discharge(P_residual, dt)
      P_after_batt = max(P_residual * dt - E_from_batt, 0.0) / dt

    # 2. macchina a stati del generatore (a ogni passo)
    need_gen = P_after_batt > cap_t
    can_stop = P_after_batt <= max(cap_t - delta, 0.0)

    hours_in_state += dt

    if gen_on:
      if can_stop and hours_in_state >= tau_on:
        gen_on = False
        hours_in_state = 0.0
    else:
      if need_gen and hours_in_state >= tau_off:
        gen_on = True
        hours_in_state = 0.0
        n_starts += 1

    # 3. potenza erogata
    if gen_on:
      requested = max(P_after_batt - cap_t, 0.0)
      if enforce_p_min:
        requested = max(requested, GEN_P_MIN)
      P_gen = min(requested, GEN_P_MAX)
      gen_on_hours += dt
    else:
      P_gen = 0.0

    # 4. rete, eccedenza del generatore, violazione
    P_grid = max(P_after_batt - P_gen, 0.0)

    # l'eccedenza non e' cedibile alla rete: carica la batteria
    P_excess_gen = max(P_gen - P_after_batt, 0.0)
    if P_excess_gen > 0:
      E_extra = battery.charge(P_excess_gen, dt)
      E_charged += E_extra
      E_gen_forced_Wh += P_excess_gen * dt

    P_violation = max(P_grid - cap_t, 0.0)
    if P_violation > 0:
      E_violation_Wh += P_violation * dt
      violation_hours += dt
      max_violation_W = max(max_violation_W, P_violation)

    # 5. costi ed emissioni (la tariffa usa P_TARIFF, non cap_t)
    if P_grid > 0:
      E_base = min(P_grid, P_TARIFF) * dt
      E_peak = max(P_grid - P_TARIFF, 0.0) * dt
      cost_grid = E_base * cost_low + E_peak * cost_high
      co2_grid_g = ((E_base + E_peak) / 1000) * grid_co2_intensity
      sum_grid_Wh += E_base + E_peak

    if P_gen > 0:
      E_from_gen = P_gen * dt
      cost_gen = generator.get_cost(E_from_gen)
      co2_gen_g = generator.get_co2(E_from_gen / 1000)
      sum_gen_Wh += E_from_gen

    throughput = E_charged + E_from_batt
    cost_batt = battery.step_cost(dt, throughput)
    co2_batt_g = battery.step_co2_g(dt, throughput)
    sum_batt_Wh += E_from_batt

    total_cost += cost_green + cost_grid + cost_gen + cost_batt
    total_co2_g += co2_green_g + co2_grid_g + co2_gen_g + co2_batt_g

    battery.step(dt)

  total_hours = df["dt_hours"].sum()
  days = total_hours / 24

  return {
      "costo_euro": round(total_cost, 3),
      "co2_kg": round(total_co2_g / 1000, 3),
      "E_da_rete_Wh": round(sum_grid_Wh, 1),
      "E_da_generatore_Wh": round(sum_gen_Wh, 1),
      "E_rete_piu_gen_Wh": round(sum_grid_Wh + sum_gen_Wh, 1),
      "E_da_batteria_Wh": round(sum_batt_Wh, 1),
      "curtailment_Wh": round(sum_curtail_Wh, 1),
      "E_violazione_Wh": round(E_violation_Wh, 1),
      "ore_violazione": round(violation_hours, 2),
      "perc_tempo_violazione": round(100 * violation_hours / total_hours, 3),
      "max_violazione_kW": round(max_violation_W / 1000, 1),
      "n_accensioni": n_starts,
      "accensioni_al_giorno": round(n_starts / days, 1),
      "ore_gen_acceso": round(gen_on_hours, 2),
      "E_gen_forzata_Wh": round(E_gen_forced_Wh, 1),
      "SOC_finale": round(battery.info()["SOC"], 3),
  }


if __name__ == "__main__":
  # controllo di regressione: senza vincoli si ritrova la politica All
  # originale (6,4 MWh, 850 kW, 400 kW)
  EXPECTED_COST = 20201.85
  EXPECTED_CO2 = 21847.27
  TOLERANCE = 0.01

  print("Controllo di regressione (profilo costante, alpha = 0):")
  baseline = simulate(SCENARIOS["costante"], 0.0,
                      delta=0, tau_on=0, tau_off=0, enforce_p_min=False)
  print(f"  costo {baseline['costo_euro']:.2f} EUR   "
        f"CO2 {baseline['co2_kg']:.2f} kg   "
        f"avviamenti {baseline['n_accensioni']}")
  assert abs(baseline["costo_euro"] - EXPECTED_COST) < TOLERANCE, \
      f"costo {baseline['costo_euro']:.2f} != {EXPECTED_COST}"
  assert abs(baseline["co2_kg"] - EXPECTED_CO2) < TOLERANCE, \
      f"CO2 {baseline['co2_kg']:.2f} != {EXPECTED_CO2}"
  print("  OK\n")

  results = []

  for scenario_name, cap in SCENARIOS.items():
    for alpha in ALPHA_LIST:
      for label, delta, tau_on, tau_off, p_min in CONTROLS:
        r = simulate(cap, alpha, delta=delta, tau_on=tau_on,
                     tau_off=tau_off, enforce_p_min=p_min)
        r["scenario"] = scenario_name
        r["alpha"] = alpha
        r["controllo"] = label
        r["delta_kW"] = delta / 1000
        r["tau_on_min"] = tau_on * 60
        r["tau_off_min"] = tau_off * 60
        results.append(r)

        print(f"{scenario_name:<15} a={alpha:<4} {label:<16} -> "
              f"{r['costo_euro']:>9.0f} EUR  "
              f"{r['co2_kg']:>8.0f} kg  "
              f"{r['n_accensioni']:>5} avvii "
              f"({r['accensioni_al_giorno']:>5.1f}/giorno)  "
              f"viol {r['ore_violazione']:>6.1f} h")

  columns = ["scenario", "alpha", "controllo", "delta_kW",
             "tau_on_min", "tau_off_min",
             "costo_euro", "co2_kg", "n_accensioni",
             "accensioni_al_giorno", "ore_gen_acceso", "E_gen_forzata_Wh",
             "E_violazione_Wh", "ore_violazione", "perc_tempo_violazione",
             "max_violazione_kW", "E_da_rete_Wh", "E_da_generatore_Wh",
             "E_rete_piu_gen_Wh", "E_da_batteria_Wh", "curtailment_Wh",
             "SOC_finale"]

  os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
  table = pd.DataFrame(results)[columns]
  table.to_csv(OUTPUT_FILE, index=False)
  print(f"\nSalvato in {OUTPUT_FILE}  ({len(results)} righe)")

  # energia rete + generatore rispetto al controllo reattivo: i vincoli
  # cambiano da dove proviene l'energia, non quanta ne serve
  print("\nInvariante rete + generatore rispetto al controllo reattivo:")
  print(f"  {'scenario':<15} {'alpha':<6} {'controllo':<16} "
        f"{'scarto [Wh]':>13} {'forzata [Wh]':>13} {'SOC fin.':>9}")
  for scenario_name in SCENARIOS:
    for alpha in ALPHA_LIST:
      sel = table[(table.scenario == scenario_name)
                  & (table.alpha == alpha)].set_index("controllo")
      ref = sel.loc["reattivo", "E_rete_piu_gen_Wh"]
      for label, *_ in CONTROLS:
        if label == "reattivo":
          continue
        r = sel.loc[label]
        print(f"  {scenario_name:<15} {alpha:<6} {label:<16} "
              f"{r['E_rete_piu_gen_Wh'] - ref:>13.1f} "
              f"{r['E_gen_forzata_Wh']:>13.1f} "
              f"{r['SOC_finale']:>9.3f}")
