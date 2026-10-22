import numpy as np
import pandas as pd


def constant_threshold(df, base_w):
  return np.full(len(df), float(base_w))


def single_event(df, base_w, start, duration_hours, reduced_w,
                 ramp_minutes=0):
  """
  Riduzione a reduced_w per duration_hours a partire da start.
  Con ramp_minutes > 0 il gradino diventa una transizione lineare.
  """
  t = pd.to_datetime(df["time"], utc=True)
  t0 = pd.Timestamp(start, tz="UTC")
  t1 = t0 + pd.Timedelta(hours=duration_hours)

  cap = np.full(len(df), float(base_w))
  inside = (t >= t0) & (t < t1)
  cap[inside.to_numpy()] = float(reduced_w)

  if ramp_minutes > 0:
    cap = _apply_ramp(cap, t, t0, t1, base_w, reduced_w,
                      ramp_minutes)
  return cap


def recurring_events(df, base_w, start_hour, end_hour, reduced_w):
  """
  Riduzione ogni giorno nella fascia oraria [start_hour, end_hour).
  """
  t = pd.to_datetime(df["time"], utc=True)
  hour_of_day = t.dt.hour.to_numpy()

  cap = np.full(len(df), float(base_w))
  if start_hour <= end_hour:
    inside = (hour_of_day >= start_hour) & (hour_of_day < end_hour)
  else:  # fascia a cavallo della mezzanotte
    inside = (hour_of_day >= start_hour) | (hour_of_day < end_hour)
  cap[inside] = float(reduced_w)
  return cap


def stochastic_events(df, base_w, events_per_month=10, mean_duration_hours=4,
                      depth_min=0.3, depth_max=0.8, seed=0):
  """
  Arrivi di Poisson, durata esponenziale (minimo 30 min) e profondita'
  uniforme, espressa come frazione di base_w.
  """
  rng = np.random.default_rng(seed)
  t = pd.to_datetime(df["time"], utc=True)
  total_hours = (t.iloc[-1] - t.iloc[0]).total_seconds() / 3600

  cap = np.full(len(df), float(base_w))
  n = rng.poisson(events_per_month * total_hours / (30 * 24))

  for _ in range(max(n, 0)):
    offset = rng.uniform(0, total_hours)
    duration = max(rng.exponential(mean_duration_hours), 0.5)
    depth = rng.uniform(depth_min, depth_max)

    t0 = t.iloc[0] + pd.Timedelta(hours=offset)
    t1 = t0 + pd.Timedelta(hours=duration)
    inside = ((t >= t0) & (t < t1)).to_numpy()

    # eventi sovrapposti: prevale il piu' severo
    cap[inside] = np.minimum(cap[inside], base_w * depth)

  return cap


def _apply_ramp(cap, t, t0, t1, base_w, reduced_w, ramp_minutes):
  """
  Sostituisce i gradini di inizio e fine evento con rampe lineari.
  """
  d = pd.Timedelta(minutes=ramp_minutes)
  for ta, tb, v_from, v_to in [(t0 - d, t0, base_w, reduced_w),
                               (t1, t1 + d, reduced_w, base_w)]:
    m = ((t >= ta) & (t < tb)).to_numpy()
    if m.sum() > 0:
      fraction = np.linspace(0, 1, m.sum())
      cap[m] = v_from + (v_to - v_from) * fraction
  return cap


def summary(cap, df):
  """
  Soglia minima, numero di eventi e ore in riduzione del profilo.
  """
  t = pd.to_datetime(df["time"], utc=True)
  dt = t.diff().dt.total_seconds().fillna(0) / 3600
  nominal = cap.max()
  reduced = cap < nominal
  reduced_hours = dt[reduced].sum()

  n_events = int(np.sum(np.diff(reduced.astype(int)) == 1))
  if reduced[0]:
    n_events += 1

  return (f"soglia nominale {nominal/1000:.0f} kW | "
          f"minimo {cap.min()/1000:.0f} kW | "
          f"{n_events} eventi | "
          f"{reduced_hours:.1f} ore in riduzione "
          f"({100*reduced_hours/dt.sum():.1f}% del tempo)")


if __name__ == "__main__":
  df = pd.read_csv("csvs/cluster_power_only_nodes_30days.csv")
  df["time"] = pd.to_datetime(df["time"], utc=True)

  print("Profili disponibili:\n")

  profiles = [
      ("costante",
       constant_threshold(df, 400_000)),
      ("evento singolo, 6 h a 150 kW",
       single_event(df, 400_000, "2020-06-05 14:00", 6, 150_000)),
      ("ricorrente, 18-21 ogni giorno a 200 kW",
       recurring_events(df, 400_000, 18, 21, 200_000)),
      ("stocastico, seme 0",
       stochastic_events(df, 400_000, seed=0)),
      ("stocastico, seme 1",
       stochastic_events(df, 400_000, seed=1)),
  ]

  for name, cap in profiles:
    print(f"  {name:<40} {summary(cap, df)}")
