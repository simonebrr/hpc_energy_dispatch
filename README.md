# Trace-Driven Energy Dispatch for HPC Clusters

This repository contains the simulation code and experimental material for the PECS Workshop paper at Euro-Par 2026:

**Trace-Driven Energy Dispatch for HPC Clusters: Cost--Carbon Trade-offs under Grid Power Constraints**

## Contents

- `rule_based_sims.ipynb`  
  Contains the rule-based simulations used in the paper, including the grid-only, battery-assisted, renewable-aware, and generator-supported dispatch policies.

- `rl_agent_sim3_optimize.py`  
  Contains the preliminary reinforcement-learning optimizer for the `All` scenario, based on battery, renewable generation, backup generator, and grid dispatch.

- `requirements.txt`  
  Python dependencies needed to reproduce the experiments.

## Setup

Create and activate a Conda environment, using requirements.txt


## Estensioni (tesi di laurea, S. Brunelli)

- `rule_based_sims.ipynb`: le quattro politiche accettano un sovraccarico di raffreddamento `alpha` (`P_tot = P_IT * (1 + alpha)`); i risultati vanno in `cooling/risultati/`. Con `alpha = 0` si ritrovano i valori originali.
- `eventi.py`: profili di soglia variabile nel tempo (costante, evento singolo, ricorrente, stocastico).
- `sim_soglia_dinamica.py`: politica All con limite operativo variabile e metriche di violazione.
- `sim_isteresi.py`: politica All con isteresi, tempi minimi di funzionamento/arresto e potenza minima del generatore.
- `sim_soglie_scalate.py`: separa l'effetto del maggiore consumo da quello della soglia invariata.
- `figura_architettura.py`, `figure_cooling.py`, `figure_soglia.py`: figure della tesi (in `figure/`).

Gli script vanno eseguiti dalla radice del repository, ad esempio `python3 sim_isteresi.py`; i CSV vengono scritti in `risultati/`.
