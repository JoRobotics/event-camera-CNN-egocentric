@echo off
if exist results_log.txt del results_log.txt
for %%C in (cache_events cache_frames) do (
  for %%S in (0 1 2) do (
    echo python train.py --cache %%C --epochs 25 --seed %%S >> results_log.txt
    python train.py --cache %%C --epochs 25 --seed %%S >> results_log.txt
  )
)
python plot_results.py results_log.txt
