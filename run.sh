#!/bin/bash

# Run
python run_model.py --root_dir=. --config=gpt2-xl_trec_sst2 --use_saved_results
python run_calibration.py --root_dir=. --experiment=gpt2-xl_trec_sst2 --config=logloss_100boots --use_saved_results
python run_plots.py --experiment gpt2-xl_trec_sst2 --config logloss_100boots --num_samples 600 --n_shots 0
