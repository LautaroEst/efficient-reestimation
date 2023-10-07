#!/bin/bash

# Run
python run_model.py --root_dir=. --config=gpt2-xl_trec_sst2_agnews_dbpedia --use_saved_results
python run_calibration.py --root_dir=. --experiment=gpt2-xl_trec_sst2_agnews_dbpedia --config=logloss_noboots #--use_saved_results
python run_plots.py --experiment gpt2-xl_trec_sst2_agnews_dbpedia --config logloss_noboots --num_samples 600 --n_shots 0
