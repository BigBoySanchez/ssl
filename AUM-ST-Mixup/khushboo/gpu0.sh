#!/bin/bash

# event 1 -california_wildfires_2018

# CUDA_VISIBLE_DEVICES=1 PYTHONHASHSEED=42 python3 run_ust.py --disaster california_wildfires_2018 --sample_scheme easy_bald_class_conf  --aum_save_dir AUM1   --train_file 5_set1   --results_file aum_mixsal_1_5
# CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=42 python3 run_ust.py --disaster california_wildfires_2018 --sample_scheme easy_bald_class_conf  --aum_save_dir AUM0   --train_file 5_set2   --results_file aum_mixsal_2_5
# CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=42 python3 run_ust.py --disaster california_wildfires_2018 --sample_scheme easy_bald_class_conf  --aum_save_dir AUM0   --train_file 5_set3   --results_file aum_mixsal_3_5

# CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=42 python3 run_ust.py --disaster california_wildfires_2018 --sample_scheme easy_bald_class_conf  --aum_save_dir AUM0   --train_file 10_set1   --results_file aum_mixsal_1_10
# CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=42 python3 run_ust.py --disaster california_wildfires_2018 --sample_scheme easy_bald_class_conf  --aum_save_dir AUM0   --train_file 10_set2   --results_file aum_mixsal_2_10
# CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=42 python3 run_ust.py --disaster california_wildfires_2018 --sample_scheme easy_bald_class_conf  --aum_save_dir AUM0   --train_file 10_set3   --results_file aum_mixsal_3_10

# CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=42 python3 run_ust.py --disaster california_wildfires_2018 --sample_scheme easy_bald_class_conf  --aum_save_dir AUM0   --train_file 25_set1   --results_file aum_mixsal_1_25
# CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=42 python3 run_ust.py --disaster california_wildfires_2018 --sample_scheme easy_bald_class_conf  --aum_save_dir AUM0   --train_file 25_set2   --results_file aum_mixsal_2_25
# CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=42 python3 run_ust.py --disaster california_wildfires_2018 --sample_scheme easy_bald_class_conf  --aum_save_dir AUM0   --train_file 25_set3   --results_file aum_mixsal_3_25

# CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=42 python3 run_ust.py --disaster california_wildfires_2018 --sample_scheme easy_bald_class_conf  --aum_save_dir AUM0   --train_file 50_set1   --results_file aum_mixsal_1_50
# CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=42 python3 run_ust.py --disaster california_wildfires_2018 --sample_scheme easy_bald_class_conf  --aum_save_dir AUM0   --train_file 50_set2   --results_file aum_mixsal_2_50
# CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=42 python3 run_ust.py --disaster california_wildfires_2018 --sample_scheme easy_bald_class_conf  --aum_save_dir AUM0   --train_file 50_set3   --results_file aum_mixsal_3_50

# event 2 - canada_wildfires_2016

# CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=42 python3 run_ust_2208.py --disaster canada_wildfires_2016 --sample_scheme easy_bald_class_conf  --aum_save_dir AUM0_canada_wildfires_2016   --train_file 5_set1   --results_file aum_mixsal_1_5 --num_labels 8 --dataset humanitarian8
# PYTHONHASHSEED=42 python3 run_ust.py --disaster canada_wildfires_2016 --sample_scheme easy_bald_class_conf  --aum_save_dir AUM0   --train_file 5_set2   --results_file aum_mixsal_2_5 --num_labels 8
# PYTHONHASHSEED=42 python3 run_ust.py --disaster canada_wildfires_2016 --sample_scheme easy_bald_class_conf  --aum_save_dir AUM0   --train_file 5_set3   --results_file aum_mixsal_3_5 --num_labels 8

# PYTHONHASHSEED=42 python3 run_ust.py --disaster canada_wildfires_2016 --sample_scheme easy_bald_class_conf  --aum_save_dir AUM0   --train_file 10_set1   --results_file aum_mixsal_1_10 --num_labels 8
# PYTHONHASHSEED=42 python3 run_ust.py --disaster canada_wildfires_2016 --sample_scheme easy_bald_class_conf  --aum_save_dir AUM0   --train_file 10_set2   --results_file aum_mixsal_2_10 --num_labels 8
# PYTHONHASHSEED=42 python3 run_ust.py --disaster canada_wildfires_2016 --sample_scheme easy_bald_class_conf  --aum_save_dir AUM0   --train_file 10_set3   --results_file aum_mixsal_3_10 --num_labels 8

# PYTHONHASHSEED=42 python3 run_ust.py --disaster canada_wildfires_2016 --sample_scheme easy_bald_class_conf  --aum_save_dir AUM0   --train_file 25_set1   --results_file aum_mixsal_1_25 --num_labels 8
# PYTHONHASHSEED=42 python3 run_ust.py --disaster canada_wildfires_2016 --sample_scheme easy_bald_class_conf  --aum_save_dir AUM0   --train_file 25_set2   --results_file aum_mixsal_2_25 --num_labels 8
# PYTHONHASHSEED=42 python3 run_ust.py --disaster canada_wildfires_2016 --sample_scheme easy_bald_class_conf  --aum_save_dir AUM0   --train_file 25_set3   --results_file aum_mixsal_3_25 --num_labels 8

# PYTHONHASHSEED=42 python3 run_ust.py --disaster canada_wildfires_2016 --sample_scheme easy_bald_class_conf  --aum_save_dir AUM0   --train_file 50_set1   --results_file aum_mixsal_1_50 --num_labels 8
# PYTHONHASHSEED=42 python3 run_ust.py --disaster canada_wildfires_2016 --sample_scheme easy_bald_class_conf  --aum_save_dir AUM0   --train_file 50_set2   --results_file aum_mixsal_2_50 --num_labels 8
# PYTHONHASHSEED=42 python3 run_ust.py --disaster canada_wildfires_2016 --sample_scheme easy_bald_class_conf  --aum_save_dir AUM0   --train_file 50_set3   --results_file aum_mixsal_3_50 --num_labels 8

# cyclone_idai_2019

# CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=42 python3 run_ust.py --disaster  cyclone_idai_2019 --sample_scheme easy_bald_class_conf  --aum_save_dir AUM0   --train_file 5_set1   --results_file aum_mixsal_1_5
# CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=42 python3 run_ust.py --disaster  cyclone_idai_2019 --sample_scheme easy_bald_class_conf  --aum_save_dir AUM0   --train_file 5_set2   --results_file aum_mixsal_2_5
# CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=42 python3 run_ust.py --disaster  cyclone_idai_2019 --sample_scheme easy_bald_class_conf  --aum_save_dir AUM0   --train_file 5_set3   --results_file aum_mixsal_3_5

# CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=42 python3 run_ust.py --disaster  cyclone_idai_2019 --sample_scheme easy_bald_class_conf  --aum_save_dir AUM0   --train_file 10_set1   --results_file aum_mixsal_1_10
# CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=42 python3 run_ust.py --disaster  cyclone_idai_2019 --sample_scheme easy_bald_class_conf  --aum_save_dir AUM0   --train_file 10_set2   --results_file aum_mixsal_2_10
# CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=42 python3 run_ust.py --disaster  cyclone_idai_2019 --sample_scheme easy_bald_class_conf  --aum_save_dir AUM0   --train_file 10_set3   --results_file aum_mixsal_3_10

# CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=42 python3 run_ust.py --disaster  cyclone_idai_2019 --sample_scheme easy_bald_class_conf  --aum_save_dir AUM0   --train_file 25_set1   --results_file aum_mixsal_1_25
# CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=42 python3 run_ust.py --disaster  cyclone_idai_2019 --sample_scheme easy_bald_class_conf  --aum_save_dir AUM0   --train_file 25_set2   --results_file aum_mixsal_2_25
# CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=42 python3 run_ust.py --disaster  cyclone_idai_2019 --sample_scheme easy_bald_class_conf  --aum_save_dir AUM0   --train_file 25_set3   --results_file aum_mixsal_3_25

# CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=42 python3 run_ust.py --disaster  cyclone_idai_2019 --sample_scheme easy_bald_class_conf  --aum_save_dir AUM0   --train_file 50_set1   --results_file aum_mixsal_1_50
# CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=42 python3 run_ust.py --disaster  cyclone_idai_2019 --sample_scheme easy_bald_class_conf  --aum_save_dir AUM0   --train_file 50_set2   --results_file aum_mixsal_2_50
# CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=42 python3 run_ust.py --disaster  cyclone_idai_2019 --sample_scheme easy_bald_class_conf  --aum_save_dir AUM0   --train_file 50_set3   --results_file aum_mixsal_3_50

# # hurricane_dorian_2019

# CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=42 python3 run_ust.py --disaster  hurricane_dorian_2019 --sample_scheme easy_bald_class_conf  --aum_save_dir AUM0   --train_file 5_set1   --results_file aum_mixsal_1_5
# CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=42 python3 run_ust.py --disaster  hurricane_dorian_2019 --sample_scheme easy_bald_class_conf  --aum_save_dir AUM0   --train_file 5_set2   --results_file aum_mixsal_2_5
# CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=42 python3 run_ust.py --disaster  hurricane_dorian_2019 --sample_scheme easy_bald_class_conf  --aum_save_dir AUM0   --train_file 5_set3   --results_file aum_mixsal_3_5

# CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=42 python3 run_ust.py --disaster  hurricane_dorian_2019 --sample_scheme easy_bald_class_conf  --aum_save_dir AUM0   --train_file 10_set1   --results_file aum_mixsal_1_10
# CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=42 python3 run_ust.py --disaster  hurricane_dorian_2019 --sample_scheme easy_bald_class_conf  --aum_save_dir AUM0   --train_file 10_set2   --results_file aum_mixsal_2_10
# CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=42 python3 run_ust.py --disaster  hurricane_dorian_2019 --sample_scheme easy_bald_class_conf  --aum_save_dir AUM0   --train_file 10_set3   --results_file aum_mixsal_3_10

# CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=42 python3 run_ust.py --disaster  hurricane_dorian_2019 --sample_scheme easy_bald_class_conf  --aum_save_dir AUM0   --train_file 25_set1   --results_file aum_mixsal_1_25
# CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=42 python3 run_ust.py --disaster  hurricane_dorian_2019 --sample_scheme easy_bald_class_conf  --aum_save_dir AUM0   --train_file 25_set2   --results_file aum_mixsal_2_25
# CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=42 python3 run_ust.py --disaster  hurricane_dorian_2019 --sample_scheme easy_bald_class_conf  --aum_save_dir AUM0   --train_file 25_set3   --results_file aum_mixsal_3_25

# CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=42 python3 run_ust.py --disaster  hurricane_dorian_2019 --sample_scheme easy_bald_class_conf  --aum_save_dir AUM0   --train_file 50_set1   --results_file aum_mixsal_1_50
# CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=42 python3 run_ust.py --disaster  hurricane_dorian_2019 --sample_scheme easy_bald_class_conf  --aum_save_dir AUM0   --train_file 50_set2   --results_file aum_mixsal_2_50
# CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=42 python3 run_ust.py --disaster  hurricane_dorian_2019 --sample_scheme easy_bald_class_conf  --aum_save_dir AUM0   --train_file 50_set3   --results_file aum_mixsal_3_50

# # hurricane_florence_2018

# CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=42 python3 run_ust.py --disaster  hurricane_florence_2018 --sample_scheme easy_bald_class_conf  --aum_save_dir AUM0   --train_file 5_set1   --results_file aum_mixsal_1_5
# CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=42 python3 run_ust.py --disaster  hurricane_florence_2018 --sample_scheme easy_bald_class_conf  --aum_save_dir AUM0   --train_file 5_set2   --results_file aum_mixsal_2_5
# CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=42 python3 run_ust.py --disaster  hurricane_florence_2018 --sample_scheme easy_bald_class_conf  --aum_save_dir AUM0   --train_file 5_set3   --results_file aum_mixsal_3_5

# CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=42 python3 run_ust.py --disaster  hurricane_florence_2018 --sample_scheme easy_bald_class_conf  --aum_save_dir AUM0   --train_file 10_set1   --results_file aum_mixsal_1_10
# CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=42 python3 run_ust.py --disaster  hurricane_florence_2018 --sample_scheme easy_bald_class_conf  --aum_save_dir AUM0   --train_file 10_set2   --results_file aum_mixsal_2_10
# CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=42 python3 run_ust.py --disaster  hurricane_florence_2018 --sample_scheme easy_bald_class_conf  --aum_save_dir AUM0   --train_file 10_set3   --results_file aum_mixsal_3_10

# CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=42 python3 run_ust.py --disaster  hurricane_florence_2018 --sample_scheme easy_bald_class_conf  --aum_save_dir AUM0   --train_file 25_set1   --results_file aum_mixsal_1_25
# CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=42 python3 run_ust.py --disaster  hurricane_florence_2018 --sample_scheme easy_bald_class_conf  --aum_save_dir AUM0   --train_file 25_set2   --results_file aum_mixsal_2_25
# CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=42 python3 run_ust.py --disaster  hurricane_florence_2018 --sample_scheme easy_bald_class_conf  --aum_save_dir AUM0   --train_file 25_set3   --results_file aum_mixsal_3_25

# CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=42 python3 run_ust.py --disaster  hurricane_florence_2018 --sample_scheme easy_bald_class_conf  --aum_save_dir AUM0   --train_file 50_set1   --results_file aum_mixsal_1_50
# CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=42 python3 run_ust.py --disaster  hurricane_florence_2018 --sample_scheme easy_bald_class_conf  --aum_save_dir AUM0   --train_file 50_set2   --results_file aum_mixsal_2_50
# CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=42 python3 run_ust.py --disaster  hurricane_florence_2018 --sample_scheme easy_bald_class_conf  --aum_save_dir AUM0   --train_file 50_set3   --results_file aum_mixsal_3_50