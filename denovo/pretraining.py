import os
import sys
 # run on GPU 0 only
# os.environ["CUDA_VISIBLE_DEVICES"]="-1" # run on CPU
file_dir = os.path.dirname(__file__)
# os.chdir(file_dir)
print(file_dir)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import json
from typing import Dict, List
import numpy as np
import pandas as pd
import tensorflow as tf
# import keras
import os
# import augmentation_libraries as aug
from encoding import DataEncoding
from generation_tl import CLM, split_input_target
from sampling import SamplingMolecules
# from quality_control import QualityControl as QC
import random

if __name__ == "__main__":

    random.seed(42)
    
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run-solo",
        help="Whether the modes in run solo in the GPU",
        default=0,
        choices=[0, 1],
        type=int,
    )

    parser.add_argument(
        "--n-samples",
        help="n number of molecules to sample",
        default=1000,
        type=int,
    )

    parser.add_argument(
        "--temperature",
        help="Temperature to sample (T < 1.0 more conservative, T > 1.0 more random)",
        default=1.0,
        type=float,
    )

    # Initialization of the parameters
    args = parser.parse_args()
    temperature = args.temperature
    n_samples = args.n_samples

    print('Parameters: temperature', temperature, 'n_samples', n_samples)
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    base_dir = f"./denovo"
    saving_directory = "."

    # Reading in the hp-combination json file and then pick a subset of it.
    with open(f"./hp_space_pretraining.json") as f:
        hp_combination = json.load(f)

    results_dir = f"{saving_directory}/results/fine_tuning"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    try:
        with open(f"{results_dir}/pre_training/best_combination.json", "r") as f:
            best_config = json.load(f)
    
        try:
            with open(f'{results_dir}/segment2label.json', 'r') as json_file:
                segment2label = json.load(json_file)
        except FileNotFoundError:
            print('Apparently there is a mistake. Combination dictionary was found, but not the dictionary. Please double check.')
            raise

    except FileNotFoundError:
        print('Pre-Training did not took place yet. Starting Pre-Training.')

        dataset_dir = f"./datasets/cleaned_ChEMBL33"
        df_train = pd.read_csv(f"{dataset_dir}/train.csv")
        df_val = pd.read_csv(f"{dataset_dir}/val.csv")

        try:
            print('Trying of reading of tokenization vocabulary.')
            with open(f'{results_dir}/segment2label.json', 'r') as json_file:
                segment2label = json.load(json_file)
            
            print(segment2label)

        except FileNotFoundError:
            print('Encoding never took place. Creating tokenization vocabulary.')
            segment2label = None

            total_list = list(df_train['cleaned_smiles']) + list(df_val['cleaned_smiles'])

            encoding = DataEncoding()
            segment2label, tokenized_smiles = encoding.create_vocab(total_list, segment2label)
            segment2label.update({'*' : len(segment2label)})

            with open(f'{results_dir}/segment2label.json', 'w') as json_file:
                json.dump(segment2label, json_file)
            
            print('Length of Segment2Label:', len(segment2label), 'Segment2label', segment2label)

        encoding = DataEncoding()

        train_smiles = list(df_train['cleaned_smiles'])
        tokenized_train = encoding.tokenizer(train_smiles)
        tokenized_train = encoding.add_tokens(tokenized_train)

        val_smiles = list(df_val['cleaned_smiles'])
        tokenized_val = encoding.tokenizer(val_smiles)
        tokenized_val = encoding.add_tokens(tokenized_val)

        # Creation of pre-training dir
        pre_train_dir = f"{results_dir}/pre_training"
        if not os.path.exists(pre_train_dir):
            os.makedirs(pre_train_dir)

        hp_combination["main_saving_dir"] = results_dir       
        hp_combination["combination_saving_dir"] = pre_train_dir
        hp_combination['contains_embedding_layer'] = False
        hp_combination["info_size"] = len(segment2label)

        trainer = CLM(model_parameters=hp_combination, mode='Train', path=pre_train_dir, use_data_generator=True)
        model, history = trainer.train_model(training_data=tokenized_train, validation_data=tokenized_val, vocab=segment2label)
            
        best_score = min(history.history["val_loss"])
        epoch_best_score = int(np.argmin(history.history["val_loss"]))
        train_score = history.history["val_loss"][-1]
        stopping_epoch = len(history.history["val_loss"]) - 1
            
        combination_dump = dict()
        combination_dump["hps"] = hp_combination
        combination_dump["misc"] = dict()
        combination_dump["misc"]["val_loss"] = train_score
        combination_dump["misc"]["stopping_epoch"] = stopping_epoch        
        combination_dump["misc"]["best_val_loss"] = best_score
        combination_dump["misc"]["best_epoch"] = epoch_best_score

            
        with open(f"{pre_train_dir}/best_combination.json", "w") as f:
            json.dump(combination_dump, f, indent=4)

    for i in range(4):
        # Starting of sampling procedure.
        if os.path.exists(f"{pre_train_dir}/unclean_sampled_molecules_{i}.json"):
            print(f"Molecules were already sampled for pretraining.")        
        else: 
            print('Start Sampling procedure.')
            with open(f"{pre_train_dir}/best_combination.json") as f:
                hp_space = json.load(f)

            sampler = SamplingMolecules(sampling_parameter=hp_space["hps"])
            molecules = sampler.sample_multiple(num_sample=n_samples, temperature=temperature, starting_char='G')

            with open(f"{pre_train_dir}/unclean_sampled_molecules_{i}.json", "w") as f:
                json.dump(molecules, f, indent=4)