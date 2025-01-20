import os
import sys
file_dir = os.path.dirname(__file__)
print(file_dir)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import augmentation_libraries as aug
from encoding import DataEncoding
from generation import CLM, split_input_target
from sampling import SamplingMolecules

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--n-trials", 
        help="Number of hp-combinations to run", 
        type=int, 
        default=2
    )

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
        default=10,  
        type=int,
    )

    parser.add_argument(
        "--temperature",
        help="Temperature to sample (T < 1.0 more conservative, T > 1.0 more random)",
        default=1.0,
        type=float,
    )

    parser.add_argument(
        "--augmentation-method",
        help="Type of augmentation method to use",
        default='random-deletion-invalid', 
        choices=['random-deletion-invalid', 'random-deletion-valid', 'protected-deletion'],
        type=str,
    )

    parser.add_argument(
        "--dataset-size",
        help="Which size of dataset will be used",
        default=1000,
        choices=[1000, 2500, 5000, 7500, 10000, 25000, 50000],
        type=int,
    )

    parser.add_argument(
        "--prob",
        help="Which probability to perform masking, deletion and substitution",
        default=0.05,
        type=float,
    )

    # Initialization of the parameters
    args = parser.parse_args()
    n_trials = args.n_trials
    temperature = args.temperature
    n_samples = args.n_samples
    augmentation_method = args.augmentation_method
    dataset_size = args.dataset_size
    prob = args.prob

    print('Parameters: n_trials', n_trials, 'temp', temperature, 'n_samples', n_samples, 'augmentation_method', augmentation_method, 'dataset size', dataset_size, 'prob', prob)
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    base_dir = f"./denovo"
    saving_directory = "."

    # Reading in the hp-combination json file and then pick a subset of it.
    with open(f"./sampled_hp_space.json") as f:
        hp_combinations = json.load(f)
    hp_spaces = hp_combinations[: n_trials]

    augmentation_dir = f"{saving_directory}/results/{augmentation_method}/dataset_{dataset_size}"
    augumentation_extra = None  # No extra lists necessary

    # Starting of encoding procedure
    encoding = DataEncoding()

    for fold in [1, 3, 5, 10]:

        best_score = np.inf

        print(f'Start dataset Size {dataset_size} and Augmentation_Fold {fold}!')

        # Creation of augmentation directory
        augmentation_multiple_dir = f"{augmentation_dir}/fold_{fold}"
        print(augmentation_multiple_dir)
        if not os.path.exists(augmentation_multiple_dir):
            os.makedirs(augmentation_multiple_dir)
        
        # Try to read the datasets for the augmentation
        # Either the datasets are already there, the segment dictionary also or the augmentation needs to take place.
        # Segment2label should be of each fold the unaugmented dataset (as deletion happens after tokenization)
        try:
            df_train = pd.read_csv(f'{augmentation_multiple_dir}/datasets/train.csv')
            augmented_train = list(df_train['smiles'])
            df_val = pd.read_csv(f'{augmentation_multiple_dir}/datasets/val.csv')
            augmented_val = list(df_val['smiles']) 

            with open(f'{augmentation_multiple_dir}/segment2label.json', 'r') as json_file:
                segment2label = json.load(json_file)

        # File was not found. Dictionary and data needs to be created.
        except FileNotFoundError:

            print(f'The data was not augmented yet. Augment fold {fold} for dataset size {dataset_size}.')

            # Directory of data fetching 
            dataset_dir = f"./datasets/subsets/subset_{dataset_size}"     # fetching the dataset
            print('Library is fetched from:', dataset_dir)

            augmentation_data_saving_dir = f'{augmentation_multiple_dir}/datasets'
            if not os.path.exists(augmentation_data_saving_dir):
                os.makedirs(augmentation_data_saving_dir)                
    
            df_train = pd.read_csv(f"{dataset_dir}/train.csv")
            smiles_train = list(df_train['cleaned_smiles'])
            df_val = pd.read_csv(f"{dataset_dir}/val.csv")
            smiles_val = list(df_val['cleaned_smiles'])

            segment2label = None

            total_list = smiles_train + smiles_val  # The vocaulary should be of both training and validation, as they might not have the same tokens
            segment2label, tokenized_smiles = encoding.create_vocab(total_list, segment2label)  # SMILES strings are tokenized and padded in this step
            
            # Saving of the dictionary
            with open(f'{augmentation_multiple_dir}/segment2label.json', 'w') as json_file:
                json.dump(segment2label, json_file)

            print('Length of Segment2Label:', len(segment2label), 'Segment2label', segment2label)

            if fold == 1:
                # No data augmentation necessary
                augmented_train = smiles_train
                augmented_val = smiles_val

            else:
                print(f'Fold {fold} needs data augmentation. Starting ...')

                # Tokenization of SMILES strings for deletion
                tokenized_train = encoding.tokenizer(smiles_train)             
                tokenized_val = encoding.tokenizer(smiles_val)

                method_train = aug.get_method(augmentation_method, smiles_train, fold, prob)
                method_val = aug.get_method(augmentation_method, smiles_val, fold, prob)    

                augmented_train = method_train.perform_augmentation(tokenized_train)
                augmented_val = method_val.perform_augmentation(tokenized_val)
            
            df_train = pd.DataFrame(augmented_train, columns=['smiles'])
            df_train.to_csv(f'{augmentation_data_saving_dir}/train.csv', sep=',')

            df_val = pd.DataFrame(augmented_val, columns=['smiles'])
            df_val.to_csv(f'{augmentation_data_saving_dir}/val.csv', sep=',')

        tokenized_train = encoding.tokenizer(augmented_train)
        tokenized_train = encoding.add_tokens(tokenized_train)        
        one_hot_smiles_train, _ = encoding.one_hot_encoding(tokenized_train, segment2label)
    
        tokenized_val = encoding.tokenizer(augmented_val)
        tokenized_val = encoding.add_tokens(tokenized_val)  
        one_hot_smiles_val, _ = encoding.one_hot_encoding(tokenized_val, segment2label)

        one_hot_smiles_train_target = None
        one_hot_smiles_val_target = None

        xTrain, yTrain = split_input_target(one_hot_smiles_train, one_hot_smiles_train_target)
        xVal, yVal =  split_input_target(one_hot_smiles_val, one_hot_smiles_val_target)

        # Starting hp combinations
        for combination_idx, hp_combination in enumerate(hp_spaces):
            
            combination_saving_dir = f"{augmentation_multiple_dir}/combination_{combination_idx}"
            if os.path.exists(f"{combination_saving_dir}/combination.json"):
                print(f"Combination {combination_idx} already exists!")

                with open(f"{combination_saving_dir}/combination.json", 'r') as f:
                    combination_dump = json.load(f)
            
                hp_combination_score = combination_dump["misc"]["val_loss"]

                with open(f"{augmentation_multiple_dir}/best_combination.json", "r") as f:
                    best_config = json.load(f)

            else:
                print(f"Start Combination {combination_idx}")
                
                hp_combination["main_saving_dir"] = augmentation_multiple_dir
                hp_combination["prob"] = prob
                hp_combination['combination_saving_dir'] = combination_saving_dir   # Saving the combination directory

                hp_combination['contains_embedding_layer'] = False
                hp_combination["info_size"] = len(segment2label)

                trainer = CLM(model_parameters=hp_combination, mode='Train', path=combination_saving_dir)
                model, history = trainer.train_model(xTrain=xTrain, yTrain=yTrain, xVal=xVal, yVal=yVal)
            
                train_score = min(history.history["val_loss"])
                stopping_epoch = int(np.argmin(history.history["val_loss"]))
            
                combination_dump = dict()
                combination_dump["hps"] = hp_combination
                combination_dump["misc"] = dict()
                combination_dump["misc"]["val_loss"] = train_score
                combination_dump["misc"]["stopping_epoch"] = stopping_epoch
            
                with open(f"{combination_saving_dir}/combination.json", "w") as f:
                    json.dump(combination_dump, f, indent=4)

                if train_score < best_score:
                    best_score = train_score
                    best_config = combination_dump

            with open(f"{augmentation_multiple_dir}/best_combination.json", "w") as f:
                json.dump(best_config, f, indent=4)

        # Starting of sampling procedure.
        for i in range(4):

            if os.path.exists(f"{augmentation_multiple_dir}/unclean_sampled_molecules_{i}.json"):
                print(f"Molecules were already sampled for augmentation method '{augmentation_method}', dataset size '{dataset_size}' and fold '{fold}x'.")        
            else: 

                with open(f"{augmentation_multiple_dir}/best_combination.json") as f:
                    hp_space = json.load(f)

                sampler = SamplingMolecules(sampling_parameter=hp_space)
                molecules = sampler.sample_multiple(num_sample=n_samples, temperature=temperature, starting_char='G')
                    
                with open(f"{augmentation_multiple_dir}/unclean_sampled_molecules_{i}.json", "w") as f:
                    json.dump(molecules, f, indent=4)