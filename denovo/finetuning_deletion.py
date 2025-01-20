import os
import sys
file_dir = os.path.dirname(__file__)
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
        default='bioisosters-based', 
        choices=['random-deletion-invalid', 'random-deletion-valid', 'protected-deletion'],
        type=str,
    )

    parser.add_argument(
        "--prob",
        help="Which probability to perform masking, deletion and substitution",
        default=0.15,
        type=float,
    )

    # Initialization of the parameters
    args = parser.parse_args()
    temperature = args.temperature
    n_samples = args.n_samples
    augmentation_method = args.augmentation_method
    prob = args.prob

    print('Parameters: temperature', temperature, 'n_samples', n_samples, 'augmentation_method', augmentation_method, 'prob', prob)
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    base_dir = f"./denovo"
    saving_directory = "."

    augumentation_extra = None  # No substructures necessary

    results_dir = f"{saving_directory}/results/fine_tuning"
    
    # Trying to open things from pre-training, raise error if pre-training did not take place
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
        print('Pre-Training did not took place yet. Please do first pretraining.')
        raise
    
    print('Start in Fine-Tuning...')

    # Parse through every target
    encoding = DataEncoding()

    # Parse through every target
    for target in ['PPAR', 'PIM1', 'JAK2']:
        
        for size in [10, 100]:

            for sim in ['similar', 'dissimilar']:

                augmentation_dir = f"{results_dir}/{target}/{sim}/subset_{size}/{augmentation_method}"
                dataset_dir = f"./datasets/fine_tuning_datasets/{target}/{sim}/subset_{size}"   # directory to fetch dataset

                for fold in [3, 5, 10]:

                    # Creation of augmentation directory
                    augmentation_multiple_dir = f"{augmentation_dir}/fold_{fold}"
                    # See if fine-tuning took place (if yes, jump to sampling)
                    try:
                        with open(f"{augmentation_multiple_dir}/best_combination.json", "r") as json_file:
                            hp_space = json.load(json_file)
                        print(f'Fine tuning already took place for {target}, size {size}, {sim}, {augmentation_method} and fold {fold}.')
                    # Start fine-tuning                    
                    except FileNotFoundError:
                        print(augmentation_multiple_dir)
                        if not os.path.exists(augmentation_multiple_dir):
                            os.makedirs(augmentation_multiple_dir)

                        print(f"Augmentation method '{augmentation_method}', size {size}, prob {prob} and fold '{fold}x' ...")

                        # Try to read the datasets for the augmentation
                        # Either the datasets are already there, the segment dictionary also or the augmentation needs to take place.
                        # The train and the validation set should be tokenized datasets
                        # Segment2label should be of each fold the unaugmented dataset
                        try:
                            df_train = pd.read_csv(f'{augmentation_multiple_dir}/datasets/train.csv')
                            augmented_train = list(df_train['smiles'])
                            df_val = pd.read_csv(f'{augmentation_multiple_dir}/datasets/val.csv')
                            augmented_val = list(df_val['smiles']) 

                        # File was not found. Data needs to be created.
                        except FileNotFoundError:

                            print(f'The data was not augmented yet. Augment size {size} and fold {fold}.')

                            # Directory of data fetching 
                            print('Library is fetched from:', dataset_dir)

                            augmentation_data_saving_dir = f'{augmentation_multiple_dir}/datasets'
                            if not os.path.exists(augmentation_data_saving_dir):
                                os.makedirs(augmentation_data_saving_dir)                
        
                            df_train = pd.read_csv(f"{dataset_dir}/train.csv")
                            smiles_train = list(df_train['SMILES'])

                            df_val = pd.read_csv(f"{dataset_dir}/val.csv")
                            smiles_val = list(df_val['SMILES'])

                            # Before deletion, the molecules should be tokenized (the function detokenizes them)
                            tokenized_train = encoding.tokenizer(smiles_train)             
                            tokenized_val = encoding.tokenizer(smiles_val)

                            # Perform augmentation
                            method_train = aug.get_method(augmentation_method, smiles_train, fold, prob)
                            method_val = aug.get_method(augmentation_method, smiles_val, fold, prob)    

                            augmented_train = method_train.perform_augmentation(tokenized_train)
                            augmented_val = method_val.perform_augmentation(tokenized_val)

                            # Save datasets
                            df_train = pd.DataFrame(augmented_train, columns=['smiles'])
                            df_train.to_csv(f'{augmentation_data_saving_dir}/train.csv', sep=',')

                            df_val = pd.DataFrame(augmented_val, columns=['smiles'])
                            df_val.to_csv(f'{augmentation_data_saving_dir}/val.csv', sep=',')

                        # Augmentation is done, now training procedure
                        # Starting of encoding procedure
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

                        print('Start fine-tuning.')
                        
                        # Fetch model configuration from pre-trained model
                        hp_combination = best_config["hps"]

                        # Initialization and finetuning                        
                        trainer = CLM(model_parameters=hp_combination, mode='Finetune', path=augmentation_multiple_dir, freeze_layers=False, pre_trained_model_path=hp_combination["combination_saving_dir"], use_data_generator=False)
                        model, history = trainer.fine_tune_model(xTrain=xTrain, yTrain=yTrain, xVal=xVal, yVal=yVal)

                        # Get some configurations
                        best_score = min(history.history["val_loss"])
                        epoch_best_score = int(np.argmin(history.history["val_loss"]))
                        train_score = history.history["val_loss"][-1]
                        stopping_epoch = len(history.history["val_loss"]) - 1

                        best_config["fine_tuning"] = dict()
                        best_config["fine_tuning"]["prob"] = prob
                        best_config["fine_tuning"]["new_learning_rate"] = hp_combination["learning_rate"] * 0.001
                        best_config["fine_tuning"]["new_batch_size"] = 2
                        best_config["fine_tuning"]["saving_dir"] = augmentation_multiple_dir
                        best_config["fine_tuning"]["val_loss"] = train_score
                        best_config["fine_tuning"]["stopping_epoch"] = stopping_epoch        
                        best_config["fine_tuning"]["best_val_loss"] = best_score
                        best_config["fine_tuning"]["best_epoch"] = epoch_best_score    

                        with open(f"{augmentation_multiple_dir}/best_combination.json", "w") as f:
                            json.dump(best_config, f, indent=4)
                            
                        print('Done fine-tuning.')

                    for i in range(4):
                        # Starting of sampling procedure.
                        if os.path.exists(f"{augmentation_multiple_dir}/unclean_sampled_molecules_finetune_{i}.json"):     # Check if already sampled
                            print(f"Molecules were already sampled for augmentation method '{augmentation_method}', size {size}' and fold '{fold}x'.")        
                        else: 
                            print('Start Sampling procedure.')
                            with open(f"{augmentation_multiple_dir}/best_combination.json") as f:
                                hp_space = json.load(f)

                            # Start sampling procedure
                            sampler = SamplingMolecules(sampling_parameter=hp_space)
                            molecules = sampler.sample_multiple(num_sample=n_samples, temperature=temperature, starting_char='G')
                                
                            with open(f"{augmentation_multiple_dir}/unclean_sampled_molecules_finetune_{i}.json", "w") as f:
                                json.dump(molecules, f, indent=4)