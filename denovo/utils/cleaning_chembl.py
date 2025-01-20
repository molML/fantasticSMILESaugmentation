# import sys
# sys.path.append('/path/to/your/module')
import os
import pandas as pd
from smiles_processing import clean_smiles, is_supported_chemical

def is_supported_smiles(smiles):
    clean = clean_smiles(smiles, to_canonical=False)
    return clean, int(clean is not None and is_supported_chemical(clean))

if __name__ == '__main__':
    # Sources path
    source_path = "./denovo/datasets/ChEMBL33.csv"
    target_path = "./denovo/datasets/cleaned_ChEMBL33"
    os.makedirs(target_path, exist_ok=True)

    # Reading in the ChEMBL33 dataframe
    df = pd.read_csv(f"{source_path}")  
    df = df.sample(frac=1)  # Shuffling of the dataframe

    # Cleaning of the SMILES dataframe
    df['cleaned_smiles'], df["is_supported"] = zip(*df["canonical_smiles"].apply(is_supported_smiles))

    # Only chose the ones who are valid and supported
    clean_df = df.query("is_supported == 1") [['cleaned_smiles']]
    clean_df = clean_df[clean_df['cleaned_smiles'].map(len) <= 150] # Sort out the ones that are too big
    clean_df = clean_df[clean_df['cleaned_smiles'].map(len) > 6]    # Sort out the ones that are too short

    clean_df_shuf = clean_df.sample(frac=1)    # Shuffling of the dataframe
    # Separating in train, val and test set
    train_frac, val_frac, test_frac = 0.7, 0.1, 0.2
        
    training_ix = int(len(clean_df_shuf) * train_frac)
    val_ix = int(len(clean_df_shuf) * val_frac) + training_ix
    test_ix = int(len(clean_df_shuf) * test_frac) + val_ix

    # Separate according to indices
    df_train = clean_df_shuf.iloc[: training_ix, :]
    df_val = clean_df_shuf.iloc[training_ix : val_ix, :]
    df_test = clean_df_shuf.iloc[val_ix :, :]

    # Check for length of dataset
    if len(df_train) + len(df_val) + len(df_test) != len(clean_df_shuf):
        raise ValueError("Error in train / val / test split.")
    
    # Save the dataset
    df_train.to_csv(f"{target_path}/train.csv", index = None)
    df_val.to_csv(f"{target_path}/val.csv", index = None)
    df_test.to_csv(f"{target_path}/test.csv", index = None)
