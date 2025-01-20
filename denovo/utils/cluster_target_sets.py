import sys
sys.path.append('/path/to/your/module')
import os

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold
import numpy as np
from tqdm import tqdm
import pandas as pd
from scipy.cluster.hierarchy import average, cut_tree    # linkage, fcluster, 
from rdkit.SimDivFilters import rdSimDivPickers
import random
from smiles_processing import clean_smiles, is_supported_chemical
import h5py
from typing import Any


random.seed(42)

def is_supported_smiles(smiles):
    clean = clean_smiles(smiles, to_canonical=False)
    return clean, int(clean is not None and is_supported_chemical(clean))


def smi_to_scaff(smiles: str, includeChirality: bool = False):
    return MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(smiles), includeChirality=includeChirality)

def get_tanimoto_matrix(smiles: list[str], radius: int = 2, nBits: int = 1024, verbose: bool = True,
                        scaffolds: bool = False, zero_diag: bool = True, as_vector: bool = False):
    """ 
    Calculates a matrix of Tanimoto similarity scores for a list of SMILES string, creating the distance matrix
    that can be further used for clustering purposes. The code was taken from van Tilborg et al. 
    'Traversing chemical space with active deep learning for low-data drug discovery'. For further information,
    please refer to the paper.
    """

    # Make a fingerprint database
    db_fp = {}
    for smi in smiles:
        if scaffolds:
            m = Chem.MolFromSmiles(smi_to_scaff(smi, includeChirality=False))
        else:
            m = Chem.MolFromSmiles(smi)
        fp = AllChem.GetMorganFingerprintAsBitVect(m, radius=radius, nBits=nBits)
        db_fp[smi] = fp

    smi_len = len(smiles)
    m = np.zeros([smi_len, smi_len], dtype=np.float16)  # We use 16-bit floats to prevent giant matrices

    # Calculate upper triangle of matrix
    for i in tqdm(range(smi_len), disable=not verbose):
        for j in range(i, smi_len):
            m[i, j] = DataStructs.TanimotoSimilarity(db_fp[smiles[i]], db_fp[smiles[j]])     # similarity

    # Fill in the lower triangle without having to loop (saves ~50% of time)
    m = m + m.T - np.diag(np.diag(m))
    # Fill the diagonal with 0's
    if zero_diag:
        np.fill_diagonal(m, 0)
    if as_vector:
        from scipy.spatial.distance import squareform
        m = squareform(m)

    return m

def hierarchical_clustering(distance_matrix, smiles, num_big_cluster=10, n_sub_cluster=20):
    """
    Performs hierarchical clustering and assigns each molecule to a cluster. Method is similar to a paper by van Tilborg et al.
    'Traversing chemical space with active deep learning for low-data drug discovery'.    
    arguments
    distance_matrix: tanimoto distance matrix.
    smiles: list of SMILES strings.
    num_clusters: number of higher-level clusters
    n_sub_cluster: number of sub-clusters within each larger cluster
    returns
    dataframe with SMILES and their corresponding cluster assignments
    """
    # Compute the hierarchical clustering linkage matrix using the average linkage method
    Z = average(distance_matrix)
    
    # Assign clusters to the data based on the hierarchical clustering result
    # num_big_cluster determines the number of higher-level (larger) clusters to divide the data into
    # n_sub_cluster specifies the number of sub-clusters within each larger cluster
    clusters = cut_tree(Z, n_clusters=[num_big_cluster, n_sub_cluster])
    
    # Create a DataFrame to store SMILES and their cluster assignments
    cluster_df = pd.DataFrame({
        'SMILES': smiles,
        'ClusterUpper': clusters[:, 0],
        'ClusterSmall': clusters[:, 1]
    })
    
    return cluster_df

def high_similarity_datasets(cluster_df, path, smarts_patterns: list, num_samples=100, radius: int = 2, 
                             nBits: int = 1024, threshold: float = 0.8, extra_check: bool = True):
    """
    Creates a dataset of high similarity molecules based on Tanimoto distance and cluster assignment according
    to hierarchial clustering analysis (however, can be adapted to any clustering method).
    arguments
    cluster_df: SMILES dataframe and cluster assignments
    path: saving path of datasets
    smarts_patterns: list of SMARTS patterns to check if the SMILES in the cluster have functional groups
    num_samples: number of molecules for dataset
    radius: radius for ECFP
    nBits: bits for ECPF
    threshold: threshold for similarity of high similar
    extra_check: if an extra check should be conducted towards high_similarity
    returns
    last high similarity dataframe
    """

    # Convert SMARTS strings to molecular objects
    smarts_mol_patterns = [Chem.MolFromSmarts(smarts) for smarts in smarts_patterns]
    if None in smarts_mol_patterns:
        raise ValueError("One or more SMARTS patterns could not be parsed. Check the SMARTS strings.")
        
    selected_smiles = []
    # Start parsing through each cluster
    for cluster_id in cluster_df['ClusterUpper'].unique():
        cluster_smiles = cluster_df[cluster_df['ClusterUpper'] == cluster_id]

        # Check if each cluster has the necessary number of samples        
        if len(cluster_smiles) < num_samples:  # Skip clusters with fewer than n_samples molecules
            continue

        # Go into subclusters to find high similarity molecules
        for subcluster_id in cluster_smiles['ClusterSmall'].unique():
            subcluster_df = cluster_smiles[cluster_smiles['ClusterSmall'] == subcluster_id]

            # Check again if necessary number of samples            
            if len(subcluster_df) < num_samples:
                continue
            
            subclustered_smiles = list(subcluster_df['SMILES']) # retrieve SMILES strings
            # Generate fingerprints for the molecules in this subcluster to calculate pairwise similarity
            mols = [Chem.MolFromSmiles(smi) for smi in subclustered_smiles]
            fps = [AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits) for mol in mols]
            
            # Calculate pairwise Tanimoto similarities
            pairwise_similarities = []
            for i in range(len(fps)):
                for j in range(i + 1, len(fps)):
                    sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
                    pairwise_similarities.append((i, j, sim))   # tuple of indice of molecule 1 and 2 and its similarity

            pairwise_similarities.sort(key=lambda x: x[2], reverse=True)  # sort list of tuples by similarity values in descending order (high to low)

            # Check if the similarity is higher or equal than the threshold
            selected_indices = set()         # no duplicated   
            for i, j, sim in pairwise_similarities:
                if sim >= threshold:
                    selected_indices.add(i)
                    selected_indices.add(j)
            selected_indices = list(selected_indices)
            
            high_similarity_df = subcluster_df.iloc[selected_indices]   # sort the dataframe out for the necessary
            # Deduplicate and ensure exact number of samples
            selected_smiles_before = list(high_similarity_df['SMILES'])

            # Check if enough samples
            if len(selected_smiles_before) < num_samples:
                continue
            else:   # Start some checks
                # Ensure substructural matches to functional groups
                matched_smiles = []
                for smi in high_similarity_df['SMILES']:    # parse through high similarity
                    mol = Chem.MolFromSmiles(smi)
                    if any(mol.HasSubstructMatch(pattern) for pattern in smarts_mol_patterns):
                        matched_smiles.append(smi)

                # Check if enough samples
                if len(matched_smiles) < num_samples:
                    continue

                # Filter to include only the rows where the SMILES are in high similarity
                high_similarity_df = high_similarity_df[high_similarity_df['SMILES'].isin(matched_smiles)]

                # Deduplicate and ensure exact number of samples
                selected_smiles = list(high_similarity_df['SMILES'])
                selected_smiles = list(set(selected_smiles))
                
                # Last check to see if the SMILES strings have a pairwise high Tanimoto index
                if extra_check:
                    matrix = get_tanimoto_matrix(selected_smiles, as_vector=True)
                    count = [x >= threshold for x in matrix]
                    if len(matrix) != len(count):
                        raise ValueError(f'The Tanimoto coefficients of the matrixs are not highly similar (above threshold of {threshold}).')
                
                # Limit the resulting dataframe to the desired number of samples
                high_similarity_df = high_similarity_df.iloc[ : num_samples]

                # Save high similarity dataframe
                high_similarity_df.to_csv(f'{path}/high_similarity_{num_samples}_{cluster_id}_{subcluster_id}.csv')     

                return high_similarity_df

def low_similarity_dataset(clustered_df, path: str,  num_samples: int = 100, radius: int = 2, nBits: int = 1024,
                           threshold: float = 0.6, extra_check = True):
    """
    Creates a dataset of low similarity molecules with the help of Leader_Picker and Tanimoto similarity.
    arguments
    cluster_df: SMILES dataframe and cluster assignments
    path: saving path of datasets
    num_samples: number of molecules for dataset
    radius: radius for ECFP
    nBits: bits for ECPF
    threshold: threshold for similarity of high similar
    extra_check: if an extra check should be conducted towards high_similarity
    returns
    low similarity dataframe
    """

    smiles_list = list(clustered_df['SMILES'])  # extract the SMILES strings

    # Generate fingerprints for LeaderPicker function
    fingerprints = list()
    for smi in smiles_list:
        m = Chem.MolFromSmiles(smi)     # convert so molecular object
        fp = AllChem.GetMorganFingerprintAsBitVect(m, radius=radius, nBits=nBits)   # create fingeprints
        fingerprints.append(fp)

    picker = rdSimDivPickers.LeaderPicker()   # LeaderPicker: Select diverse samples based on fingerprints

    # Select indices of fingerprints with low similarity (threshold: similarity limit, num_samples: number of sampled to pick)
    selected_indices = picker.LazyBitVectorPick(fingerprints, len(fingerprints), threshold, pickSize=num_samples)

    low_similarity_dataset = clustered_df.iloc[selected_indices]    # extract SMILES from indices

    # Extra check if the pairwise Tanimoto similarity is below threshold
    if extra_check:
        selected_smiles = list(low_similarity_dataset['SMILES'])    # get SMILES of low similarity
        matrix = get_tanimoto_matrix(selected_smiles, as_vector=True)   # get tanimoto similarity matrix
        tanimoto_threshold = 1 - threshold  # convert threshold to Tanimoto similarity threshold
        # Count how many Tanimoto coefficients are below the Tanimoto similarity
        count = [x <= tanimoto_threshold for x in matrix]   
        # Ensure that all pairwise tanimoto similarity satisfy the threshold condition
        if len(matrix) != len(count):
            raise ValueError(f'The Tanimoto coefficients of the matrixs are not highly similar (above threshold of {threshold}).')

    low_similarity_dataset.to_csv(f'{path}/low_similarity_{num_samples}.csv')   # Save dataset

    return low_similarity_dataset


def save_hdf5(obj: Any, filename: str):
    """
    Save big dataframes in hdf5 format
    """
    hf = h5py.File(filename, 'w')
    hf.create_dataset('obj', data=obj)
    hf.close()


def load_hdf5(filename: str):
    """
    Load hdf5 files
    """
    hf = h5py.File(filename, 'r')
    obj = np.array(hf.get('obj'))
    hf.close()

    return obj

def samples(dataframe, size):
    # Sample without replacement, used for training and validation test
    dataframe = dataframe.sample(frac=1, random_state = 42)
    sampled_df = dataframe.sample(n=size, random_state=42, replace=False)
    return sampled_df


if __name__ == '__main__':
    df_patterns = pd.read_csv('./denovo/patterns.csv', delimiter=';', quotechar='"',)
    filtered_smarts = df_patterns[df_patterns['type']=='functional-group']
    pattern = filtered_smarts['smarts']

    for target in ['PPAR', 'PIM1', 'JAK2']: 
        source_path = f'./denovo/datasets/fine_tuning_datasets'
        target_path = f'{source_path}/{target}'
        os.makedirs(target_path, exist_ok=True)

        df = pd.read_csv(f'{source_path}/{target}.csv')

        # Cleaning of the SMILES dataframe
        df['cleaned_smiles'], df["is_supported"] = zip(*df["smiles"].apply(is_supported_smiles))
        print('SMILES that are valid and supported:', df['is_supported'].value_counts())

        clean_df = df.query("is_supported == 1") [['cleaned_smiles']]
        clean_df = clean_df[clean_df['cleaned_smiles'].map(len) <= 150] # Sort out the ones that are too big
        clean_df = clean_df[clean_df['cleaned_smiles'].map(len) > 6]    # Sort out the ones that are too short

        clean_df.to_csv(f'{target_path}/{target}_clustered.csv')

        smiles_list = list(clean_df['cleaned_smiles'])
        similarity = get_tanimoto_matrix(smiles_list, verbose=True, scaffolds=False, zero_diag=True, as_vector=True)    # generate tanimoto similarity matrix
        distance = 1 - similarity
        save_hdf5(distance, f'{target_path}/tanimoto_distance_vector')  # save distance matrix
        del similarity

        # Perform hierarchial clustering
        cluster_df = hierarchical_clustering(distance, smiles_list, num_big_cluster=15, n_sub_cluster=30)

        cluster_df.to_csv(f'{target_path}/clustered_df.csv')    # Save clustered df

        high_similarity_df = high_similarity_datasets(cluster_df, target_path, pattern) # perform high similarity dataset extraction

        # create subset of the bigger dataframe
        for size in [10, 100]:
            sampled_df = samples(high_similarity_df, size)

            train_frac, val_frac = 0.8, 0.2  # training and validation split

            training_ix = int(len(sampled_df) * train_frac) 
            val_ix = int(len(sampled_df) * val_frac) + training_ix

            df_train = sampled_df.iloc[: training_ix, :]
            df_val = sampled_df.iloc[training_ix : , :]

            if len(df_train) + len(df_val) != len(sampled_df):
                raise ValueError("Error in train / val split.")
        
            subset_path = f"{target_path}/similar/subset_{size}"
            os.makedirs(subset_path, exist_ok=True)

            df_train.to_csv(f"{subset_path}/train.csv", index = None)
            df_val.to_csv(f"{subset_path}/val.csv", index = None)

        # Perform low similarity datasets
        low_similarity_df = low_similarity_dataset(cluster_df, path=target_path, num_samples=100)

        for size in [10, 100]:
            
            sampled_df = samples(low_similarity_df, size)

            train_frac, val_frac = 0.8, 0.2

            training_ix = int(len(sampled_df) * train_frac)
            val_ix = int(len(sampled_df) * val_frac) + training_ix

            df_train = sampled_df.iloc[: training_ix, :]
            df_val = sampled_df.iloc[training_ix : , :]

            if len(df_train) + len(df_val) != len(sampled_df):
                raise ValueError("Error in train / val split.")
        
            subset_path = f"{target_path}/dissimilar/subset_{size}"
            os.makedirs(subset_path, exist_ok=True)

            df_train.to_csv(f"{subset_path}/train.csv", index = None)
            df_val.to_csv(f"{subset_path}/val.csv", index = None)