"""
Creation of subsets of dataset sizes 1000, 2500, 5000, 7500, 10000, and 25000.
The subsets were created using clustering method by van Tilborg et at. (Karman line).
"""

import os
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy.sparse import csgraph
from scipy.linalg import eigh
from kneed.knee_locator import KneeLocator
from sklearn.cluster import SpectralClustering
from smiles_processing import clean_smiles, is_supported_chemical
import matplotlib.pyplot as plt

def is_supported_smiles(smiles):
    """
    Cleaning of the SMILES according.
    arguments
    smiles: SMILES string to clean
    returns
    cleaned smiles string, bool
    """
    clean = clean_smiles(smiles, to_canonical=False)
    return clean, int(clean is not None and is_supported_chemical(clean))


def get_generic_scaffold(smiles):
    """
    Generation of scaffold of a SMILES string. Either the generic scaffold is returned,
    or if the molecule does not have a scaffold, then the SMILES strings are returned.
    arguments
    smiles: SMILES string
    returns
    scaffold in SMILES format
    """
    mol = Chem.MolFromSmiles(smiles)    # Transform SMILES to molecular object

    if mol is None:
        raise ValueError("Invalid SMILES string")

    # Get the Bemis-Murcko scaffold, which is a common type of scaffold
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    # Simplification of the scaffold
    scaffold = MurckoScaffold.MakeScaffoldGeneric(scaffold)

    # Check if the scaffold has any ring structures or if it is an aliphatic chain
    if scaffold.GetNumAtoms() == 0:        
        return smiles   # no rings present, return the original molecule

    return Chem.MolToSmiles(scaffold)   # Return the scaffold back as SMILES string

def smiles_to_ecfp(smiles, radius = 2, nBits = 2048):
    """
    Convert a SMILES string to an ECFP fingerprint. Radius is 2, nBits is 20248.
    arguments
    smiles: SMILES string to convert.
    returns
    ECFP fingerprints
    """

    mol = Chem.MolFromSmiles(smiles)    # Transforming SMILES into molecular object

    # In case of invalid SMILES string
    if mol is None:
        raise ValueError("Invalid SMILES string")
    
    return Chem.AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits)   # Returns ECFP4 fingerprints.

def calculate_tanimoto_similarity(fp1, fp2):
    """
    Calculate the pairwise Tanimoto similarity between two ECFP fingerprints. One fingerprint represent one molecule,
    another the other. The fingerprints are calculated with the function (smiles_to_ecfp).
    arguments
    fp1: Fingerprint of molecule 1
    fp2: Fingerprint of molecule 2
    returns
    Tanimoto Similarity
    """
    return Chem.DataStructs.TanimotoSimilarity(fp1, fp2)

def generate_affinity_matrix(smiles_list):
    """ 
    Whole pipeline of generation of affinity matrix based on Tanimoto similarity for a list of SMILES strings.
    This is performed according to the idea by van Tilborg et al.
    arguments
    smiles_list: list of SMILES string to perform clustering algorithm
    returns
    affinity matrix
    """

    # Creation of fingerprints
    fingerprints = [smiles_to_ecfp(smiles) for smiles in smiles_list]

    # Creation of affinity matrix
    n_affinity = len(fingerprints)
    affinity_matrix = np.zeros((n_affinity, n_affinity))    # square matrix

    # Start parsing through fingerprints, calculate Tanimoto similarity and add the to the affinity matrix
    for i in range(len(fingerprints)):  
        for j in range(len(fingerprints)):
            # To avoid redundant calculations, parsing only for j = 0 , ... , len(fingerprints)
            # and for i = 0, ... , j
            if i <= j:  
                similarity = calculate_tanimoto_similarity(fingerprints[i], fingerprints[j])
                # Mirror the triangle (symmetric matrix)
                affinity_matrix[i, j] = similarity
                affinity_matrix[j, i] = similarity

    return affinity_matrix

def calculations_clusters(smiles_df, path):
    """
    Calculates the clusters with the help of the Laplacian matrix and Eigendecomposition of the matrix.
    First step is the calculation of the affinity matrix (mirror list of Tanimoto similarities of ECFP4). Afterwards, 
    the Laplacian matrix from the affinity matrix is calculated. Eigendecomposition is performed to
    calculate the eigenvalues, which then helps to calculates the best number of clusters with the help of the Elbow
    method. Afterwards, Spectral Clustering is performed with the help of the optimal number of clusters from the Elbow
    Method and the SMILES strings get a label belonging to one cluster.
    arguments
    smiles_df: dataframe of cleaned SMILES string
    path: path to save the figure of the elbow method
    returns
    smiles_df with updated scaffolds and labels 
    """
    
    # Calculation of the scaffolds with Bemis-Murcko Scaffolds
    smiles_df['scaffolds'] = smiles_df['cleaned_smiles'].apply(get_generic_scaffold)
    # Calculation of affinity matrix
    affinity_matrix = generate_affinity_matrix(list(smiles_df['scaffolds']))
    # Calculation of the normalized Laplacian matrix
    laplacian = csgraph.laplacian(affinity_matrix, normed=True)

    # Eigendecomposition of Laplacian
    eigenvalues = eigh(laplacian, eigvals_only=True)
    # Calculation of the optimal cluster with the Elbow method
    knee = KneeLocator(range(len(eigenvalues)), eigenvalues, curve='concave', direction='increasing')
    
    # Create the figure for sanity check
    knee.plot_knee(figsize=(13,10), title='Elbow Point of Clusters', xlabel='Number of Clusters', ylabel='Eigenvalues')
    plt.xlim((-5, len(eigenvalues) + 5))
    plt.ylim((0, max(eigenvalues)))
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(f'{path}/knee_method.png')

    optimal_clustering = knee.elbow # Number of optimal clusters

    # Perform Spectral clustering with the optimal clustering amount and the affinity matrix.
    spectral = SpectralClustering(n_clusters = optimal_clustering, affinity='precomputed')  # affinity matrix already available
    labels = spectral.fit_predict(affinity_matrix)  # creation of labels for the affinity matrix, each SMILES string will get a label

    smiles_df['ClusterLabel'] = labels  # Add the labels to the SMILES dataframe
    return smiles_df

if __name__ == '__main__':
    # Sources path
    source_path = "./denovo/datasets/ChEMBL33.csv"
    target_path = "./denovo/datasets/subsets_real"
    os.makedirs(target_path, exist_ok=True)

    # Reading in the ChEMBL33 dataframe
    df = pd.read_csv(f"{source_path}")
    df = df.sample(frac=1)  # Shuffling of the dataframe

    df = df.sample(n=100000)    # Choosing randomly of a smaller dataframe (too big to perform cluster on 2M)

    # Cleaning of the SMILES dataframe
    df['cleaned_smiles'], df["is_supported"] = zip(*df["canonical_smiles"].apply(is_supported_smiles))
    # print('SMILES that are valid and supported:', df['is_supported'].value_counts())

    clean_df = df.query("is_supported == 1") [['cleaned_smiles']]
    # Sort out molecules towards length
    clean_df = clean_df[clean_df['cleaned_smiles'].map(len) <= 150] # Sort out the ones that are too big
    clean_df = clean_df[clean_df['cleaned_smiles'].map(len) > 6]    # Sort out the ones that are too short

    # Subsample again randomly 50k, as 100k is still too big
    clean_df_shuf = clean_df.sample(n=50000)    # 50000
    clean_df_shuf = clean_df_shuf.sample(frac=1)    # Shuffling of the dataframe

    # Start of the clustering procedure
    smiles_clustered = calculations_clusters(clean_df_shuf, target_path)
    smiles_clustered.to_csv(f'{target_path}/clustered_smiles.csv')  # Saving the clustered dataset as a starting point

    # The smaller dataframes should be included in the bigger ones, so we start from big to small
    total_samples = [50000, 25000, 10000, 7500, 5000, 2500, 1000]
    previous_sample = 0
    cumulative_sampled_df = pd.DataFrame()
    sampled_df = None 
    # Parse through sizes
    for size in total_samples:

        # Count each cluster how many it should have in the new dataframe (subset)
        cluster_counts = smiles_clustered['ClusterLabel'].value_counts()  
        # Calculation of the proportion  relative towards the size of the dataframe
        # Number needs to be an integer
        sample_counts = (cluster_counts / cluster_counts.sum() * size).round().astype(int)

        # Sample the DataFrame based on the calculated sample_counts
        if sampled_df is None:  # Create the starting dataframe of the bigger sizes
            sampled_df = pd.concat(
                [smiles_clustered[smiles_clustered['ClusterLabel'] == cluster].sample(n=count, random_state=15) 
                for cluster, count in sample_counts.items()]
            )   # Choose of methods belonging to a specific cluster
        elif sampled_df is not None:    # Available dataframe of bigger size
            sampled_df = pd.concat(
                [sampled_df[sampled_df['ClusterLabel'] == cluster].sample(n=count, random_state=15)
                for cluster, count in sample_counts.items()]
            )                        

        # Separation in training and validation set
        train_frac, val_frac = 0.9, 0.1
        
        training_ix = int(len(sampled_df) * train_frac)
        val_ix = int(len(sampled_df) * val_frac) + training_ix

        df_train = sampled_df.iloc[: training_ix, :]
        df_val = sampled_df.iloc[training_ix : , :]

        if len(df_train) + len(df_val) != len(sampled_df):  # Sanity check
            raise ValueError("Error in train / val split.")
        
        subset_path = f"{target_path}/subset_{size}"
        os.makedirs(subset_path, exist_ok=True) # Create directory

        df_train.to_csv(f"{subset_path}/train.csv", index = None)
        df_val.to_csv(f"{subset_path}/val.csv", index = None)

        print(f'Done with sampling size: {size}')

