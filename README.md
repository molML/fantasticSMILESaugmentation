# SMILES augmentation beyond enumeration for generative deep learning in low data regimes

## Table of Contents
- [Description](#description)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [How To Cite](#howtocite)
- [License](#license)

## Description
This project uses four different approaches to augment SMILES strings and analyses its performance for generative deep learning, especially in low data scenarios. The methods are compared to the well-known SMILES enumeration and no data augmentation.

Following (novel) approaches are analysed:
- **Token Deletion**, which removes specific symbols (‘tokens’) from a SMILES string to generate variations of the original input. We performed three deletion strategies:
    • Random deletion
    • Random deletion with enforced validity
    • Random deletion with protection of certain tokens
2. **Atom Masking**, which replaces specific atoms with a placeholder (‘mask’). We investigated two atom masking strategies:
    • Random masking
    • Masking of functional groups 
3. Bioisosteric Substitution, which replaces groups of tokens with their respective bioisosteres. 
4. Augmentation by self-training, which is defined as the process of feeding a generative deep learning approach its own generated samples. 

The deletion of tokens, masking of atoms and bioisosteric substitution is each controlled by a probability (p).

## Prerequisites

The project was performed with following softwares:

Python (3.9.18)
Tensorflow (2.17.0)
Keras (3.4.1)

## Installation

Clone the repository:

```git clone https://github.com/molML/fantasticSMILESaugmentation```

The environment can be install with the following command:

```conda env create -f environment.yml```

## Usage

### Augment your dataset

As mentioned before, different types of augmentation can take place (in total eight). Following can be chosen as strings:
- Enumeration: 'enumeration'
- Random deletion: 'random-deletion-invalid'
- Random deletion with enforced validity: 'random-deletion-valid'
- Random deletion with protection of certain tokens: 'protected-deletion'
- Random masking: 'random-masking'
- Masking of functional groups: 'group-masking'
- Bioisosteric Substitution: 'bioisosters-based'
- Augmentation by self-training: 'self-training'

### Considerations

Depending which type of augmentation method is taking place, different things need to be considered:
For **Token Deletion**, the SMILES strings need to be tokenized before. This is done as follow:

```python

from encoding import DataEncoding

encoder = DataEncoding()
smiles_list = encoding.tokenizer(smiles)    # smiles: List of SMILES string
```

For **Masking of functional groups** or **Bioisosteric substitution**, a list of SMARTS strings needs or a functional group dictionary to be called to determine which groups can be masked or substituted:

```python

# Open pattern file if bioisosteric substitution or functional group masking
if augmentation_method == 'bioisosters-based':
    with open(f"patterns_dict.json") as f:
        augumentation_extra = json.load(f)
elif augmentation_method == 'group-masking':
    df_patterns = pd.read_csv('patterns.csv', delimiter=';', quotechar='"',)
    filtered_smarts = df_patterns[df_patterns['type']=='functional-group']  # maybe can be expanded towards scaffolds, different rings, etc
    augumentation_extra = filtered_smarts['smarts']
else:
    augumentation_extra = None  # To perform it uniformly with one command
```

### Perform augmentation for all methods except of augmentation by self-training

```python

import augmentation_libraries as aug

augmentation_fold = 10  # different augmentation folds are possible
prob = 0.15     # probability of deletion, masking or substitution

smiles_list = list(smiles_dataframe['smiles']) # or the tokenized molecules for deletion

# Define the augmentation method one wants to use, see strings above
method = '...'

# Initialize the class of augmentation
aug_method = aug.get_method(method, smiles_list, augmentation_fold, prob)

# For all methods except masking
augmented_smiles = aug_method.perform_augmentation(augumentation_extra)

# For the masking methods
augmented_smiles, augmented_smiles_label = aug_method.perform_augmentation(augumentation_extra)
```

### Perform augmentation for augmentation by self-training

The model needs to be trained once without data augmenation. Afterwards, the model without augmentation is called to perform sampling of new SMILES strings. Novel SMILES strings are iteratively saved into a file.

```python
import augmentation_libraries as aug

method = 'self-training'
augmentation_fold = 10  # different augmentation folds are possible
prob = 0.15     # probability of deletion, masking or substitution, not important here
smiles_list = list(smiles_dataframe['smiles']) 

 # Calling in parameter dictionary of model trained on non-augmented datasets
with open('path_to_results/fold_1/best_combination.json') as f:    
    hp_space = json.load(f)

aug_method = aug.get_method(method, smiles_list, augmentation_fold, prob)

# temperature: determines if more conservative or diverse sampling takes place
# sampling_parameters: hyperparameter combination and path of saved model
# augmentation_saving_dir: where the augmented SMILES should be saved (as self-training can take long, it is iteratively saved)
# list_sampled: is a list of already augmented smiles is available
augmented_smiles = method_train.perform_augmentation(temperature=0.5, sampling_parameters=hp_space, augmentation_saving_dir='path_to_results/train', list_sampled=None) 

```


### Training of the chemical language model

It is up to you how you want to train your model, but we provide in generation.py a code of a chemical language model where small and big datasets can be trained (with the possibility of using Data Generator), but also fine tuning and prediction can take place.

## How to Cite:

Brinkmann H, Argante A, ter Steege H, Grisoni F. Going beyond SMILES enumeration for generative deep learning  in low data regimes. ChemRxiv. 2025; doi:10.26434/chemrxiv-2025-fdnnq  This content is a preprint and has not been peer-reviewed.

## License
The novel SMILES augmentations are under MIT license.



