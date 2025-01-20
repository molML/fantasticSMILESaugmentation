from rdkit import Chem
from abc import ABC, abstractmethod
from .utils.SmilesEnumerator import SmilesEnumerator
from rdkit import Chem
from denovo.sampling import SamplingMolecules
import copy
import csv

class BaseAugmentation(ABC):
    def __init__(self, smiles_list, augmentation_multiple, prob):
        self.smiles_list = smiles_list
        self.canonical_smiles = [Chem.MolToSmiles(Chem.MolFromSmiles(smiles)) for smiles in self.smiles_list]
        self.augmentation_multiple = augmentation_multiple
        self.prob = prob

    @abstractmethod
    def perform_augmentation(self):
        pass

class Selftraining(BaseAugmentation):

    def perform_augmentation(self, temperature=0.5, sampling_parameters=None, augmentation_saving_dir=None, list_sampled=None):

        """
        Performing of data augmentation for Self-Training (previously named 'temperature-based augmentation'). Molecules are
        sampled based on the model trained on no augmented data and the sampled molecules are added to the augmented list,
        which contains the starting dataset and is iteratively added. The molecules are saved in-between, in case the augmentation
        procedure takes too long.
        arguements
        temperature: temperature which sampling takes place (set to 0.5 to perform conservative sampling)
        sampling_parameters: dictionary of the sampling parameters, including saving directory of the model
        augmentation_saving_dir: directory of the augmented molecules
        list_sampled: in-between saved molecules
        returns
        augmented_smiles
        """
        
        sampler = SamplingMolecules(sampling_parameters)
        
        # Checking for availability of SMILES list of already augmented molecules
        if list_sampled is not None:
            # Deep copy of the already started list
            augmented_smiles = copy.deepcopy(list_sampled)
        else:
            # Deep copy of the starting list
            augmented_smiles = copy.deepcopy(self.smiles_list)
            
            # Initialize a CSV file to save augmented molecules incrementally
            with open(f'{augmentation_saving_dir}.csv', mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['smiles'])  # Write header 
                for smiles in augmented_smiles:
                    writer.writerow([smiles])           
        
        # Canonicalization of the augmented SMILES for checking of duplicates
        mols = [Chem.MolFromSmiles(smi) for smi in augmented_smiles]
        can_smiles = [Chem.MolToSmiles(mol, canonical=True) for mol in mols]
        can_aug_smiles = copy.deepcopy(can_smiles)

        # Setting point of until when augmentation takes place
        target_length = (len(self.smiles_list) * self.augmentation_multiple)

        # Starting of the augmentation procedure
        while len(augmented_smiles) < target_length:
            # Sampling of one molecule
            molecule_string = sampler.sample_one(temperature=temperature)
            molecule_string = molecule_string[1 : -1]     # Eliminating the starting and end character
            aug_molecule = "".join([str(item) for item in molecule_string]) # Joining the string together

            # Canonicalizing to check if it is on the list of already available molecules
            mol = Chem.MolFromSmiles(aug_molecule, sanitize=True)
            if mol is not None:
                can_aug_molecule = Chem.MolToSmiles(mol, canonical=True)
                # Check if the sampled SMILES string is not in the list of molecules
                if can_aug_molecule not in can_aug_smiles:
                    augmented_smiles.append(aug_molecule)   # Add molecule to the augmented SMILES list
                    can_aug_smiles.append(can_aug_molecule) # Add molecule to the canonicalized augmented SMILES list

                     # Save the new molecule to the CSV file on the go (easier for longer datasets)
                    with open(f'{augmentation_saving_dir}.csv', mode='a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([aug_molecule])     # Add the not canonicalized molecule                  

        return augmented_smiles

class Enumeration(BaseAugmentation):

    def perform_augmentation(self, *args):

        """
        Performing of data augmentation for enumeration. The method is according to Bjerrum (2017), where a SMILES list is randomized
        by starting the parsing of the molecule at random points. 
        arguements
        None
        returns
        augmented_smiles
        """

        # SMILES enumeration as performed by Bjerrum (2017)
        enumerator = SmilesEnumerator()
        augmented_smiles = copy.deepcopy(self.smiles_list)

        # Setting point of until when augmentation takes place
        target_length = len(self.smiles_list) * self.augmentation_multiple
       
       # Starting of augmentation procedure
        while len(augmented_smiles) < target_length:
            # Parse through each SMILES string of the starting list
            for smiles in self.smiles_list:
                # Use the function randomize_smiles function
                aug_string = enumerator.randomize_smiles(smiles)
                if aug_string not in augmented_smiles and aug_string not in self.canonical_smiles:  
                    augmented_smiles.append(aug_string)
        
        return augmented_smiles