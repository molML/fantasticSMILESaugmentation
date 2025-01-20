from abc import ABC, abstractmethod
import random
from rdkit import Chem
import copy

class BaseDeletionAugmentation(ABC):
    def __init__(self, smiles_list, augmentation_multiple, prob):
        self.smiles_list = smiles_list
        self.canonical_smiles = [Chem.MolToSmiles(Chem.MolFromSmiles(smiles)) for smiles in self.smiles_list]
        self.aug_multiple = augmentation_multiple
        self.prob = prob     # Proabability of deletion

        # Ensure the deletion_probability is between 0 and 1
        if not (0 <= self.prob <= 1):
            raise ValueError("Deletion probability must be between 0 and 1.")
        
    @abstractmethod
    def perform_augmentation(self, *args):
        pass

class DeletionRandom(BaseDeletionAugmentation):

    def perform_augmentation(self, tokenized_smiles):
        """
        Performing of data augmentation for deletion (random). Here, if is not important if we have a valid or
        invalid SMILES strings, but whateer is there will be put as augmented set.
        arguements
        tokenized_smiles: List of tokenized SMILES strings for deletion
        returns
        augmented_smiles 
        """

        # Starting of augmented list containing starting SMILES strings
        augmented_smiles = copy.deepcopy(self.smiles_list)
        # Setting point of until when augmentation takes place
        target_length = (len(self.smiles_list) * self.aug_multiple)

        # Begin augmentation procedure
        while len(augmented_smiles) < target_length:
            # Parsing though tokenized SMILES string
            for smiles in tokenized_smiles:
                # Construct the new string with randomly deleted characters
                # Keeping is the opposite from deleting, therefore invertable     
                aug_molecule = ''.join(char for char in smiles if random.random() > self.prob)  # Keep the char if the 'coinflip' is above probability
                # Check if aug_molecule not already in augmented SMILES list
                if aug_molecule not in self.canonical_smiles and aug_molecule not in augmented_smiles:
                    augmented_smiles.append(aug_molecule)

        return augmented_smiles

class DeletionValid(BaseDeletionAugmentation):

    def perform_augmentation(self, tokenized_smiles):
        """
        Performing of data augmentation for deletion (validity). Here, we only allow valid SMILES string, meaning that it
        needs to result in a valid molecular object.
        arguements
        tokenized_smiles: List of tokenized SMILES strings for deletion
        returns
        augmented_smiles 
        """

        # Starting of augmented list containing starting SMILES strings
        augmented_smiles = copy.deepcopy(self.smiles_list)
        # Setting point of until when augmentation takes place
        target_length = (len(self.smiles_list) * self.aug_multiple)

        # Begin augmentation procedure
        while len(augmented_smiles) < target_length: 
            # Parsing though tokenized SMILES string     
            for smiles in tokenized_smiles:
                # Construct the new string with randomly deleted characters
                # Keeping is the opposite from deleting, therefore invertable
                aug_molecule = ''.join(char for char in smiles if random.random() > self.prob)  # Keep the char if the 'coinflip' is above probability
                mol = Chem.MolFromSmiles(aug_molecule, sanitize=False)  # Try to transform the molecule into molecular object
                # Check for validity
                if mol: # Only append if molecular object is not None
                    # Check if aug_molecule not already in augmented SMILES list
                    if aug_molecule not in augmented_smiles and aug_molecule not in self.canonical_smiles:
                        augmented_smiles.append(aug_molecule)
                else:
                    pass
        
        return augmented_smiles

class DeletionProtected(BaseDeletionAugmentation):

    def perform_augmentation(self, tokenized_smiles, protection_list=None, valid=False):

        """
        Performing of data augmentation for deletion (random). Here, in addition of random deletion, if the character is part of
        the 'protection_list', then the character is not deleted. A flag is added to see if only valid or both valid and invalid
        SMILES strings are allowed.
        arguements
        tokenized_smiles: list of tokenized SMILES strings for deletion
        protection_list: list of tokens to be protected
        valid: flag is the SMILES string needs to be valid or not (default false)
        returns
        augmented_smiles 
        """

        # If protection list not explicit done
        if not protection_list:
            protection_list = ['(', ')', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '%']

        # Starting of augmented list containing starting SMILES strings
        augmented_smiles = copy.deepcopy(self.smiles_list)
        # Setting point of until when augmentation takes place
        target_length = (len(self.smiles_list) * self.aug_multiple)

        while len(augmented_smiles) < target_length:  
            # Begin augmentation procedure
            for smiles in tokenized_smiles:
                # Construct the new string with randomly deleted characters
                tok_molecule = []
                for char in smiles:
                    # Either 'coinflip' says to delete or token belongs to list of protected tokens
                    if random.random() > self.prob or char in protection_list: 
                        tok_molecule.append(char)
                aug_molecule = ''.join(tok_molecule)
                # If valid flag, then SMILES string of aug_molecule needs to be able to be transformed into a molecular object.
                if valid:
                    mol = Chem.MolFromSmiles(aug_molecule, sanitize=False)
                    if mol:
                        if aug_molecule not in self.canonical_smiles and aug_molecule not in augmented_smiles:
                            augmented_smiles.append(aug_molecule)
                    else:
                        pass
                elif not valid:
                    augmented_smiles.append(aug_molecule)
        
        return augmented_smiles
