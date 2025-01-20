import os
file_dir = os.path.dirname(__file__)
os.chdir(file_dir)

from rdkit.Chem import RWMol
from rdkit import Chem
import random
import copy



class RandomMasking:

    def __init__(self, smiles_list, augmentation_multiple, prob):

        self.smiles_list = smiles_list
        self.canonical_smiles = [Chem.MolToSmiles(Chem.MolFromSmiles(smiles)) for smiles in self.smiles_list]
        self.aug_multiple = augmentation_multiple
        self.prob = prob

        if not (0 <= self.prob <= 1):
            raise ValueError("Deletion probability must be between 0 and 1.")
        
    def mask_mol(self, mol):
        """
        Random masking of any molecule (mol). First, the molecules is converted into an editable molecular object.
        Next, every atom in the molecule is parsed and with a probability (prob), the atom is replaced by a dummy atom.
        The molecule (masked_mol) is afterwards checked towards sanity.
        arguments
        mol: Molecular object to be masked.
        prob: Probability of the masking per atom.
        returns
        masked_mol
        """

        res = Chem.RWMol(mol) 
        atoma = Chem.Atom(0)    # Dummy atom
        # Start a batch edit for random masking of the molecules 
        res.BeginBatchEdit()
        # Remove all the atoms of the substructure and replace them with a dummy atom
        for atom in mol.GetAtoms():
            aid = atom.GetIdx()
            if random.random() <= self.prob: 
                res.ReplaceAtom(aid, atoma) # Atom id is replaces by dummy atom id.
        # Commit all batch changes at once    
        res.CommitBatchEdit()
        # Try to sanitize the new molecule properly.
        sanitFail = Chem.SanitizeMol(res,catchErrors=True)
        if sanitFail: return 'Molecule with random atom masking could not be sanitized properly.'   # error
        #Chem.SanitizeMol(res)    
        masked_mol = res.GetMol()
        return masked_mol

    def perform_augmentation(self, *args):
        """
        Performing of data augmentation based on random masking function. The molecules are converted to molecular
        objects, suffled. Data augmentation preceeds as long as the sum of the original and augmented smiles list is
        shorter than the augmentation fold.
        arguments
        none
        returns
        total_augmented_smiles, total_target_smiles 
        """

        mols = [Chem.MolFromSmiles(x) for x in self.smiles_list]
        random.shuffle(mols)

        augmented_smiles = copy.deepcopy(self.smiles_list)
        target_smiles = []

        target_length = (len(self.smiles_list) * self.aug_multiple)

        while len(augmented_smiles) < target_length:
                     
            # Start parsing towards each molecular object
            for mol in mols:
                original_mol = mol
                # Mask the atoms randomly in the 
                masked_mol  = self.mask_mol(original_mol)
                        
                if masked_mol != 'no_match' and masked_mol !='Molecule with random atom masking could not be sanitized properly.':
                    # # Count the amount of times the molecule was masked
                    # noise_ratio = Chem.MolToSmiles(masked_mol).count('*') / len(Chem.MolToSmiles(masked_mol))

                    final_aug_smiles = Chem.MolToSmiles(masked_mol, canonical=False) #canonical=False
                    original_smiles = Chem.MolToSmiles(original_mol, canonical=False)

                    if final_aug_smiles not in augmented_smiles and final_aug_smiles not in self.canonical_smiles and len(final_aug_smiles) == len(original_smiles):  # 
                        wrong = []
                        # In case the SMILES length are not the same, sanitz check where it fails
                        for letter_idx in range(len(original_smiles)):
                            letter = final_aug_smiles[letter_idx]
                            targ_letter = original_smiles[letter_idx]
                            if targ_letter != letter:
                                wrong.append(letter)

                        # If the masking is the only difference between the lengths 
                        if set(wrong) == {'*'}: 
                            augmented_smiles.append(final_aug_smiles)
                            target_smiles.append(Chem.MolToSmiles(original_mol))
                            mol = masked_mol
                
        total_target_smiles = self.smiles_list + target_smiles

        return augmented_smiles, total_target_smiles 
