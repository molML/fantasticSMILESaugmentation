from rdkit.Chem import RWMol
from random import randrange
from rdkit import Chem
import random
import copy
import os

file_dir = os.path.dirname(__file__)
os.chdir(file_dir)

class GroupMasking:

    def __init__(self, smiles_list, augmentation_multiple, prob):

        self.smiles_list = smiles_list
        self.canonical_smiles = [Chem.MolToSmiles(Chem.MolFromSmiles(smiles)) for smiles in self.smiles_list]
        self.aug_multiple = augmentation_multiple
        self.prob = prob

        if not (0 <= self.prob <= 1):
            raise ValueError("Deletion probability must be between 0 and 1.")
        
    def get_connection_bonds(self, mol, query_mol):
        """
        Determines the connection bonds of the molecular object towards the specific functional group. After determining if the functional
        group (query_mol) is within the molecule (mol), tuples are created with the matching atom id belonging to the bonds (atom_match). Afterwards,
        neighbours / connection bond to each of the atom ids are calculated and the id of the neighbour is added to a list (connection_bonds).
        arguemtns
        mol: Molecular object
        query_mol: Functional group
        returns
        connection_bonds, atom_match

        """
                
        # Substructure search on group retrieves all matching atoms
        # Returns tuple of the indices of the molecule's atom that match a substructure query
        matching_indices_list = mol.GetSubstructMatches(query_mol)
        if not matching_indices_list: return (False, 'no_match')    # Handles the case that it does not have the substructure
        
        # Retrieve one of the indexes of random groups in all the matching groups (random retrival)
        rand_idx = randrange(len(matching_indices_list))
        # Retrieves the matching tuple
        atom_match = matching_indices_list[rand_idx]
                    
        # Determination of connection bonds of the query    
        connection_bonds = []
        # If no dummies are present in the query match we need to find bonds connecting query to structure.
        for atom_idx in atom_match:
            atom = mol.GetAtomWithIdx(atom_idx)
            neighbours = atom.GetNeighbors()
            for neighbour in neighbours:
                # If a neighbour of atom is not part of the matched substructure add the bond as connection_bond.
                # The connecting bonds are bonded to the match structure but are not part of the structure.
                if neighbour.GetIdx() not in atom_match:
                    connection_bonds.append((atom_idx, neighbour.GetIdx())) # First it the atom with the matching substructure, second is neighbour
                    
        return connection_bonds, atom_match

    def replace_substruct(self, atom_match_no_dummy, mol):
        """
        
        Replaces the substructure of the atom matches with a dummy atom. After the molecular object (mol) is transformed into a editable
        one, each susbtructure match (atom_match_no_dummy) is parsed and based on a probability, the atom is substituted with a dummy atom 
        so we have a masked molecule (mask_mol).
        arguments
        atom_match_no_dummy: list of atom with matched substructure
        mol: molecular object
        prob: probability of masking
        returns
        masked_mol
        
        """
        res = Chem.RWMol(mol) 
        atoma = Chem.Atom(0)
        # Start a batch edit
        res.BeginBatchEdit()
        # Replace all the atoms of the substructure with a dummy atom
        for aid in atom_match_no_dummy:
            if random.random() <= self.prob:
                res.ReplaceAtom(aid, atoma)
        # Commit all batch changes at once    
        res.CommitBatchEdit()
        sanitFail = Chem.SanitizeMol(res, catchErrors=True)
        if sanitFail: return 'error'   
        masked_mol = res.GetMol()
        return masked_mol

    def mask_mol(self, mol, query_mol):
        """

        Masking of the molecular object. First, the connection bonds and atoms matching the substructure (atom_match_no_dummy) are retrieved 
        with the get_connection_bonds function, and replaced with the replace_substructure function. A masked molecular object is returned
        (masked_mol).
        arguments
        mol: Molecular object
        query_mol: Substructure that can be replaced or masked.
        prob: Probability of replacement
        returns
        masked_mol, atom_match_no_dummy

        """

        connection_bonds, atom_match_no_dummy = self.get_connection_bonds(mol, query_mol)
        if connection_bonds == False:   # If no connection bonds, returns tuple of no_match
            return 'no_match', 'no_match'
        
        # Replace each atom in the substruct with a dummy atom with a probability
        masked_mol= self.replace_substruct(atom_match_no_dummy, mol)
        return masked_mol, atom_match_no_dummy

    def perform_augmentation(self, query_list):
        """

        Data augmenation for group masking. Instead of random masking atoms, only atoms from functional groups will be masked. 
        After converting the SMILES strings (smiles_list) and SMARTS strings of the substructures (query_list) to molecular objects, the 
        augmentation starts. It is performed until the length of the SMILES and augmented SMILES is equal or bigger the augemntation fold.
        the query_list of functional groups to molecular objects.
        The list of query molecular objects is suffled, and then with the function mask_mol they are parsed.
        arguments
        query_list:
        returns
        masked molecule
        """
        # Transform the SMILES string to molecular objects
        mols = [Chem.MolFromSmiles(x) for x in self.smiles_list]
        
        # Transform the SMARTS of the functional groups to molecular objects
        query_mols = []
        for x in query_list:
            query_mol = Chem.MolFromSmarts(x)
            if query_mol != None:
                query_mols.append(query_mol)

        # Start of augmentation method
        target_smiles, error_list = [], []
        augmented_smiles = copy.deepcopy(self.smiles_list)
        target_length = (len(self.smiles_list) * self.aug_multiple)

        no_change_counter = 0
        current_length = len(augmented_smiles)

        while len(augmented_smiles) < target_length:
            idx = 0
            # Start parsing of the molecular objects
            for mol in mols:
                original_mol = mol 
                total_matches = 0   # Total times the molecule could be masked
                # Create indexes for shuffling of them.
                indexes = [[i] for i in range(len(query_mols))]
                random.shuffle(indexes)
                # Extract random group from groups_list
                for query_idx in indexes:
                    query_mol = query_mols[query_idx[0]]
                    masked_mol, atom_match_no_dummy = self.mask_mol(mol, query_mol)
                    
                    # If valid molecule with matched substructures
                    if masked_mol != 'no_match' and masked_mol != 'error':
                        mol = masked_mol
                        total_matches += len(atom_match_no_dummy)   # Incrementation of total matchers with the length of matched substrctures tuples
                        if total_matches > len(Chem.MolToSmiles(masked_mol)): # If the length is bigger than the len of the masked mol, then break
                            break
                        
                    if masked_mol == 'error':   # Not valid molecule is appended to the error_list
                        error_list.append((idx, query_idx))

                # If valid molecule with matched substructures        
                if masked_mol != 'no_match' and masked_mol != 'error':
                    noise_ratio = Chem.MolToSmiles(masked_mol).count('*')/len(Chem.MolToSmiles(masked_mol))
                    # Canonical = false, so that the molecules are the same as the original ones in the order
                    # The order of the SMILES are kept the same!
                    final_aug_smiles = Chem.MolToSmiles(masked_mol, canonical=False) 
                    original_smiles = Chem.MolToSmiles(original_mol, canonical=False)

                    # Check is masked molecule already in augmented list or smiles list and the length of the original and augmented molecule
                    # is the same
                    if final_aug_smiles not in augmented_smiles and final_aug_smiles not in self.canonical_smiles and len(final_aug_smiles) == len(original_smiles):      # and final_aug_smiles not in smiles_list
                        wrong = []
                        # Sanity check: At each position, the molecules need to be the same, if the difference is only the masking, then right one.
                        for letter_idx in range(len(original_smiles)):
                            letter = final_aug_smiles[letter_idx]
                            targ_letter = original_smiles[letter_idx]
                            if targ_letter!= letter:
                                wrong.append(letter)
                        if set(wrong) == {'*'}:
                            augmented_smiles.append(final_aug_smiles)
                            target_smiles.append(Chem.MolToSmiles(original_mol))
                            mol = masked_mol

                idx += 1
            
            # Check if the length has changed
            if len(augmented_smiles) == current_length:
                no_change_counter += 1
            else:
                no_change_counter = 0  # Reset the counter if the length changes

            # Break the loop if no change occurs for 1000 iterations
            if no_change_counter >= 1000:
                break

            current_length = len(augmented_smiles) 

        # total_augmented_smiles = augmented_smiles + smiles_list
        total_target_smiles = self.smiles_list + target_smiles

        return augmented_smiles, total_target_smiles 