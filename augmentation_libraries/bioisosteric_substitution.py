

# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 18:00:06 2024

@author: 20213709
"""

# -*- coding: utf-8 -*-
"""
Created on Sat May 11 16:58:04 2024

@author: 20213709
"""

# -*- coding: utf-8 -*-
"""
Created on Fri May  3 16:56:53 2024

@author: 20213709
"""

import numpy as np
import pandas as pd
import copy

from rdkit import Chem
from rdkit.Chem import BRICS
import random
import os

file_dir = os.path.dirname(__file__)
os.chdir(file_dir)
 
class molecule():
    def __init__(self, mol, skeleton_mol, mol_char):
        self.m = mol
        self.smiles = Chem.MolToSmiles(mol)
        self.skeleton = skeleton_mol
        self.char =mol_char
    
    def to_dict(self):
        return {
            'skeleton': self.skeleton,
            'characterization': self.char,
            'mol':self.m,
            'smiles':self.smiles
        }
          
class fragment():

    """
    Processing of molecular fragments as RDKit Mol objects (fragment_mol). Extraction of information about the molecular 
    structure (with focus on 'dummy' atoms and their connectivity).
    """
    def __init__(self, fragment_mol):
        self.m = fragment_mol   # molecular object to be fragmented
        self.smiles = Chem.MolToSmiles(fragment_mol)    #SMILES string of the molecular object
        
        # Dictionary will map the indices of the dummy atoms to the indices of their bonded neighbour atoms
        dummy_bond_atom_dict = {}   
        implicit_atoms_list = []

        # Identification of dummy atom (looping thorugh all atoms)
        for d_atom in fragment_mol.GetAtoms():
            if not d_atom.GetAtomicNum():
                dummy_idx = d_atom.GetIdx() # Get index of dummy atom
                # Get the bond atom of the dummy atom
                bond_atom = d_atom.GetNeighbors()[0]
                neighbour_idx = bond_atom.GetIdx()
                dummy_bond_atom_dict[dummy_idx] = neighbour_idx
                # If dummy atom has the property 'implicit_atom', then add the value of this property
                if d_atom.HasProp('implicit_atom'):
                    implicit_atoms_list.append(d_atom.GetProp('implicit_atom'))
        
        self.implicit = implicit_atoms_list     # list of implicit atoms
        self.degree = len(dummy_bond_atom_dict)     # storing of number of dummy atoms
        self.dummy_bond = dummy_bond_atom_dict      # dictionary of mapping of dummy atom indices to neighbouring atom indices
        self.occurence = 1      # set intitial occurence count of the fragment
        self.Nats = fragment_mol.GetNumAtoms()  # total number of atoms in the fragment
    
    def to_dict(self):
        """ 
        Returns dictionary of the above determined characteristics in form of dictionary.
        """
        return {
            'degree': self.degree,
            'occurence': self.occurence,
            'frag_mol': self.m,
            'smiles': self.smiles,
            'Nats': self.Nats,
            'implicit': self.implicit
        }

class Bioisosters:
    def __init__(self, smiles_list, augmentation_multiple, prob):

        self.smiles_list = smiles_list
        self.canonical_smiles = [Chem.MolToSmiles(Chem.MolFromSmiles(smiles)) for smiles in self.smiles_list]
        self.aug_multiple = augmentation_multiple
        self.prob = prob

        # Probability needs to be between 0 (0 %) and 1 (100%)
        if not (0 <= self.prob <= 1):
            raise ValueError("Deletion probability must be between 0 and 1.")
            

    def BRICSfragmentation(self, m, mol_obj_list, frag_list):
        """
        Fragmentation of molecules (m) based on their BRICS fragments, an editable version (eMol) is created and dummy atoms are inserted 
        into the fragments between cleavable bonds. The editable molecular object is fragments and canonicalized to assure consistency and 
        tracking of reconstruction. A dictionary (frag_index_dict) is created to map each fragment to an index and a list (fragment_molecule_to_list)
        is create to further categorize the fragments. Duplicates are marked as such, so that no duplicates are available.
        The skeleton of the molecule is rebuild, where the dummy atoms are mapped into their original position, and then fragments are added
        through core atoms to create the connectivity between dummy and core atoms.
        The dummy atoms are connected with the original atoms with their original bond types with a dictionary (mapping_dict). Lastly 
        Lastly, a dictionary is created (mol_obj) with information about the skeleton, original molecule, fragments and characterization and
        added to a list (mol_obj_list).
        arguments 
        m: molecular object
        mol_obj_list: list of dictionary with mol_obj with indixes skeleton, characterization, mol, smiles
        frag_list: list of non-duplicated fragments
        returns
        mol_obj_list, frag_list
        """

        # Find the bonds in a molecule that BRICS would cleave
        bonds = BRICS.FindBRICSBonds(m, randomizeOrder=False, silent=True)
        eMol = Chem.RWMol(m)    # Makes it possible to edit the molecular object: (e)ditable (Mol)ecule

        bond_index = 0
        eMol.BeginBatchEdit()   # Start of batch editing
        # Parsing of the BRICS bonds
        for indices, dummyTypes in bonds:
            # Store information from the BRICS bonds that can be cleaved
            ia,ib = indices
            obond = m.GetBondBetweenAtoms(ia,ib)
            bondType = obond.GetBondType()  # bond type: single, double or triple

            # Removing of the cleavable bonds
            eMol.RemoveBond(ia, ib)

            # Creating the dummy atom with no implicit H, bond index 0 and setting it where ib was
            atoma = Chem.Atom(0)    
            atoma.SetNoImplicit(True)
            atoma.SetIntProp('bond_index',(bond_index))
            atoma.SetProp('implicit_atom',m.GetAtomWithIdx(ib).GetSymbol())
            # Adding of new dummy atom and new bond
            idxa = eMol.AddAtom(atoma)
            eMol.AddBond(ia,idxa,bondType)

            # Add matching dummy on the other side of the bond fracture as before
            atomb = Chem.Atom(0)
            atomb.SetNoImplicit(True)
            atomb.SetIntProp('bond_index',(bond_index))
            atomb.SetProp('implicit_atom', m.GetAtomWithIdx(ia).GetSymbol())
            idxb = eMol.AddAtom(atomb)
            eMol.AddBond(ib,idxb,bondType)

            bond_index += 1 # Next bond_index

        # End manipulation of molecular structure
        eMol.CommitBatchEdit()  
        mol = eMol.GetMol()

        # Creation of mol_frags of the editable molecular object to fragment a molecule and track the mapping of the atoms to the resulting fragments. 
        frag_map = []
        mol_frags = Chem.GetMolFrags(mol, asMols=True, fragsMolAtomMapping=frag_map)
        
        # Canonicalize the indexing in the fragments:
        canonical_frags = []
        for frag in mol_frags:
            # Clean the mapping first
            for i, atom in enumerate(frag.GetAtoms()):
                atom.SetAtomMapNum(0)   # Clear the atom mapping number to zero to prevent that previous mapping inteferes with canonicalisation process
            # Canonicalize the molecule
            canon_smiles = Chem.MolToSmiles(frag)
            canonic_mol = Chem.MolFromSmiles(canon_smiles)

            # Map the properties back to the new canonicalized molecule fragment
            new_order = list(frag.GetPropsAsDict(includePrivate=True,includeComputed=True)['_smilesAtomOutputOrder'])   # Catches new order and propertiesof canonicalised molecular object ('_smilesAtomOutputOrder')
            
            # Mapping of each atom to find dummy atoms
            for d_atom in frag.GetAtoms():
                if not d_atom.GetAtomicNum():   # Checks if the atomic number is 0, so a dummy atom
                    # Get all the properties of the old dummy atom
                    d_idx = d_atom.GetIdx() # Index of dummy in the original fragment
                    bond_index_holder = d_atom.GetIntProp('bond_index') # Bond index
                    implicit = d_atom.GetProp('implicit_atom')  # Stores the symbol of the original atom involved in the broken bond

                    # Get new index
                    new_index = new_order.index(d_idx)  # Finds position in new order
                    # Assign all properties to new dummy in canonicalized molecule
                    new_dummy = canonic_mol.GetAtomWithIdx(new_index)
                    new_dummy.SetIntProp('bond_index',(bond_index_holder))
                    new_dummy.SetProp('implicit_atom', implicit)
            
            canonical_frags.append(canonic_mol)
        
        # Check all the fragments and get their frag_index
        frag_index_dict = {}

        # Create characterization for molecule object
        mol_char = []
        for frag in canonical_frags:
            frag_index, frag_list = self.fragment_molecule_to_list(frag, frag_list)  # Calls function fragment_molecule_to_list
            frag_index_dict[frag] = frag_index  # adding to the dictionary of fragments
            mol_char.append(frag_index)

        # Create a skeleton molecule with all the information needed to rebuild the molecule from the fragments
        blank_mol = Chem.Mol()
        blank = Chem.RWMol(blank_mol)
        blank.BeginBatchEdit()
        # Make dict mapping old idx to new idx
        mapping_dict = {}
        # Initiate the dictionary
        for k in range(bond_index):
            mapping_dict[k] = []    # Start empty list
        for frag in canonical_frags:
            frag_index= frag_index_dict[frag]   # Which index the fragment has
            # Add core atom for fragment
            core_atom = Chem.Atom(0)    # Creating of new atom
            core_atom.SetIntProp('frag_index',frag_index)  # Set integer property (index), tagging the dummy atom with the fragment unique identifier
            core_atom.SetIntProp('core_bool', 1)  
            core_atom.SetIsotope(frag_index)
            core_idx = blank.AddAtom(core_atom)

            # Get all dummy atom indexes in the query
            for d_atom in frag.GetAtoms():
                if not d_atom.GetAtomicNum():
                    dummy_idx = d_atom.GetIdx()
                    side_atom = Chem.Atom(0)    # Creating of new dummy atom
                    side_atom.SetIntProp('frag_index', frag_index)  
                    side_atom.SetIntProp('dummy_index', dummy_idx)  
                    side_atom.SetIntProp('core_bool', 0)    
                    idxa = blank.AddAtom(side_atom)
                    blank.AddBond(core_idx, idxa)
                    
                    # Map added atom back to the bond it signifies in the parent molecule
                    bond_index_holder1 = d_atom.GetIntProp('bond_index')
                    mapping_dict[bond_index_holder1].append(idxa)

        # Add original bonds in the mask molecule
        bonds = BRICS.FindBRICSBonds(m, randomizeOrder=False, silent=True)  # Find again the cleavable bonds
        bond_index2 = 0
        for indices, dummyTypes in bonds:
            # Store information from the BRICSbonds
            ia,ib = indices
            obond = m.GetBondBetweenAtoms(ia,ib)
            bondType = obond.GetBondType()
            new_ia, new_ib = mapping_dict[bond_index2]
            blank.AddBond(new_ia, new_ib, bondType)
            bond_index2 += 1
        blank.CommitBatchEdit()
        skeleton_mol = blank.GetMol()
        mol_obj = molecule(m, skeleton_mol, mol_char)
        mol_obj_list.append(mol_obj)
        
        return mol_obj_list, frag_list

    def fragment_molecule_to_list(self, frag_mol, frag_list):
        """
        Takes a fragmented molecular object and processes the molecule to see if it is already in the fragment list.
        If not, adds to the list. If molecule already in the list, then increase of occurance. At the end, the index of the
        fragment (wheter newly added or existing) and the list of updated fragments (frag_list) are returned.
        arguments
        frag_mol: fragmented molecular object
        frag_list: list of fragments
        returns
        frag_index, frag_list
        """

        frag_obj = fragment(frag_mol)   # Creation of fragment object
        
        # Checking if duplicates is empty or duplicates_implicit is empty added
        # Creates the list of fragments that have the same SMILES string and implicit properties
        duplicates = [frag_obj2 for frag_obj2 in frag_list if frag_obj2.smiles == frag_obj.smiles and frag_obj2.implicit == frag_obj.implicit]

        if len(duplicates) == 0:    # No duplicated, adding of the frag_obj to the list
            frag_list.append(frag_obj)
            frag_index = frag_list.index(frag_obj)  # Get index of newly added fragment in the list
        else:
            duplicates[0].occurence += 1    # fragment is already in the list
            frag_index = frag_list.index(duplicates[0]) # index of the already existing fragment

        return frag_index, frag_list

    def back_substitution(self, skeleton_mol, frag_list):
        """
        
        Reconstruction of molecular structure (skeleton_mol) by replacing core atoms (placeholders) in a skeleton with corresponding fragments of a
        list. The atoms of the skeleton molecule are iterated and determined if they are a core atom. If they are a core atom, the fragment 
        index (frag_index) is retrieved and with the help of the fragments list (frag_list), retrieves a fragment to combine to a new molecule.
        Afterwards, the neighbours of of the core atom are parsed and for each neighbour, the dummy atom is retrieved and removed.
        The bond between non-core neighbours and the side atom are identified and added to the skeleton of the appropiate atoms.
        The core and side atoms are removed.

        arguments
        skeleton_mol: Skeleton of molecular object.
        frag_list: Fragment list ot be inserted into skeleton
        returns
        iterator_mol
        
        """
        iterator_mol = skeleton_mol # Copy of the molecule that will be modified
        core_bool = True
        while core_bool == True:
            core_bool = False   # Stops after one iteration, should only continue if it finds a core atom of all atoms

            # Find the core atoms in the skeleton molecule
            for atom in iterator_mol.GetAtoms():
                # If atom is a core atom
                if atom.HasProp('core_bool') and atom.GetIntProp('core_bool') == 1:
                    core_bool = True
                    core_atom = atom
                    frag_index = core_atom.GetIntProp('frag_index')     # Retrieve fragmentation index
                    # Combine the current iterator mol with the corresponding fragment of the core atom
                    combined_mol = Chem.CombineMols(frag_list[frag_index].m, iterator_mol)  # Cobmination of fragment and molecule
                    res_combined= Chem.RWMol(combined_mol)  # Converting combined molecule into editable object
                    frag_Nats = frag_list[frag_index].m.GetNumAtoms()   # Number atoms of fragment from fragment list
                    res_combined.BeginBatchEdit()   # Start editing the skeleton

                    # Loop through all the side_atoms around the core atom.
                    for side_atom in core_atom.GetNeighbors():
                        #get the number of the corresponding dummy in the fragment.
                        side_idx = side_atom.GetIdx()   # Get idx of side atom
                        # Retrieving the dummy atom of the fragment
                        dummy_binder_idx = side_atom.GetIntProp('dummy_index') 
                        ib = frag_list[frag_index].dummy_bond[dummy_binder_idx]
                        # Remove dummy atom from fragment
                        res_combined.RemoveAtom(dummy_binder_idx)

                        # Loop through the neighbors in side atom / Retrieving bonds between non-core neighbours and the side atom
                        for neighbor in side_atom.GetNeighbors():
                            # Get non-core atom neighbors
                            if neighbor.HasProp('core_bool') == False or neighbor.GetIntProp('core_bool') == 0:
                                # Storing of information
                                ia = neighbor.GetIdx()
                                obond = iterator_mol.GetBondBetweenAtoms(ia,side_idx)
                                bondType= obond.GetBondType()

                        # Add bond between appropiate atoms in res_combined
                        res_combined.AddBond(ia + frag_Nats, ib, bondType)

                    # Remove the core atom and its neighbours from the new molecule
                    core_idx = core_atom.GetIdx() + frag_Nats
                    res_combined.RemoveAtom(core_idx)
                    for side_atom in core_atom.GetNeighbors():
                        side_idx = side_atom.GetIdx() + frag_Nats
                        res_combined.RemoveAtom(side_idx)    
                    res_combined.CommitBatchEdit()
                    iterator_mol = res_combined.GetMol()
                    # Break and go to next while loop to refresh the loop for the new iterator_mol
                    break
        return iterator_mol

    def augmentation_substitution(self, skeleton_mol, core_index, replace_dict, frag_list, aug_frags_list):     
        """
        Performs the substitution on the skeleton molecule (skeleton_mol) with dummy atoms between functional groups.
        With the skeleton molecules, the core atoms (core_index) are retrieved. The fragment selection takes place by
        retrieving the fragment index from the core atom properties, and uses the index for the dict (replace_dict)
        which contains possible replacement fragments and frequencies. There are normalized towards a probability
        distribution and with the help of a cumulative distribution approach, a fragment is chosen.
        The fragment is converted to a molecular object (aug_frag_mol), and dummy atoms are identified in the new fragments and
        mapped towards the original fragment dummy atoms.
        The new fragment and the skeleton molecule are combined.
        Afterwards, the new molecule is iterated through the neighbours of the core atom, mapped and dummy atoms are replaced,
        while simunteneaosuly adding new bonds. Core atom and neighbours are removed from the new molecule.
        arguments
        skeleton_mol: Skeleton molecule to be substituted
        core_index: Index of atom within the molecule to be interchanged
        replace_dict: Dictionary of the fragment to be substituted
        returns
        iterator_mol
        """
        iterator_mol = skeleton_mol
        core_atom = iterator_mol.GetAtomWithIdx(core_index)     # Gets the atom with the idx
                        
        frag_index = core_atom.GetIntProp('frag_index') # Return the value of the property
        # Read in the query csv from the fragment
        df_aug_frags = replace_dict[frag_index]
        
        # Extract the frequencies from each bio-isostere 
        frequencies = list(df_aug_frags['Frequency'][0:4].copy())   # Highest frquencies
        # Parse through frquencies
        for k in range(len(frequencies)):
            frequency = frequencies[k]
            frequencies[k] = float(frequency.split()[0])
        # Normalize all frequencias as probabilities
        probabilities = [x / sum(frequencies) for x in frequencies]
        
        # Choose index randomly based on distribution probabilities
        cumulative_distribution = np.cumsum(probabilities)  # Calculation of cumulative probabilties
        cumulative_distribution /= cumulative_distribution[-1]  # Ensures that cumulative distribution sums to 1
        uniform_samples = np.random.rand(1)     # Generation of single random number between 0 and 1

        # Selecting of a index for replacement where uniform_samples would fit into the cumulative distribution
        # array to mantain order
        chosen_frag_idx = np.searchsorted(cumulative_distribution, uniform_samples, side="right")   # 'right' in case of tie, right matched
        aug_frag_smiles = df_aug_frags['Candidate Fragments'][chosen_frag_idx[0]]   # Corresponding fragment of the chosen index
        aug_frag_mol = Chem.MolFromSmiles(aug_frag_smiles)  # Convert the fragment to molecular object
              
        # Make class object of the fragment
        aug_frag_obj = fragment(aug_frag_mol)

        # Get the indexes of the r-groups in replacement fragment
        r_group_idxs = []
        for d_atom in aug_frag_obj.m.GetAtoms():    # Looping through atoms in the molecular object
            # Identification of the dummy atoms
            if not d_atom.GetAtomicNum() and d_atom.HasProp('molAtomMapNumber'):
                r_group_idxs.append(d_atom.GetIdx())

        original_frag = frag_list[frag_index]

        # Get the order of the dummy atoms in the smiles since R numbers are assigned in that order.
        # Create dict mapping the order of the dummy to the new dummy atoms.
        # Mapping and replacing dummy atoms.
        rgroup_mapping_dict = {}
        k = 0
        for d_atom in original_frag.m.GetAtoms():          
            if not d_atom.GetAtomicNum():
                dummy_idx = d_atom.GetIdx()
                rgroup_mapping_dict[dummy_idx] = r_group_idxs[k]
                k += 1

        # Combine the current iterator mol with the corresponding fragment of the core atom. 
        # This is done by adding and removing bonds.
        combined_mol = Chem.CombineMols(aug_frag_obj.m, iterator_mol)
        res_combined= Chem.RWMol(combined_mol)
        frag_Nats = aug_frag_obj.m.GetNumAtoms()    # Nats = Number of atoms in augmented fragment
        # Start editing of the molecule
        res_combined.BeginBatchEdit()
        # Loop through all the side_atoms around the core atom. This means iteratore over neighbours of core atom.
        for side_atom in core_atom.GetNeighbors():  # Iterate over direct bonded atoms
            # Get the number of the corresponding dummy in the fragment.
            side_idx = side_atom.GetIdx()   # Index of neighbouring atom
            old_rgroup_idx = side_atom.GetIntProp('dummy_index')    # Which dummy atom was originally bonded to fragment.
            # Map the old dummy idx to new dummy idx
            dummy_binder_idx = rgroup_mapping_dict[old_rgroup_idx]  # Index of dummy atom in the new fragment that will replace the old dummy atom
            ib = aug_frag_obj.dummy_bond[dummy_binder_idx]
            # Remove dummy atom from fragment
            res_combined.RemoveAtom(dummy_binder_idx)

            # Loop through the neighbors in side atom
            # Recreates bonds for the new fragment.
            for neighbor in side_atom.GetNeighbors():   # Is bonded to side atom and not core atom
                # Get non-core atom neighbors
                if neighbor.HasProp('core_bool') == False or neighbor.GetIntProp('core_bool') == 0:
                    ia = neighbor.GetIdx()  # Index of neighbouring atom
                    obond = iterator_mol.GetBondBetweenAtoms(ia, side_idx)  # Bond between ia, side_idx in original skeleton.
                    bondType= obond.GetBondType()

            # Adding of bond between neighbouring atom in the new fragment and the dummy atom
            res_combined.AddBond(ia + frag_Nats, ib, bondType)

        # Remove the core atom and its neighbours from the new molecule.
        core_idx = core_atom.GetIdx() + frag_Nats
        res_combined.RemoveAtom(core_idx)
        for side_atom in core_atom.GetNeighbors():
            side_idx = side_atom.GetIdx()+ frag_Nats
            res_combined.RemoveAtom(side_idx)

        res_combined.CommitBatchEdit()
        iterator_mol = res_combined.GetMol()   
        
        return iterator_mol

    def perform_augmentation(self, aug_frags_list_dict):     
        """
        Pipeline of the substitution to bioisosters.
        First step is the elimination of stereochemistry, as the v.1 of the augmentation method does not include stereochemistry.
        Afterwards, the molecules are converted into the molecular object with the BRICSfragmentation, and the list of fragmented molecules
        is parsed towards available fragmentations. The substitution dictionaries are read in, then the fragmentation begins.
        Here, while the number of augmented and original SMILES strings are less than len(smiles_list) * augmentation_multiple, we iterate
        towards the molecular objects to do fragmented substitutions.
        First, the molecule is parsed to see if each molecule has augmentable fragments. If yes, based on a probability, it is chosen which
        core atom is substituted. When substitution, the core atom list is updated, and back substitution is performed and transformed into
        a SMILES strings and added to the list (if it is not already in the list).
        
        arguments
        smiles_list: list of SMILES strings to be substituted
        aug_frags_list_dict: dictionary of possible fragmentations
        prob: probability that an fragment is augmented
        aug_multiple: how many times substitution can take place
        returns
        augmented_smiles, mol_obj_list, frags_list
        """
      
        # Transition of SMILES into molecular objects
        mol_list = [Chem.MolFromSmiles(smiles) for smiles in self.smiles_list]

        # Initialization of the augmented SMILES, fragment and molecular obejct list
        frags_list, mol_obj_list = [], []

        augmented_smiles = copy.deepcopy(self.smiles_list)
        target_length = (len(self.smiles_list) * self.aug_multiple)

        # Fragmentation of each molecule and fragmentation list with the BRICSfragmentation function
        for m in mol_list:
            mol_obj_list, frags_list = self.BRICSfragmentation(m, mol_obj_list, frags_list)

        # Get the new indexes of the structures.
        excel_name_dict = {}
        aug_frags_list = []
        # Iteration of fragments list to see if it in the list of augmentable fragments and if its implicit
        # properties contain only 'C'
        for k in range(len(frags_list)):
            frag = frags_list[k]
            # Checks if the fragment is in the list of augmentatable fragments, or if the implicit attribute of
            # frag is a set containing only the string 'C'
            # Implicit C: SwissBioisosteres only looks at fragments that are bound to a carbon, then when looking for
            # the dummy, it needs to be bound to a carbon atom 
            if frag.smiles in aug_frags_list_dict.keys() and set(frag.implicit) == {'C'}: 
                aug_frags_list.append(k)
                excel_name_index = aug_frags_list_dict[frag.smiles]     # Index of fragment to be augmented
                excel_name_dict[k] = excel_name_index   # id to id parsing
        
        # Read in the substitution dictionary of indexes to substitute:
        replace_dict = {}
        for frag_index in aug_frags_list:
            excel_name_index = excel_name_dict[frag_index]
            csv_name = f'{file_dir}/substitution_groups/SwissBioisostere_fragment_output_{excel_name_index}.csv'
            df_aug_frags = pd.read_csv(csv_name, skiprows=[0]).dropna()
            replace_dict[frag_index] = df_aug_frags

        # Create a random permutation of the obj_list to avoid biases towards input order.
        obj_indexes = [i for i in range(len(mol_obj_list))]
        random.shuffle(obj_indexes)
        
        no_change_counter = 0
        current_length = len(augmented_smiles)

        # Start of the augmentation
        while len(augmented_smiles) < target_length:   # len(self.smiles_list) +   # !=
            
            # Substitute each molecule and only append it to the list if it is changed
            for rand_obj in obj_indexes:
                # Get the current mol object
                mol_obj = mol_obj_list[rand_obj]
                # Check if molecule has augmentable fragments
                substitute_bool = False
                for frag_idx in mol_obj.char:
                    if frag_idx in aug_frags_list: substitute_bool = True   # Do substitution
        
                if substitute_bool == True: 
                # Keep substituting till the random number breaks the loop.
                    iterator_mol = mol_obj.skeleton
            
                    # Choose which of the augmentable core atoms to substitute
                    skeleton_aug_char = []
                    for atom in iterator_mol.GetAtoms():
                        if atom.HasProp('core_bool') and atom.GetIntProp('core_bool') == 1:
                            core_atom = atom
                            frag_index = core_atom.GetIntProp('frag_index')
                            # Only append the core_idxs of cores with substitutable frag_indexes
                            if frag_index in aug_frags_list:
                                skeleton_aug_char.append(core_atom.GetIdx())    # Get the id of the atom to substitute
                                
                    skeleton_aug_char_original = skeleton_aug_char
                    # Check if there are augmentable fragments present
                    if len(skeleton_aug_char_original) != 0:
                        
                        skeleton_aug_char_indexes = [i for i in range(len(skeleton_aug_char))]
                        # Generate a random permutation of the augmentable core atoms to avoid bias towards input order.
                        random.shuffle(skeleton_aug_char_indexes)
                        # Loop through all augmentable fragments and augment with prob.
                        for k in range(len(skeleton_aug_char_original)):
                            random_skeleton_aug_idx = skeleton_aug_char_indexes[k]
                            rand_frag = skeleton_aug_char[random_skeleton_aug_idx]
                            # 'Coinflip' to choose if it gets augmented or not
                            rand_numb = random.random()
                            if rand_numb <= self.prob:
                                core_index = rand_frag
                                # Calls the function which performs the augmentation
                                iterator_mol = self.augmentation_substitution(iterator_mol, core_index, replace_dict, frags_list, aug_frags_list)   
                                
                                # Replace augmented core atom from orignal indexes list
                                skeleton_aug_char_original[random_skeleton_aug_idx] = 'removed'
                                
                                # Find the new indexes of the core atoms after adding a structure and map them back to the original list.
                                # Order does not change when augmenting, so they can be mapped back based on order
                                skeleton_aug_char=[]
                                for atom in iterator_mol.GetAtoms():
                                    if atom.HasProp('core_bool') and atom.GetIntProp('core_bool') == 1:
                                        core_atom = atom
                                        frag_index = core_atom.GetIntProp('frag_index')
                                        # Only append the core_idxs of cores with substitutable frag_indexes.
                                        if frag_index in aug_frags_list:
                                            skeleton_aug_char.append(core_atom.GetIdx()) 
                                
                                # Get all instinces where the core atom index was removed and add them back to the new list.
                                indices = [i for i, x in enumerate(skeleton_aug_char_original) if x == "removed"]
                                for removed_idx in indices:
                                    skeleton_aug_char.insert(removed_idx, 'removed')
                                
                    # Substitute all original fragments that are not augmented back. 
                    final_substituted_mol = self.back_substitution(iterator_mol, frags_list)
        
                    # Only add the changed molecules to the dataframe
                    final_substituted_smiles = Chem.MolToSmiles(final_substituted_mol)
                    if final_substituted_smiles not in augmented_smiles and final_substituted_smiles not in self.canonical_smiles: 
                        augmented_smiles.append(final_substituted_smiles)
                        
            # Check if the length has changed
            if len(augmented_smiles) == current_length:
                no_change_counter += 1
            else:
                no_change_counter = 0  # Reset the counter if the length changes

            # Break the loop if no change occurs for 1000 iterations
            if no_change_counter >= 1000:
                break

            current_length = len(augmented_smiles) 

        return augmented_smiles 