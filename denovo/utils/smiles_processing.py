import re
from typing import List
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.SaltRemover import SaltRemover

from chemistry import is_element

RDLogger.DisableLog("rdApp.*")

_ELEMENTS_STR = r"(?<=\[)Cs(?=\])|Si|Xe|Ba|Rb|Ra|Sr|Dy|Li|Kr|Bi|Mn|He|Am|Pu|Cm|Pm|Ne|Th|Ni|Pr|Fe|Lu|Pa|Fm|Tm|Tb|Er|Be|Al|Gd|Eu|te|As|Pt|Lr|Sm|Ca|La|Ti|Te|Ac|Cf|Rf|Na|Cu|Au|Nd|Ag|Se|se|Zn|Mg|Br|Cl|Pb|U|V|K|C|B|H|N|O|S|P|F|I|b|c|n|o|s|p"
__REGEXES = {
    "segmentation": rf"(\[[^\]]+]|{_ELEMENTS_STR}|"     # All characters included within a braket [ XX ]
    + r"\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%\d{2}|\d)",
    "segmentation_sq": rf"(\[|\]|{_ELEMENTS_STR}|"
    + r"\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%\d{2}|\d)",    # All possible characters that can be used inclusive brackets
    "leading_mass": rf"\[\d+({_ELEMENTS_STR})",     # Ions or brackets elements [ X ...
    "solo_element": rf"\[({_ELEMENTS_STR})\]",      # Elements in [ X ]
    "rings": r"\%\d{2}|\d",     # Percent with two digits afterwards
}
_RE_PATTERNS = {name: re.compile(pattern) for name, pattern in __REGEXES.items()}

__N_MAX_RINGS = 50

__SUPPORTED_ELEMENTS = {
    "C",
    "H",
    "O",
    "N",
    "S",
    "P",
    "F",
    "Cl",
    "Br",
    "I",
    "c",
    "n",
    "o",
    "s",
}

def clean_smiles(
    smiles: str,
    remove_salt=True,
    desalt=False,
    uncharge=True,
    sanitize=True,
    remove_stereochemistry=True,
    to_canonical=True,
):
    """
    Cleaning of a simple SMILES string.
    arguments
    smiles: SMILES string to be cleaned
    remove_salt: return None if the SMILES represents a salt for True
    desalt: remove salts from the molecule
    uncharge: neutralize formal charges
    sanitize: sanitize the molecule to ensure validity
    remove_stereochemistry: remove stereochemistry
    to_canonical: convert to canonical SMILES
    returns
    cleaned SMILES string or None (if invalid)
    """

     # Remove salt if input is identified as a salt
    if remove_salt and is_salt(smiles):
        return None

    if remove_stereochemistry:  # remove stereochemistry
        smiles = eliminate_manual_stereochemistry(smiles)

    mol = Chem.MolFromSmiles(smiles)    #transform to molecular object
    if mol is None: # not valid
        return None
    salt_remover = SaltRemover()
    uncharger = rdMolStandardize.Uncharger()
    if desalt:  #desalt
        mol = salt_remover.StripMol(mol, dontRemoveEverything=True)
    if uncharge:    # uncharge
        mol = uncharger.uncharge(mol)
    if sanitize:    # sanitize
        sanitization_flag = Chem.SanitizeMol(
            mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL, catchErrors=True
        )
        # SANITIZE_NONE is the "no error" flag of rdkit!
        if sanitization_flag != Chem.SanitizeFlags.SANITIZE_NONE:
            return None

    return Chem.MolToSmiles(mol, canonical=to_canonical)    # returns smiles strings

def is_supported_chemical(smiles, verbosity=False):
    """
    Checks if a given SMILES string represents a supported chemical structure.
    arguments
    smiles: SMILES string to check
    verbosit: prints detailed messages about unsupported features
    returns
    true if the SMILES is supported, false otherwise
    """
    def contains_unsupported_element(smiles):
        """
        Determines if the SMILES string contains elements not in the supported set.
        arguments
        smiles: SMILES string
        returns
        true if unsupported elements are presentotherwise false.
        """
        tokens = set(segment_smiles(smiles, segment_sq_brackets=True))  # tokenize SMILES string
        elements = {token for token in tokens if is_element(token)} # checks if supported
        return len(elements.difference(__SUPPORTED_ELEMENTS)) > 0   # returns number of unsupported

    atomic_mass = contains_atomic_mass(smiles)  # check for atomic mass in SMILES
    solo_element = contains_solo_element(smiles)    # check for solo elements
    n_rings = find_n_rings(smiles)  # count the number of rings
    unsupported_element = contains_unsupported_element(smiles)  # check if unsupported element
    if verbosity:
        if atomic_mass:
            print("We don't support SMILES with atomic mass.")
        if solo_element:
            print("We don't support SMILES with non-charged metals.")
        if n_rings > __N_MAX_RINGS:
            print(
                f"We don't support chemical with more than {__N_MAX_RINGS} rings."
            )
        if unsupported_element:
            print(
                f"We don't support chemicals that contain elements not in {__SUPPORTED_ELEMENTS}."
            )

    # Returns True if all checks pass, therefore SMILES is supported
    return (
        not atomic_mass
        and not solo_element
        and not n_rings > __N_MAX_RINGS
        and not unsupported_element
    )

def eliminate_manual_stereochemistry(smiles, replace_dict=None):
    """ 
    Elimination of Stereochemistry.
    Arguments:
    smiles_list: list of SMILES 
    replace_dict: dict for replace of strings
    Return: 
    replaced SMILES-list 
    """

    if not replace_dict:
        replace_dict = {'[C@H]':'C', '[C@@H]':'C', '[C@]':'C', '[C@@]':'C', 
                        '/C' : 'C', 'C/' : 'C', 
                        '\\':"", '/':"",
                        '[P@]': 'P', '[P@@]': 'P',
                        '[N@+]': '[N+]', '[N@@+]': '[N+]', 
                        '[S@@+]': '[S+]', '[S@+]': '[S+]'}
    
    for key in replace_dict.keys():
        if key in smiles:
            smiles = smiles.replace(key, replace_dict[key])

    return smiles

def is_salt(smiles: str, negate_result=False) -> bool:
    """
    Checks if a given SMILES string represents a salt (if it has a dot).
    arguments
    smiles: SMILES string
    negate_result: if True, reverses the result of the salt check
    returns
    true if the SMILES salt, False otherwise (negated if `negate_result` is True)
    """
    is_salt = "." in set(smiles)
    if negate_result:
        return not is_salt
    return is_salt

def contains_solo_element(smiles):
    """
    Determines if the SMILES string contains solo elements.
    arguments
    smiles: SMILES string
    returns
    true if solo elements are found
    """
    # Match against a regex pattern for solo elements
    return len(_RE_PATTERNS["solo_element"].findall(smiles)) > 0

def contains_atomic_mass(smiles):
    """
    Checks if the SMILES string contains atomic masses.
    arguments
    smiles: SMILES string
    returns
    true if atomic masses are found
    """
    return len(_RE_PATTERNS["leading_mass"].findall(smiles)) > 0

def find_n_rings(smiles: str) -> int:
    '''
    Finds all higher-numbered rings structures and returns the amount of rings in the molecule.
    arguments: SMILES string
    return: Number 
    '''
    return len(_RE_PATTERNS["rings"].findall(smiles)) // 2      # Rings start and end with the same number, only count the number once.


def segment_smiles(smiles: str, segment_sq_brackets=True) -> List[str]:
    ''' 
    Tokenized the SMILES string and gives each element and special characters.
    arguments
    SMILES string and segment_sq
    return
    list of tokenized SMILES string
    '''
    regex = _RE_PATTERNS["segmentation_sq"]
    if not segment_sq_brackets:
        regex = _RE_PATTERNS["segmentation"]
    return regex.findall(smiles)
    