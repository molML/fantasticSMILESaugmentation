from .bioisosteric_substitution import Bioisosters
from .masking_functional_group import GroupMasking
from .masking_random import RandomMasking
from .deletion import DeletionRandom
from .deletion import DeletionValid
from .deletion import DeletionProtected
from .baselines import Selftraining
from .baselines import Enumeration

__NAME_TO_METHOD = {
    "bioisosters-based": Bioisosters,
    "group-masking": GroupMasking,
    "random-masking": RandomMasking,
    "random-deletion-invalid": DeletionRandom,
    "random-deletion-valid": DeletionValid,
    "protected-deletion": DeletionProtected,
    "self-training": Selftraining,   
    "enumeration": Enumeration,       

}

def get_method(name: str, smiles_list: list, aug_fold: int, prob: float):
    if name not in __NAME_TO_METHOD.keys():
        raise ValueError(
            f"'{name}' is an unknown method. The known names are: {list(__NAME_TO_METHOD.keys())}"
        )
    return __NAME_TO_METHOD.get(name)(smiles_list, aug_fold, prob)


def get_vectorizer_names():
    return list(__NAME_TO_METHOD.keys())