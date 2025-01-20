import math
from helpers import load_json, get_package_path


def get_periodic_table_properties(fillna=None):
    package_path = get_package_path()
    properties = load_json(
        f"{package_path}/utils/data/chemistry/periodic_table_properties.json"
    )
    if fillna:
        properties = {
            element: {
                feature: fillna if math.isnan(float(value)) else value
                for feature, value in element_props.items()
            }
            for element, element_props in properties.items()
        }
    return properties


_ELEMENTS = get_periodic_table_properties().keys()


def is_aromatic(token):
    '''
    An element is considered aromatic, if lower case is not in _ELEMENTS, but the uppercase character is.
    arguments: token
    returns: 1: is aromatic, 0 is not aromatic
    '''
    return token not in _ELEMENTS and token.title() in _ELEMENTS


def is_element(token):
    return token in _ELEMENTS or is_aromatic(token)

