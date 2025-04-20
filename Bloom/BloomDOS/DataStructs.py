from typing import List, Dict, Set, Optional, TypedDict

####################################################################
# SMARTS Data Structures
####################################################################

class EnhancedSmartsDict(TypedDict, total=False):
    # check for default values in EnhancedSmarts.py
    notation: str
    description: str
    smarts: str
    ignore_patt_idx: Optional[List[int]] # default []
    # assign flags to units
    enums: Optional[List[str]] # default []
    smiles: Optional[str] # default None
    # annotation cannot be removed by other annotations
    avoid_dereplication: Optional[bool] # default False
    # annotation can remove other annotations
    can_dereplicate: Optional[bool] # default True
    # this is used to assign enums to variable groups
    variable_map: Optional[Dict[int, List[str]]] # default {}
    # this is used to assign atomic numbers to variable groups
    variable_to_atomic_num_map: Optional[Dict[int, List[int]]] # default {}
    # certain atoms part of the smarts query that is not annotated and cannot be designated as unknown
    unknown_restriction_map: Optional[List[int]] # default []
    # certain atoms designated as alpha carbons in polyketide systems
    alpha_restriction_map: Optional[List[int]] # default []
    # determines which units to remove if overlap
    priortization: Optional[int] # default 0
    # determines whether or not the matches are uniquified
    # useful for queries with multiple variable groups and variable restrictions
    uniquify: Optional[bool] # default True

class EnhancedSmartsResponse(TypedDict):
    description: str
    notation: str
    patt: Set[int]
    patt_len: int
    enums: Set[str]
    avoid_dereplication: bool
    alpha_hit: Set[int]
    alpha_restrictions: Set[int]
    variable_restrictions: Dict[int, Set[int]]
    unknown_restrictions: Set[int]
    priortization: int