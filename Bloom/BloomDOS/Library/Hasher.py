import xxhash

from Bloom.BloomDOS.DataStructs import EnhancedSmartsDict


def standardize_dict(d: dict):
    return sorted([(k, sorted(v)) for k, v in d.items()])


def standardize_list(l: list):
    return sorted(l)


def get_hash_id(unit: EnhancedSmartsDict):
    # important features of a unit
    smarts = unit["smarts"]
    ignore_patt_idx = standardize_list(unit.get("ignore_patt_idx", []))
    enums = standardize_list(unit.get("enums", []))
    avoid_dereplication = unit.get("avoid_dereplication", False)
    can_dereplicate = unit.get("can_dereplicate", True)
    variable_map = standardize_dict(unit.get("variable_map", {}))
    variable_to_atomic_num_map = standardize_dict(
        unit.get("variable_to_atomic_num_map", {})
    )
    unknown_restriction_map = standardize_list(
        unit.get("unknown_restriction_map", [])
    )
    alpha_restriction_map = standardize_list(
        unit.get("alpha_restriction_map", [])
    )
    priortization = unit.get("priortization", 0)
    uniquify = unit.get("uniquify", True)
    # pool unit into string representation
    features = [
        smarts,
        ignore_patt_idx,
        enums,
        avoid_dereplication,
        can_dereplicate,
        variable_map,
        variable_to_atomic_num_map,
        unknown_restriction_map,
        alpha_restriction_map,
        priortization,
        uniquify,
    ]
    hash_id = xxhash.xxh32(
        " | ".join([str(x) for x in features]), seed=1
    ).hexdigest()
    return hash_id
