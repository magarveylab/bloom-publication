from BearLite.GeneralUtils import curdir
from BearLite.Library.Hasher import get_hash_id
import pandas as pd
import ast
import os
import xxhash

# define dir where substrates/reactions can be found
pk_dir = os.path.join(curdir, 'Library/Polyketide/Tables/')

class AlphaSearch:
    # these are motifs to label potential alpha carbons in polyketide systems
    def __init__(self):
        self.library = []
        self.create_library()
        self.add_hash_ids()

    def create_library(self):
        alpha_flags_fp = '{}/alpha_flags.csv'.format(pk_dir)
        for rec in pd.read_csv(alpha_flags_fp).to_dict('records'):
            # prepare ignore_patt_idx
            ignore_patt_idx = ast.literal_eval(rec['ignore_patt_idx'])
            # prepare enums
            enums = [rec['enum']]
            self.library.append({'notation': rec['notation'],
                                 'description': rec['description'],
                                 'smiles': rec['smiles'],
                                 'smarts': rec['smarts'],
                                 'enums': enums,
                                 'ignore_patt_idx': ignore_patt_idx})
            
    def add_hash_ids(self):
        for unit in self.library:
            unit['hash_id'] = get_hash_id(unit)
