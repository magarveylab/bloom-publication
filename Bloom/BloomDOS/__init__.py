import os
from functools import partial
from multiprocessing import Pool
from typing import List, Optional, TypedDict, Union

import timeout_decorator
from tqdm import tqdm

from Bloom.BloomDOS.Molecule import Molecule


class BloomSubmission(TypedDict):
    metabolite_id: Union[int, str]
    smiles: str


@timeout_decorator.timeout(300)
def single_submission(
    submission: BloomSubmission,
    standardize: bool = False,
    output_dir: Optional[str] = None,
):
    os.makedirs(output_dir, exist_ok=True)
    metabolite_id = submission["metabolite_id"]
    smiles = submission["smiles"]
    output_fp = f"{output_dir}/{metabolite_id}.json"
    if os.path.exists(output_fp) == False:
        try:
            print(f"Processing {metabolite_id}...")
            m = Molecule(smiles)
            if standardize:
                m.standardize()
            m.predict_biosynthesis()
            if output_dir != None:
                m.export_graph(output_fp)
        except Exception as e:
            print(f"Error processing {metabolite_id}: {e}")
    else:
        print(f"Skipping {metabolite_id}, already processed.")


def multiprocess_subission(
    submission_list: List[BloomSubmission],
    output_dir: str,
    cpus: int = 3,
    standardize: bool = False,
):
    os.makedirs(output_dir, exist_ok=True)
    funct = partial(
        single_submission,
        standardize=standardize,
        output_dir=output_dir,
    )
    pool = Pool(cpus)
    process = pool.imap_unordered(funct, submission_list)
    [p for p in tqdm(process, total=len(submission_list))]
