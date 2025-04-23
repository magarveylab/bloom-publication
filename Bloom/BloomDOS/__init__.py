import os
from functools import partial
from multiprocessing import Pool
from typing import List, Optional, TypedDict, Union

from tqdm import tqdm

from Bloom.BloomDOS.Molecule import Molecule


class BloomSubmission(TypedDict):
    submission_id: Union[int, str]
    smiles: str


def single_submission(
    submission: BloomSubmission,
    standardize: bool = False,
    output_dir: Optional[str] = None,
):
    os.makedirs(output_dir, exist_ok=True)
    submission_id = submission["submission_id"]
    smiles = submission["smiles"]
    output_fp = f"{output_dir}/{submission_id}.json"
    if os.path.exists(output_fp) == False:
        m = Molecule(smiles)
        if standardize:
            m.standardize()
        m.predict_biosynthesis()
        if output_dir != None:
            m.export_graph(output_fp)


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
