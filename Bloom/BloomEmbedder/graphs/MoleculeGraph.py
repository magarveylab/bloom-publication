import json
from glob import glob

from rdkit import Chem
from rdkit.Chem import PeriodicTable

from Bloom.BloomEmbedder import curdir
from Bloom.CommonUtils.HeteroGraph import HeteroGraph

periodic_table = Chem.GetPeriodicTable()


def get_vocab(vocab_dir: str):
    vocab_filenames = glob(vocab_dir)
    vocab = {}
    for fp in vocab_filenames:
        key = fp.split("/")[-1].split(".")[0]
        vocab[key] = json.load(open(fp))
    return vocab


def get_node_vocab(
    vocab_dir: str = f"{curdir}/models/molecule/vocab/node_vocab/*",
):
    return get_vocab(vocab_dir=vocab_dir)


def get_edge_vocab(
    vocab_dir: str = f"{curdir}/models/molecule/vocab/edge_vocab/*",
):
    return get_vocab(vocab_dir=vocab_dir)


def get_atom_word_from(atom: Chem.Atom):
    atom_type = PeriodicTable.GetElementSymbol(
        periodic_table, atom.GetAtomicNum()
    )
    hybridization = str(atom.GetHybridization())
    charge = str(atom.GetFormalCharge())
    chirality = str(atom.GetChiralTag())
    is_aromatic = str(atom.GetIsAromatic())
    word = f"{atom_type}-{hybridization}-{charge}-{chirality}-{is_aromatic}"
    return word


def get_bond_word_from(bond: Chem.Bond):
    bond_type = str(bond.GetBondType())
    is_conjugated = str(bond.GetIsConjugated())
    stereo = str(bond.GetStereo())
    word = f"{bond_type}-{is_conjugated}-{stereo}"
    return word


class MoleculeGraph(HeteroGraph):

    def __init__(self, graph_id: str):
        # define schema
        schema = {}
        schema["node_types"] = [
            "Molecule",
            "MetaboliteMolecularRegion",
            "Atom",
        ]
        schema["edge_types"] = [
            ("Atom", "bond", "Atom"),
            ("Atom", "atom_to_unit", "MetaboliteMolecularRegion"),
            (
                "MetaboliteMolecularRegion",
                "mol_region_adj",
                "MetaboliteMolecularRegion",
            ),
        ]

        schema["node_embedding_dim"] = {}
        schema["edge_embedding_dim"] = {}
        self.graph_id = graph_id
        super().__init__(schema=schema)

    @classmethod
    def build_from_bloom_graph(cls, graph_id: str, bloom_graph_fp: str):
        bloom_graph = json.load(open(bloom_graph_fp))
        G = cls(graph_id=graph_id)
        edge_type_lookup = {e[1]: e for e in G.edge_types}
        # add molecule node
        G.add_node(node_type="Molecule", label="[CLS]")
        # add units
        unit_traceback = {}
        unit_node_lookup = {}
        for n in bloom_graph["nodes"]:
            unit_node_id = G.add_node(
                node_type="MetaboliteMolecularRegion",
                label=n.get("top_label", "[UNK]"),
            )
            node_id = n["node_id"]
            unit_node_lookup[node_id] = unit_node_id
            unit_traceback[unit_node_id] = node_id
        # add unit connections
        for e in bloom_graph["edges"]:
            n1 = unit_node_lookup[e["n1"]]
            n2 = unit_node_lookup[e["n2"]]
            G.add_edge(
                n1=n1, n2=n2, edge_type=edge_type_lookup["mol_region_adj"]
            )
        # add atoms
        atom_idx_to_atom_node_id = {}
        smiles = bloom_graph["smiles"]
        mol = Chem.MolFromSmiles(smiles)
        for a in mol.GetAtoms():
            atom_idx = a.GetIdx()
            map_num = a.GetAtomMapNum()
            atom_label = get_atom_word_from(a)
            atom_node_id = G.add_node(node_type="Atom", label=atom_label)
            atom_idx_to_atom_node_id[atom_idx] = atom_node_id
            # add unit to atom connections
            if map_num in unit_node_lookup:
                unit_node_id = unit_node_lookup[map_num]
                G.add_edge(
                    n1=atom_node_id,
                    n2=unit_node_id,
                    edge_type=edge_type_lookup["atom_to_unit"],
                )
        # add bonds
        for b in mol.GetBonds():
            # add atom to atom connections
            a1_node_id = atom_idx_to_atom_node_id[b.GetBeginAtomIdx()]
            a2_node_id = atom_idx_to_atom_node_id[b.GetEndAtomIdx()]
            bond_label = get_bond_word_from(bond=b)
            G.add_edge(
                n1=a1_node_id,
                n2=a2_node_id,
                edge_type=edge_type_lookup["bond"],
                label=bond_label,
            )
        # note unit traceback maps unit to rows of tensors
        return {"graph": G, "unit_traceback": unit_traceback}
