from typing import Dict, Literal, Optional

import networkx as nx
import torch
from rdkit import Chem
from rdkit.Chem import PeriodicTable
from torch_geometric.data import Data

periodic_table = Chem.GetPeriodicTable()


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


def get_mol_graph_from(
    smiles: str,
    add_special_tokens: bool = True,
    mol_type: Literal["reactant", "product", "molecule"] = "molecule",
) -> nx.Graph:
    G = nx.Graph()
    mol_nodes = set()
    if add_special_tokens:
        # add CLS nodes
        G.add_node(0, word="[MOL]", extra_x=[0], atom_idx=None, mol_type=None)
    buffer_idx = len(G.nodes)
    # add molecule
    mol = Chem.MolFromSmiles(smiles)
    # flag reactant molecules
    if mol_type == "reactant":
        extra_x = [1]
    else:
        extra_x = [0]
    # add atoms
    for atom in mol.GetAtoms():
        atom_idx = atom.GetIdx()
        node_id = atom_idx + buffer_idx
        map_num = atom.GetAtomMapNum()
        G.add_node(
            node_id,
            atom_idx=atom_idx,
            word=get_atom_word_from(atom=atom),
            extra_x=extra_x,
            mol_type=mol_type,
            map_num=map_num,
        )
        mol_nodes.add(node_id)
    # add bonds
    for bond in mol.GetBonds():
        bond_idx = bond.GetIdx()
        a1 = bond.GetBeginAtomIdx() + buffer_idx
        a2 = bond.GetEndAtomIdx() + buffer_idx
        G.add_edge(
            a1,
            a2,
            word=get_bond_word_from(bond=bond),
            extra_edge_attr=extra_x,
            bond_idx=bond_idx,
        )
    return G


def get_rxn_graph_from(smiles: str) -> nx.Graph:
    reactant, product = smiles.split(">>")
    # special token
    G = nx.Graph()
    G.add_node(0, word="[RXN]", extra_x=[0], atom_idx=None, mol_type=None)
    # prepare graphs for reactant and product
    H = get_mol_graph_from(
        smiles=reactant, add_special_tokens=False, mol_type="reactant"
    )
    I = get_mol_graph_from(
        smiles=product, add_special_tokens=False, mol_type="product"
    )
    # combine graphs
    Y = nx.disjoint_union(G, H)
    Y = nx.disjoint_union(Y, I)
    return Y


def get_graph_tensor_from(
    G: nx.Graph, atom_vocab: Dict[str, int], bond_vocab: Dict[str, int]
) -> Data:
    atom_unk_tok_id = atom_vocab["[UNK]"]
    bond_unk_tok_id = bond_vocab["[UNK]"]
    # node features
    x = torch.LongTensor(
        [
            [atom_vocab.get(G.nodes[n]["word"], atom_unk_tok_id)]
            for n in sorted(G.nodes)
        ]
    )
    extra_x = torch.FloatTensor(
        [G.nodes[n]["extra_x"] for n in sorted(G.nodes)]
    )
    # edge index
    edge_index = torch.LongTensor([[n1, n2] for n1, n2 in G.edges])
    edge_index = edge_index.transpose(-1, -2)
    # edge features
    edge_attr = torch.LongTensor(
        [
            [bond_vocab.get(a["word"], bond_unk_tok_id)]
            for n1, n2, a in G.edges(data=True)
        ]
    )
    extra_edge_attr = torch.FloatTensor(
        [a["extra_edge_attr"] for n1, n2, a in G.edges(data=True)]
    )
    datapoint = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        extra_x=extra_x,
        extra_edge_attr=extra_edge_attr,
    )
    return datapoint


def rxnsmiles2tensor(
    smiles: str, atom_vocab: Dict[str, int], bond_vocab: Dict[str, int]
) -> Data:
    G = get_rxn_graph_from(smiles=smiles)
    return get_graph_tensor_from(
        G=G, atom_vocab=atom_vocab, bond_vocab=bond_vocab
    )
