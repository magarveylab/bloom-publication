from typing import Dict, Tuple

from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

from Bloom.BloomGraph.DataStructs import ImageDataDict


def highlight_mol(
    smiles,
    atoms_to_colour: Dict[int, tuple] = {},
    node_to_atoms: Dict[int, set] = {},
    default_colour: Tuple[float, float, float] = (0.827, 0.827, 0.827),
) -> ImageDataDict:
    # note smiles must be mapped to nodes
    node_dict = {}
    mol = Chem.MolFromSmiles(smiles)

    all_atoms = set(a.GetIdx() for a in mol.GetAtoms())
    coloured_atoms = set(atoms_to_colour.keys())
    missing_atoms = all_atoms - coloured_atoms
    atoms_to_colour.update({a: default_colour for a in missing_atoms})

    # Get colours for bonds
    bonds_to_colour = {}
    for b in mol.GetBonds():
        a1 = b.GetBeginAtomIdx()
        a2 = b.GetEndAtomIdx()
        if a1 not in atoms_to_colour:
            continue
        for n in node_to_atoms:
            if a1 in node_to_atoms[n] and a2 in node_to_atoms[n]:
                bonds_to_colour[b.GetIdx()] = atoms_to_colour[a1]

    # Create clean mol with no map numbers
    clean_mol = Chem.Mol(mol)
    [x.SetAtomMapNum(0) for x in clean_mol.GetAtoms()]

    # Label first atom of each node
    for node, atoms in node_to_atoms.items():
        if len(atoms) == 0:
            continue
        clean_mol.GetAtomWithIdx(min(atoms)).SetAtomMapNum(node)
    # create a SVG drawer for our molecule
    draw_object = rdMolDraw2D.MolDraw2DSVG(700, 700)
    # set the bond pallet to black so we can see highlighting
    draw_object.drawOptions().useBWAtomPalette()
    # draw the molecule with highlighting
    rdMolDraw2D.PrepareAndDrawMolecule(
        draw_object,
        clean_mol,
        highlightAtoms=list(atoms_to_colour.keys()),
        highlightAtomColors=atoms_to_colour,
        highlightBonds=list(bonds_to_colour.keys()),
        highlightBondColors=bonds_to_colour,
    )

    draw_object.FinishDrawing()
    # create svg
    svg = draw_object.GetDrawingText().replace("svg:", "")
    return {"svg": svg}
