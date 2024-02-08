import sys
import numpy as np
from openmm.app.pdbfile import _formatIndex, PDBFile
from rdkit import Chem
from rdkit.Chem import AllChem, RemoveHs


class temporary_seed:
    def __init__(self, seed):
        self.seed = seed
        self.backup = None

    def __enter__(self):
        self.backup = np.random.randint(2 ** 32 - 1, dtype=np.uint32)
        np.random.seed(self.seed)

    def __exit__(self, *_):
        np.random.seed(self.backup)


def generate_conformer(mol):
    ps = AllChem.ETKDGv2()
    id = AllChem.EmbedMolecule(mol, ps)
    if id == -1:
        print('rdkit coords could not be generated without using random coords. using random coords now.')
        ps.useRandomCoords = True
        AllChem.EmbedMolecule(mol, ps)
    # else:
    #    AllChem.MMFFOptimizeMolecule(mol_rdkit, confId=0)


def read_strings_from_txt(path):
    # every line will be one element of the returned list
    with open(path) as file:
        lines = file.readlines()
        return [line.rstrip() for line in lines]


def remove_all_hs(mol):
    params = Chem.RemoveHsParameters()
    params.removeAndTrackIsotopes = True
    params.removeDefiningBondStereo = True
    params.removeDegreeZero = True
    params.removeDummyNeighbors = True
    params.removeHigherDegrees = True
    params.removeHydrides = True
    params.removeInSGroups = True
    params.removeIsotopes = True
    params.removeMapped = True
    params.removeNonimplicit = True
    params.removeOnlyHNeighbors = True
    params.removeWithQuery = True
    params.removeWithWedgedBond = True
    return RemoveHs(mol, params)


def pdbfile_writeFooter(topology, file=sys.stdout, allBonds=False):
    """Write out the footer for a PDB file. This is a function from openmm.app.pdbfile but changed to save all atoms

    Parameters
    ----------
    topology : Topology
        The Topology defining the molecular system being written
    file : file=stdout
        A file to write the file to
    """
    # Identify bonds that should be listed as CONECT records.

    conectBonds = []
    for atom1, atom2 in topology.bonds():
        if allBonds:
            conectBonds.append((atom1, atom2))
        elif atom1.residue.name not in PDBFile._standardResidues or atom2.residue.name not in PDBFile._standardResidues:
            conectBonds.append((atom1, atom2))
        elif atom1.name == 'SG' and atom2.name == 'SG' and atom1.residue.name == 'CYS' and atom2.residue.name == 'CYS':
            conectBonds.append((atom1, atom2))
    if len(conectBonds) > 0:

        # Work out the index used in the PDB file for each atom.

        atomIndex = {}
        nextAtomIndex = 0
        prevChain = None
        for chain in topology.chains():
            for atom in chain.atoms():
                if atom.residue.chain != prevChain:
                    nextAtomIndex += 1
                    prevChain = atom.residue.chain
                atomIndex[atom] = nextAtomIndex
                nextAtomIndex += 1

        # Record which other atoms each atom is bonded to.

        atomBonds = {}
        for atom1, atom2 in conectBonds:
            index1 = atomIndex[atom1]
            index2 = atomIndex[atom2]
            if index1 not in atomBonds:
                atomBonds[index1] = []
            if index2 not in atomBonds:
                atomBonds[index2] = []
            atomBonds[index1].append(index2)
            atomBonds[index2].append(index1)

        # Write the CONECT records.

        for index1 in sorted(atomBonds):
            bonded = atomBonds[index1]
            while len(bonded) > 4:
                print("CONECT%5s%5s%5s%5s" % (
                    _formatIndex(index1, 5), _formatIndex(bonded[0], 5), _formatIndex(bonded[1], 5),
                    _formatIndex(bonded[2], 5)), file=file)
                del bonded[:4]
            line = "CONECT%5s" % _formatIndex(index1, 5)
            for index2 in bonded:
                line = "%s%5s" % (line, _formatIndex(index2, 5))
            print(line, file=file)
    print("END", file=file)
