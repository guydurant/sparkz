from dataclasses import dataclass
import numpy as np
from boltz.data import const
from pdbfixer import PDBFixer
from openmm.app import PDBFile
from io import StringIO

@dataclass
class Atom:
    """Stores details of atoms to be used as tokens."""

    atom: str
    atom_id: int
    atom_name: str
    alt_loc: str
    resname: str
    chain_id: str
    res_id: int
    code_for_insertion: str
    x: float
    y: float
    z: float
    occupancy: float
    temp_factor: float
    segment_id: str
    element: str
    charge: str

    def __str__(self):
        return f"{self.atom:<6}{self.atom_id:>5}  {self.atom_name:<3}{self.alt_loc:<1}{self.resname:<3} {self.chain_id:<1}{self.res_id:>4}{self.code_for_insertion:<1}   {self.x:>8.3f}{self.y:>8.3f}{self.z:>8.3f}{self.occupancy:>6.2f}{self.temp_factor:>6.2f}      {self.segment_id:<4}{self.element:>2}{self.charge:>2}"



def get_atom(pdbblock):
    """Return an Atom object from a pdbblock."""
    atom = pdbblock[0:6].strip()
    atom_id = int(pdbblock[6:11].strip())
    atom_name = pdbblock[12:16].strip()
    alt_loc = pdbblock[16:17].strip()
    resname = pdbblock[17:20].strip()
    chain_id = pdbblock[21:22].strip()
    res_id = int(pdbblock[22:26].strip())
    code_for_insertion = pdbblock[26:27].strip()
    x = float(pdbblock[30:38].strip())
    y = float(pdbblock[38:46].strip())
    z = float(pdbblock[46:54].strip())
    occupancy = float(pdbblock[54:60].strip())
    temp_factor = float(pdbblock[60:66].strip())
    segment_id = pdbblock[67:76].strip()
    element = pdbblock[76:78].strip()
    charge = pdbblock[78:80].strip()

    return Atom(
        atom,
        atom_id,
        atom_name,
        alt_loc,
        resname,
        chain_id,
        res_id,
        code_for_insertion,
        x,
        y,
        z,
        occupancy,
        temp_factor,
        segment_id,
        element,
        charge,
    )


class Molecule:
    """Amino acid, ligand or nucleotide molecule."""

    def __init__(self, pdbblock, remove_hs=True):
        self.pdbblock = "\n".join([line for line in pdbblock.split("\n") if line[76:78].strip() != "H"])
        unorderd_atoms = {
            i: get_atom(line)
            for i, line in enumerate(self.pdbblock.split("\n"))
            if line[0:6] == "ATOM  " or line[0:6] == "HETATM"
        }
        self.resname = self.get_resname(unorderd_atoms)
        ordered_atoms = {}
        if self.resname in const.ref_atoms:
            for j, atom in enumerate(const.ref_atoms[self.resname]):
                for i in unorderd_atoms:
                    if unorderd_atoms[i].atom_name == atom:
                        ordered_atoms[j] = unorderd_atoms[i]
                        break
            self.atoms = ordered_atoms
        else:
            self.atoms = unorderd_atoms
        for i in self.atoms:
            self.atoms[i].atom_id = i + 1
        # self.resname = self.get_resname()
        self.resid = self.get_resid()
        self.chain_id = self.get_chain()
        self.coords = self.get_coords()
        if remove_hs:
            self.remove_hs()

        # self.self_consistency_pdbblock_check()
        # assert self.pdbblock == self.get_new_pdbblock(), f"Inconsistent PDB format detected {self.chain_id}{self.resid}{self.resname} {"\n"+self.get_new_pdbblock()} and {"\n"+self.pdbblock}"

    def self_consistency_pdbblock_check(self):
        """Check if the pdbblock is consistent with the atoms."""
        pdbblock_without_hydrogens = "\n".join(
            line for line in self.pdbblock.split("\n")[:-1] if line[76:78] != " H"
        )
        new_pdbblock = self.get_new_pdbblock()
        assert pdbblock_without_hydrogens == new_pdbblock, f"Inconsistent PDB format detected \n{pdbblock_without_hydrogens}\n and \n{new_pdbblock}"

    def remove_hs(self):
        """Remove hydrogen atoms from the molecule."""
        self.atoms = {
            i: atom for i, atom in self.atoms.items() if atom.element != "H"
        }

    def get_chain(self):
        chain_ids = list(set([self.atoms[i].chain_id for i in self.atoms]))
        if len(chain_ids) != 1:
            raise ValueError(f"The token is not in a single chain. Token has {chain_ids}.")
        return chain_ids[0]

    def get_resname(self, atoms):
        """Return the residue name of the token."""
        resnames = list(set([atoms[i].resname for i in atoms]))
        if len(resnames) != 1:
            raise ValueError(f"The token is not a single residue. Token has {resnames}.")
        return resnames[0]
    
    def get_resid(self):
        """Return the residue id of the token."""
        resids = list(set([self.atoms[i].res_id for i in self.atoms]))
        if len(resids) != 1:
            raise ValueError(f"The token is not a single residue. Token has {resids}")
        return resids[0]

    def get_coords(self):
        """Return the coordinates of the token."""
        return np.array(
            [(self.atoms[i].x, self.atoms[i].y, self.atoms[i].z) for i in self.atoms]
        )

    def get_new_pdbblock(self, previous_atom_num=None, chain=None, res_id=None, resname=None):
        """Return a new pdbblock with updated atom numbers."""
        new_pdbblock = []
        for num, i in enumerate(self.atoms):
            atom = self.atoms[i]
            if previous_atom_num is not None:
                atom.atom_id = previous_atom_num + 1
            if chain is not None:
                atom.chain_id = chain
            if res_id is not None:
                atom.res_id = res_id
            if resname is not None:
                atom.resname = resname
            new_pdbblock.append(str(atom))
        return "\n".join(new_pdbblock)

    def remove_atom(self, atom_name):
        """Remove an atom from the molecule."""
        self.atoms = {
            i: atom for i, atom in self.atoms.items() if atom.atom_name != atom_name
        }
        self.coords = self.get_coords()
        
    # def repair_missing_atoms(self, res_lines, missing_atoms):
    #     single_residue = PDBFixer(pdbfile=StringIO("".join(res_lines)))
    #     single_residue.findMissingResidues()
    #     single_residue.findMissingAtoms()
    #     single_residue.addMissingAtoms()
    #     fake_file = StringIO()
    #     PDBFile.writeFile(single_residue.topology, single_residue.positions, fake_file)
    #     return [
    #         i
    #         for i in fake_file.getvalue().split("\n")
    #         if i[12:16].strip() in missing_atoms
    #     ]


if __name__ == "__main__":
    test_residue = """ATOM    307  CA  ILE A  23      34.336 -14.207  70.107  1.00 12.14      A    C  
ATOM    308  C   ILE A  23      34.310 -15.191  71.277  1.00 12.26      A    C  
ATOM    309  O   ILE A  23      33.372 -15.977  71.417  1.00 13.02      A    O  
ATOM    310  CB  ILE A  23      33.265 -13.113  70.312  1.00 12.56      A    C  
ATOM    311  CG1 ILE A  23      33.177 -12.231  69.071  1.00 12.86      A    C  
ATOM    312  CG2 ILE A  23      33.544 -12.302  71.565  1.00 13.49      A    C  
ATOM    313  CD1 ILE A  23      32.058 -11.249  69.151  1.00 14.46      A    C  """

    token = Molecule(test_residue.split("\n"))
    print(token.resname)
    print(token.chain_id)
    print(token.coords)
    new_pdbblock = token.get_new_pdbblock()
    print(new_pdbblock)
    print(test_residue)
    assert new_pdbblock == test_residue
