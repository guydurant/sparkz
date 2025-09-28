from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign, rdDetermineBonds
from processing.molecule import Molecule
from processing.constants import CRYSTALLISATION_AIDS, MODIFIED_RESIDUES, DNA_EXCLUSION
from boltz.data import const
from pdbfixer import PDBFixer
from openmm.app import PDBFile
from io import StringIO
from scipy.spatial.distance import cdist
import numpy as np
import logging
import pickle
import io
from rdkit.Chem import rdFMCS


logger = logging.getLogger(__name__)


class Complex:
    """
    A class to handle protein-ligand complexes from PDB files for guidance and hallucination.
    """
    def __init__(self, pdb_path, ligand, ligand_code, replace_modified_res=True, cache_dir="/homes/user"):
        """Initialize the Complex class.

        Args:
            pdb_path (str): Path to the PDB file.
            ligand (str): Path to the ligand file.
            ligand_code (str): CCD code for the ligand.
            replace_modified_res (bool, optional): Whether to replace modified residues with canonical amino acid. Defaults to True.
            cache_dir (str, optional): Directory for caching. Defaults to "/homes/user".
        """
        
        self._load_complex(
            pdb_path,
            ligand_path=ligand,
            ligand_code=ligand_code,
            replace_modified_res=replace_modified_res,
        )
        self.ccd_pkl = f"{cache_dir}/ccd.pkl"
        
    
    def _load_complex(self, pdb_path, ligand_path, ligand_code, replace_modified_res=True):
        """Calls various functions to load and process the complex.

        Args:
            pdb_path (str): Path to the PDB file.
            ligand_path (str): Path to the ligand file.
            ligand_code (str): CCD code for the ligand.
            replace_modified_res (bool, optional): Whether to replace modified residues with canonical amino acid. Defaults to True.
        """
        self.protein_molecules = {}
        self.heteratoms_molecules = {}
        self.replace_modified_res = replace_modified_res
        self.read_pdbblock(pdb_path)
        self.ligand = self.read_ligand(ligand_path)
        self.filter_heteratoms()
        self.clean_protein()
        self.ligand_code = ligand_code
        self.sequence = self.get_sequence()
        

    def read_pdbblock(self, pdb_path):
        """
        Reads the PDB block from the specified PDB file. 
        Stores generated Molecule objects in protein_molecules and heteratoms_molecules dictionaries.

        Args:
            pdb_path (str): Path to the PDB file.
        """
        with open(pdb_path, "r") as f:
            pdbblock = f.readlines()
        current_pdbblock = ""
        current_residue = None
        for line in pdbblock:
            if line[0:6] == "ATOM  ":
                mol_type = "protein"
            elif line[0:6] == "HETATM":
                mol_type = "heteroatoms"
            else:
                continue
            unique_key = (
                mol_type,  # molecule type
                line[17:20].strip(),  # resname
                line[22:26].strip(),  # res_id
                line[21].strip(),  # chain_id
            )
            if unique_key != current_residue and current_residue is not None:
                if current_residue[0] == "protein":
                    try:
                        self.protein_molecules[current_residue] = Molecule(
                            current_pdbblock
                        )
                    except ValueError:
                        logger.debug(
                            f"Skipping {current_residue} as has no parsable atoms."
                        )
                elif current_residue[0] == "heteroatoms":
                    try:
                        if current_residue[1] in MODIFIED_RESIDUES:
                            self.protein_molecules[current_residue] = Molecule(
                                current_pdbblock
                            )
                        else:
                            self.heteratoms_molecules[current_residue] = Molecule(
                                current_pdbblock
                            )
                    except ValueError:
                        logger.debug(
                            f"Skipping {current_residue} as has no parsable atoms."
                        )
                else:
                    raise ValueError("Unknown residue type.")
                current_pdbblock = line
                current_residue = unique_key
            elif current_residue is None:
                current_residue = unique_key
            elif unique_key == current_residue:
                current_pdbblock += line
            else:
                raise ValueError("Unknown residue type.")
    
    @classmethod
    def from_ligandmpnn_scpack(cls, pdb_path, ligand_file, ligand_code, replace_modified_res=True, cache_dir=".cache"):
        """
        Create a Complex instance from prediction from LigandMPNN-SCPacker.

        Args:
            pdb_path (str): Path to the PDB file.
            ligand_file (str): Path to the ligand file.
            ligand_code (str): CCD code for the ligand.
            replace_modified_res (bool, optional): Whether to replace modified residues with canonical amino acid. Defaults to True.
            cache_dir (str, optional): Directory for caching. Defaults to ".cache".

        Returns:
            Complex: A Complex instance.
        """
        from processing.ligandmpnn_scpack import restype_3to1, get_ligandmpnn_sc_prior
        from pymol import cmd
        from ligandmpnn.data_utils import (
            restype_str_to_int,
        )
        obj = cls(pdb_path, ligand_file, ligand_code, replace_modified_res=replace_modified_res)
        redesign_pocket, outer_pocket = obj.get_protein_pocket(inner_threshold=5, outer_threshold=8)
        total_pdbblock = ""
        with open(pdb_path, "r") as f:
            pdb_lines = f.readlines()
        ligand_pdb = Chem.MolToPDBBlock(Chem.MolFromMolFile(ligand_file))
        total_pdbblock = "".join(pdb_lines) + ligand_pdb
        with open("total_pdbblock.pdb", "w") as f:
            f.write(total_pdbblock)
        fixed_residues = {i: restype_str_to_int[restype_3to1[res[1]]] for i, res in enumerate(obj.protein_molecules.keys()) if res not in redesign_pocket}
        get_ligandmpnn_sc_prior(
            "total_pdbblock.pdb",   
            fixed_residues, 
            cache=".cache", outfile="ligand_scpacked.pdb"
        )
        with open("ligand_scpacked.pdb", "r") as f:
            protein_scpacked_file = f.read()
        # remove any UNK residues
        protein_scpacked_file = "\n".join(
            [line for line in protein_scpacked_file.split("\n") if "UNL" not in line]
        )
        with open("ligand_scpacked_no_lig.pdb", "w") as f:
            f.write(protein_scpacked_file)
        cmd.reinitialize()
        cmd.load("ligand_scpacked_no_lig.pdb", "ligandmpnn_attempt")
        cmd.load(pdb_path, "ground_truth")
        cmd.align("ligandmpnn_attempt", "ground_truth")
        cmd.save("ligand_scpacked_no_lig_aligned.pdb", "ligandmpnn_attempt")
        cmd.delete("all")
        new_cls = cls(
            "ligand_scpacked_no_lig_aligned.pdb",
            ligand_file,
            ligand_code,
            replace_modified_res=replace_modified_res,
        )
        new_cls.redesign_pocket = redesign_pocket
        new_cls.outer_pocket = outer_pocket  
        return new_cls


    def filter_heteratoms(self):
        """Filter out heteroatoms that are not crystallization aids.
        
        Modifies the heteratoms_molecules dictionary in place.
        """
        for key in self.heteratoms_molecules.copy():
            if key[1] in CRYSTALLISATION_AIDS:
                del self.heteratoms_molecules[key]
            # remove if no atoms in molecule
            elif self.heteratoms_molecules[key].atoms == {}:
                del self.heteratoms_molecules[key]

    def check_residue_for_missing_atoms(self, res, molecule):
        """Check a residue for missing atoms.
        
        Args:
            res (tuple): A tuple representing the residue (mol_type, resname, res_id, chain_id).
            molecule (Molecule): The Molecule object representing the residue.
            
        Returns:
            list or str or None: A list of missing atom names, 'DNA' if the residue is DNA, 'heteroatom' if treated as heteroatom, or None if no atoms are missing.
        """
        if res[1] in MODIFIED_RESIDUES:
            if not self.replace_modified_res:
                logger.debug("Keeping modified residue %s", res, '. Note it is buggy')
                return None
            else:
                logger.debug("Replacing modified residue %s with %s", res, MODIFIED_RESIDUES[res[1]])
                atom_types = [i.atom_name for i in molecule.atoms.values()]
                missing_atoms = [a for a in const.ref_atoms[MODIFIED_RESIDUES[res[1]]] if a not in atom_types]
                for atom in missing_atoms:
                    logger.debug("Missing atom %s in modified residue %s", atom, res)
                return missing_atoms if missing_atoms else None
        if res[1] in DNA_EXCLUSION:
            logger.debug("Skipping DNA residue %s", res, 'as DNA not supported')
            del self.protein_molecules[res]
            return 'DNA'
        if res[1] not in const.ref_atoms and res not in MODIFIED_RESIDUES:
            logger.debug("Treating residue %s as a heteroatom", res)
            self.heteratoms_molecules[res] = molecule
            del self.protein_molecules[res]
            return 'heteroatom'
        atom_types = [i.atom_name for i in molecule.atoms.values()]
        missing_atoms = [a for a in const.ref_atoms[res[1]] if a not in atom_types]
        for atom in missing_atoms:
            logger.debug("Missing atom %s in residue %s", atom, res)
        return missing_atoms if missing_atoms else None


    def clean_additional_atoms_in_residue(self, res, molecule):
        """Remove any additional atoms in a residue that are not part of the standard set.

        Args:
            res (tuple): A tuple representing the residue (mol_type, resname, res_id, chain_id).
            molecule (Molecule): The Molecule object representing the residue.
        """
        key = res[1]
        if res[1] in MODIFIED_RESIDUES:
            if not self.replace_modified_res:
                return
            else:
                atom_types = [i.atom_name for i in molecule.atoms.values()]
                for atom in atom_types:
                    if atom not in const.ref_atoms[MODIFIED_RESIDUES[res[1]]]:
                        logger.debug("Additional atom %s in modified residue %s", atom, res)
                        molecule.remove_atom(atom)
                key = MODIFIED_RESIDUES[res[1]]
        
        atom_types = [i.atom_name for i in molecule.atoms.values()]
        for atom in atom_types:
            if atom not in const.ref_atoms[key]:
                logger.debug("Additional atom %s in residue %s", atom, res)
                molecule.remove_atom(atom)


    def check_residues_for_missing_and_additional_atoms(self):
        """Check all residues for missing and additional atoms.

        Returns:
            dict: A dictionary mapping residue keys to lists of problematic atoms.
        """
        problematic_res = {}
        for res in list(self.protein_molecules.keys()):
            molecule = self.protein_molecules[res]
            result = self.check_residue_for_missing_atoms(res, molecule)
            if result == 'heteroatom':
                continue
            elif result == 'DNA':
                continue
            elif result:
                problematic_res[res] = result

        for res, molecule in self.protein_molecules.items():
            self.clean_additional_atoms_in_residue(res, molecule)

        return problematic_res


    def repair_missing_atoms(self, res_lines, missing_atoms):
        """Repair missing atoms in a residue using PDBFixer.
        Args:
            res_lines (list): List of lines representing the residue in PDB format.
            missing_atoms (list): List of missing atom names.
        Returns:
            list: List of lines representing the repaired atoms in PDB format.
        """
        single_residue = PDBFixer(pdbfile=StringIO("".join(res_lines)))
        single_residue.findMissingResidues()
        single_residue.findMissingAtoms()
        single_residue.addMissingAtoms()
        fake_file = StringIO()
        PDBFile.writeFile(single_residue.topology, single_residue.positions, fake_file)
        return [
            i
            for i in fake_file.getvalue().split("\n")
            if i[12:16].strip() in missing_atoms
        ]

    def clean_protein(self):
        """ Clean the protein by checking and repairing missing atoms.
        
        Modifies the protein_molecules dictionary in place.
        
        """
        self.problematic_res = self.check_residues_for_missing_and_additional_atoms()
        for res in self.problematic_res:
            missing_atoms = self.problematic_res[res]
            if len(missing_atoms) > 1:
                res_lines = self.protein_molecules[res].get_new_pdbblock().split("\n")
                final_line = res_lines[-1]
                new_lines = [
                    final_line[:12]
                    + f"{atom:>4}"
                    + final_line[16:30]
                    + f"{999.999:8.3f}" * 3
                    + final_line[54:77]
                    + "XX".rjust(2)
                    + "  "
                    for atom in missing_atoms
                ]
                combined_lines = res_lines + new_lines
                ordered_lines = []
                if res[1] in MODIFIED_RESIDUES and self.replace_modified_res:
                    key = MODIFIED_RESIDUES[res[1]]
                else:
                    key = res[1]
                for atom in const.ref_atoms[key]:
                    for line in combined_lines:
                        if line[12:16].strip() == atom:
                            ordered_lines.append(line)
                new_pdbblock = "\n".join(ordered_lines)
                self.protein_molecules[res] = Molecule(new_pdbblock)
            else:
                res_lines = self.protein_molecules[res].get_new_pdbblock().split("\n")
                repaired_lines = self.repair_missing_atoms(res_lines, missing_atoms)
                final_line = res_lines[-1]
                new_lines = [
                    final_line[:12]
                    + line[12:16]
                    + final_line[16:30]
                    + line[30:54]
                    + final_line[54:77]
                    + line[77:]
                    for line in repaired_lines
                ]
                combined_lines = res_lines + new_lines
                ordered_lines = []
                if res[1] in MODIFIED_RESIDUES and self.replace_modified_res:
                    key = MODIFIED_RESIDUES[res[1]]
                else:
                    key = res[1]
                for atom in const.ref_atoms[key]:
                    for line in combined_lines:
                        if line[12:16].strip() == atom:
                            ordered_lines.append(line)
                new_pdbblock = "\n".join(ordered_lines)
                try:
                    self.protein_molecules[res] = Molecule(new_pdbblock)
                except ValueError:
                    logger.debug(f"Skipping {res} due as has no parsable atoms.")
                    del self.protein_molecules[res]

    def three_letter_to_one_letter(self, res):
        """Convert a three-letter residue code to a one-letter code.
        Args:
            res (str): Three-letter residue code.
        Returns:
            str: One-letter residue code.
        """
        if res in MODIFIED_RESIDUES:
            return const.prot_token_to_letter[MODIFIED_RESIDUES[res]]
        return const.prot_token_to_letter[res]

    def get_sequence(self):
        """Get the amino acid sequence of the protein using Protein objects.

        Returns:
            dict: A dictionary mapping chain IDs to their amino acid sequences.
        """
        sequence = {}
        for key in self.protein_molecules:
            if key[3] not in sequence:
                sequence[key[3]] = ""
            sequence[key[3]] += self.three_letter_to_one_letter(key[1])
        return sequence

    def read_ligand(self, ligand_path):
        """Read a ligand file and return the corresponding RDKit molecule.

        Args:
            ligand_path (str): Path to the ligand file.

        Raises:
            ValueError: If the ligand file format is not supported.

        Returns:
            Chem.Mol: The RDKit molecule object.
        """
        rdkit_funcs = {
            "sdf": Chem.MolFromMolFile,
            "mol": Chem.MolFromMolFile,
            "pdb": Chem.MolFromPDBFile,
            "mol2": Chem.MolFromMol2File,
        }
        ext = ligand_path.split(".")[-1]
        if ext not in rdkit_funcs:
            raise ValueError(f"Unsupported ligand format {ext}.")
        return rdkit_funcs[ext](ligand_path)
    
    def alanine_mutate_inner_pocket(self, inner_threshold=5, outer_threshold=8):
        """Mutate residues in the inner pocket (to be designed) to alanine.
        
        Args:
            inner_threshold (float, optional): Distance threshold for inner pocket. Defaults to 5.
            outer_threshold (float, optional): Distance threshold for outer pocket. Defaults to 8.
        """
        if not hasattr(self, "redesign_pocket") or not hasattr(self, "outer_pocket"):
            redesign_pocket, outer_pocket = self.get_protein_pocket(
                inner_threshold, outer_threshold
            )
        else:
            # if alanine_mutate_inner_pocket is called, use the below
            redesign_pocket = self.redesign_pocket
            outer_pocket = self.outer_pocket
        self.redesign_pocket = redesign_pocket
        self.outer_pocket = outer_pocket

        for key in redesign_pocket:
            if key not in self.protein_molecules:
                logger.debug(f"Residue {key} not found in protein molecules.")
                continue

            res = self.protein_molecules[key]
            if res.resname != "ALA":
                res.resname = "ALA"
                res.pdbblock = res.get_new_pdbblock(resname="ALA")
                missing_atoms = self.check_residue_for_missing_atoms(key, res)
                self.clean_additional_atoms_in_residue(key, res)

                if missing_atoms:
                    res_lines = res.get_new_pdbblock(resname="ALA").split("\n")
                    final_line = res_lines[-1]
                    new_lines = [
                        final_line[:12]
                        + f"{atom:>4}"
                        + final_line[16:30]
                        + f"{999.999:8.3f}" * 3
                    + final_line[54:77]
                    + "XX".rjust(2)
                    + "  "
                    for atom in missing_atoms
                ]
                else:
                    res_lines = res.get_new_pdbblock(resname="ALA").split("\n")
                    new_lines = []
                combined_lines = res_lines + new_lines
                ordered_lines = []
                for atom in const.ref_atoms["ALA"]:
                    for line in combined_lines:
                        if line[12:16].strip() == atom:
                            ordered_lines.append(line)
                new_pdbblock = "\n".join(ordered_lines)
                self.protein_molecules[key] = Molecule(new_pdbblock)
            else:
                new_pdbblock = res.get_new_pdbblock(resname="ALA")
                self.protein_molecules[key] = Molecule(new_pdbblock)
    
    def alter_sequence(self, new_sequence, inner_threshold=5, outer_threshold=8, sequence_buffer=10, fix_hetatms=True,  guide_ligand=True, guide_sc=False):
        """Alter the sequence of the protein in the redesign pocket to a new sequence.

        Args:
            new_sequence (list): The new amino acid sequence to introduce.
            inner_threshold (float, optional): Distance threshold for inner pocket. Defaults to 5.
            outer_threshold (float, optional): Distance threshold for outer pocket. Defaults to 8.
            sequence_buffer (int, optional): Buffer size for expanding the residue list. Defaults to 10.
            fix_hetatms (bool, optional): Whether to fix heteroatoms. Defaults to True.
            guide_ligand (bool, optional): Whether to guide the ligand. Defaults to True.
            guide_sc (bool, optional): Whether to guide side chains. Defaults to False.

        """
        if not hasattr(self, "redesign_pocket") or not hasattr(self, "outer_pocket"):
            redesign_pocket, outer_pocket = self.get_protein_pocket(
                inner_threshold, outer_threshold
            )
        else:
            # if alanine_mutate_inner_pocket is called, use the below
            redesign_pocket = self.redesign_pocket
            outer_pocket = self.outer_pocket
        expanded_residues = self.expand_residue_list(outer_pocket, sequence_buffer)
        expanded_residues = sorted(list(set(expanded_residues)), key=lambda x: (x[3], int(x[2])))
        # sort and set expanded_residues
        for key, new_res in zip(expanded_residues, new_sequence):
            res = self.protein_molecules[key]
            three_letter_new_res = const.prot_letter_to_token[new_res]
            if three_letter_new_res == key[1]:
                continue
            print(f"Changing residue {key} from {key[1]} to {three_letter_new_res}")
            if res.resname != three_letter_new_res:
                old_pdbblock = res.get_new_pdbblock()
                res.resname = three_letter_new_res
                res.pdbblock = res.get_new_pdbblock(resname=three_letter_new_res)
                missing_atoms = self.check_residue_for_missing_atoms((key[0], three_letter_new_res, key[2], key[3]), res)
                self.clean_additional_atoms_in_residue((key[0], three_letter_new_res, key[2], key[3]), res)
                if missing_atoms:
                    res_lines = self.protein_molecules[key].get_new_pdbblock(resname=three_letter_new_res).split("\n")
                    final_line = res_lines[-1]
                    new_lines = [
                        final_line[:12]
                        + f"{atom:>4}"
                        + final_line[16:30]
                        + f"{999.999:8.3f}" * 3
                        + final_line[54:77]
                        + "XX".rjust(2)
                        + "  "
                        for atom in missing_atoms
                    ]
                    combined_lines = res_lines + new_lines
                    ordered_lines = []
                    for atom in const.ref_atoms[three_letter_new_res]:
                        for line in combined_lines:
                            if line[12:16].strip() == atom:
                                ordered_lines.append(line)
                    new_pdbblock = "\n".join(ordered_lines)
                    self.protein_molecules[key] = Molecule(new_pdbblock)
                else:
                    new_pdbblock = res.get_new_pdbblock(resname=three_letter_new_res)
                    self.protein_molecules[key] = Molecule(new_pdbblock)
        return self.process_for_guidance(
            inner_threshold=inner_threshold,
            outer_threshold=outer_threshold,
            sequence_buffer=sequence_buffer,
            fix_hetatms=fix_hetatms,
            guide_ligand=guide_ligand,
            guide_sc=guide_sc,
        )  

    def process_for_guidance(
        self, inner_threshold=5, outer_threshold=8, sequence_buffer=10, fix_hetatms=True, guide_ligand=True, guide_sc=False
    ):
        """Process the protein structure for guidance for ScrewzFix.

        Args:
            inner_threshold (int, optional): Threshold for inner pocket. Defaults to 5.
            outer_threshold (int, optional): Threshold for outer pocket. Defaults to 8.
            sequence_buffer (int, optional): Buffer size for sequence alignment. Defaults to 10.
            fix_hetatms (bool, optional): Whether to fix heteroatoms. Defaults to True.
            guide_ligand (bool, optional): Whether to guide ligand. Defaults to True.
            guide_sc (bool, optional): Whether to guide side chains. Defaults to False.

        Returns:
            dict: A dictionary containing the processed protein structure.
        """
        # if self.redesign_pocket and self.outer_pocket do not exist, use the below
        if not hasattr(self, "redesign_pocket") or not hasattr(self, "outer_pocket"):
            redesign_pocket, outer_pocket = self.get_protein_pocket(
                inner_threshold, outer_threshold
            )
            self.redesign_pocket = redesign_pocket
            self.outer_pocket = outer_pocket
        else:
            # if alanine_mutate_inner_pocket is called, use the below
            redesign_pocket = self.redesign_pocket
            outer_pocket = self.outer_pocket
        hetatm_pocket = self.get_hetatm_pocket(outer_threshold)
        expanded_residues = self.expand_residue_list(outer_pocket, sequence_buffer)
        new_sequence, new_chain = self.get_new_sequence(expanded_residues)
        protein_pocket_coords = np.array(
            [
                [atom.x, atom.y, atom.z]
                for key, molecule in self.protein_molecules.items()
                for atom in molecule.atoms.values()
                if key in expanded_residues
            ]
        )
        if fix_hetatms:
            hetatm_pocket_coords = []
            hetatm_smiles_list = []
            for key, molecule in self.heteratoms_molecules.items():
                if key in hetatm_pocket:
                    hetatm_coords = np.array(
                        [[atom.x, atom.y, atom.z] for atom in molecule.atoms.values()]
                    )
                    if hetatm_coords.size > 0:
                        mol = Chem.MolFromPDBBlock(molecule.pdbblock)
                        try:
                            hetatom_atom_mapping, hetatom_smiles = self.create_atom_mapping(mol, ccd=key[1])
                        except Exception as e:
                            logger.error(f"Error creating atom mapping for {key}: {e}")
                            try:
                                hetatom_atom_mapping, hetatom_smiles = self.create_atom_mapping(mol)
                            except Exception as e:
                                logger.error(f"Error creating atom mapping for {key} without CCD: {e}")
                                hetatom_atom_mapping = {i: i for i in range(mol.GetNumAtoms())}
                                hetatom_smiles = Chem.MolToSmiles(mol)
                        hetatm_smiles_list.append(hetatom_smiles)
                        reordered_hetatm_coords = np.array(
                            [
                                mol.GetConformer().GetPositions()[hetatom_atom_mapping[i]]
                                for i in range(len(mol.GetConformer().GetPositions()))
                            ]
                        )
                        hetatm_pocket_coords.append(
                            reordered_hetatm_coords
                        )
            if len(hetatm_pocket_coords) > 0:
                hetatm_pocket_coords = np.concatenate(hetatm_pocket_coords, axis=0)
            else:
                hetatm_pocket_coords = np.array([])
        else:
            hetatm_pocket_coords = np.array(
                [
                    [999.999, 999.999, 999.999]
                    for key, molecule in self.heteratoms_molecules.items()
                    for atom in molecule.atoms.values()
                    if key in hetatm_pocket
                ]
            )
        if len(hetatm_pocket_coords) == 0:
            pocket_coords = protein_pocket_coords
        else:
            if hetatm_pocket_coords.ndim == 1:
                hetatm_pocket_coords = hetatm_pocket_coords.reshape(1, -1)
            pocket_coords = np.concatenate(
                [protein_pocket_coords, hetatm_pocket_coords]
            )
        pocket_constraint_residue_indices = self.get_pocket_constraint_residue_indices(
            redesign_pocket, new_chain
        )
        protein_pdbblock, chains_used = self.get_protein_pdbblock(
            new_chain, redesign_pocket
        )
        hetatm_pdbblock = self.get_hetatom_pdbblock(chains_used, hetatm_pocket)
        whole_pdbblock = protein_pdbblock + hetatm_pdbblock
        pocket_modified_residues = self.get_modified_residues(new_chain)
        whole_pocket_atom_indices = list(
            range(len(protein_pocket_coords) + len(hetatm_pocket_coords))
        )
        cleaned_whole_pocket_atom_indices = self.drop_missing_residues(
            pocket_coords, whole_pocket_atom_indices
        )
        redesigned_pocket_atom_indices = self.get_pocket_redesign_atom_indices(
            redesign_pocket, expanded_residues, new_chain, guide_backbone=False, guide_sc=guide_sc
        )
        redesigned_pocket_atom_indices_without_backbone = self.get_pocket_redesign_atom_indices(
            redesign_pocket, expanded_residues, new_chain, guide_backbone=True, guide_sc=guide_sc
        )
        constrain_atom_indices = [
            i
            for i in cleaned_whole_pocket_atom_indices
            if i not in redesigned_pocket_atom_indices
        ]
        constrain_atom_indices_with_backbone = [
            i 
            for i in cleaned_whole_pocket_atom_indices
            if i not in redesigned_pocket_atom_indices_without_backbone
        ]
        constrain_atom_indices_with_ligand = constrain_atom_indices_with_backbone + [
            len(pocket_coords) + j for j in range(self.ligand.GetNumHeavyAtoms())
        ]
        pocket_coords_altered = np.array(
            [
                atom if i in constrain_atom_indices_with_backbone else [999.999, 999.999, 999.999]
                for i, atom in enumerate(pocket_coords)
            ]
        )
        lig_atom_mapping, lig_smiles = self.create_atom_mapping(self.ligand)
        
        lig_mol_copy = Chem.Mol(self.ligand)
        lig_mol_copy = Chem.RemoveAllHs(lig_mol_copy)
        
        reordered_ligand_coords = np.array(
            [
                lig_mol_copy.GetConformer().GetPositions()[lig_atom_mapping[i]]
                for i in range(len(lig_mol_copy.GetConformer().GetPositions()))
            ]
        )
        combined_coords = np.concatenate(
            [pocket_coords_altered, reordered_ligand_coords], axis=0
        )
        pocket_coords_centred = self.centre_coords(
            combined_coords, constrain_atom_indices_with_backbone
        )
        ligand_coords_centred = pocket_coords_centred[
            len(pocket_coords_altered) :
        ]
        return {
            "sequence": new_sequence,
            "smiles": lig_smiles,
            "pocket_coords": pocket_coords_centred,
            "pocket_constraint_residue_indices": pocket_constraint_residue_indices,
            "whole_pocket_atom_indices": constrain_atom_indices,
            "whole_pocket_and_ligand_atom_indices": constrain_atom_indices_with_ligand if guide_ligand else constrain_atom_indices_with_backbone,
            "pdbblock_ref": whole_pdbblock,
            "other_hetatms": hetatm_smiles_list,
            "modified_residues": pocket_modified_residues if not self.replace_modified_res else {},
            "sidechain_atom_mask": self.get_sidechain_atom_indices(expanded_residues),
            "docking_sphere_centre": np.mean(ligand_coords_centred, axis=0),
            "ligand_atom_indices": [
                i + len(pocket_coords_altered) for i in range(self.ligand.GetNumHeavyAtoms())
            ],
        }
    
    def get_sidechain_atom_indices(self, expanded_residues):
        """
        Get indices of side chain atoms (excluding backbone atoms CA, C, N, O) for sidechain clash guidance.
        
        Args:
            expanded_residues (list): List of residue keys in the expanded pocket.
        Returns:
            list: List of indices of side chain atoms.
        """
        backbone_atoms = {"CA", "C", "N", "O"}
        indices = []
        idx = 0
        for res in self.protein_molecules:
            if res in expanded_residues:
                for atom in self.protein_molecules[res].atoms.values():
                    if atom.atom_name not in backbone_atoms:
                        indices.append(idx)
                    idx += 1
        return indices

    def create_atom_mapping(self, mol, ccd=None):
        """Generate full atom mapping between reference and query molecules.

        Args:
            mol (rdkit.Chem.Mol): The query molecule.
            ccd (str, optional): The name of the CCD file. Defaults to None.

        Raises:
            ValueError: If the query molecule has no conformer.
            ValueError: If the atom mapping is incomplete.

        Returns:
            dict: A dictionary mapping atom indices from the query to the reference.
            str: The SMILES representation of the molecule.
        """
        if ccd is not None:
            try:
                mols = self.ccd_pkl
                with open(mols, "rb") as f:
                    ccd_mol = pickle.load(f)
                template_mol = ccd_mol[ccd]
                template_mol = Chem.RemoveAllHs(template_mol)
                query_mol = AllChem.AssignBondOrdersFromTemplate(
                    template_mol, mol
                )
                query_mol = Chem.RemoveAllHs(query_mol)
            except Exception as e:
                logger.error(f"Error in assign bond orders from ccd: {e}. Will just use no bond orders.")
                query_mol = Chem.RemoveAllHs(mol)
        else:
            query_mol = Chem.RemoveAllHs(mol)
        ref_mol = Chem.MolFromSmiles(Chem.MolToSmiles(query_mol))
        ref_mol = Chem.AddHs(ref_mol)
        
        AllChem.EmbedMolecule(ref_mol, randomSeed=42)
        ref_mol = Chem.RemoveAllHs(ref_mol)
        if query_mol.GetNumConformers() == 0:
            raise ValueError("Query molecule has no conformer")
        _, _, atom_mapping = Chem.rdMolAlign.GetBestAlignmentTransform(
            ref_mol, query_mol
        )
        if len(atom_mapping) != query_mol.GetNumAtoms() != len(ref_mol.GetAtoms()):
            raise ValueError(f"Atom mapping incomplete: not all atoms mapped {len(atom_mapping)} / {query_mol.GetNumAtoms()} atoms")
        return {i[0]: i[1] for i in atom_mapping}, Chem.MolToSmiles(mol)

    def centre_coords(self, coords, mask):
        """Centre coordinates based on the mean of the masked indices.
        Args:
            coords (np.ndarray): Array of coordinates.
            mask (list): List of indices to use for centering.
        Returns:
            np.ndarray: Centered coordinates.
        """
        return coords - np.mean(coords[mask], axis=0)

    def drop_missing_residues(self, pocket_coords, whole_pocket_atom_indices):
        """Drop missing residues from the pocket coordinates.

        Args:
            pocket_coords (np.ndarray): Array of pocket coordinates.
            whole_pocket_atom_indices (list): List of all pocket atom indices.

        Returns:
            np.ndarray: Array of pocket coordinates with missing residues dropped.
        """
        missing_residues = []
        # get where pocket coords are 999.999
        for i in whole_pocket_atom_indices:
            if np.all(pocket_coords[i] == 999.999):
                missing_residues.append(i)
        return [i for i in whole_pocket_atom_indices if i not in missing_residues]

    def get_pocket_redesign_atom_indices(
        self, redesign_pocket, expanded_residues, new_chains, guide_backbone=True, guide_sc=False
    ):
        """Get indices of atoms in the redesign pocket.

        Args:
            redesign_pocket (list): List of residue keys in the redesign pocket.
            expanded_residues (list): List of residue keys in the expanded pocket.
            new_chains (list): List of new chain indices.
            guide_backbone (bool, optional): Whether to include backbone atoms. Defaults to True.
            guide_sc (bool, optional): Whether to include side chain atoms. Defaults to False.

        Returns:
            list: List of atom indices in the redesign pocket.
        """
        residue_keys_atoms = []
        backbone_atoms = {"CA", "C", "N", "O"}
            
        count = 0
        for key, molecule in self.protein_molecules.items():
            if key in expanded_residues:
                if key in redesign_pocket:
                    for atom in molecule.atoms.values():
                        if not (guide_backbone and atom.atom_name in backbone_atoms):
                            if not guide_sc:        
                                residue_keys_atoms.append(count)
                        count += 1
                else:
                    count += len(molecule.atoms)
        return residue_keys_atoms

    def get_pocket_constraint_residue_indices(self, redesign_pocket, new_chains):
        """Get indices of residues in the redesign pocket that are part of the new chains.

        Args:
            redesign_pocket (list): List of residue keys in the redesign pocket.
            new_chains (list): List of new chain indices.

        Returns:
            list: List of indices of residues in the redesign pocket that are part of the new chains.
        """
        pocket_constraint_residue_indices = []
        new_chains_combined = [index for chain in new_chains for index in chain]
        for index in new_chains_combined:
            if list(self.protein_molecules.keys())[index] in redesign_pocket:
                pocket_constraint_residue_indices.append(
                    new_chains_combined.index(index)
                )
        return pocket_constraint_residue_indices

    def get_protein_pocket(self, inner_threshold, outer_threshold):
        """Get the protein pocket based on distance thresholds.

        Args:
            inner_threshold (float): Inner distance threshold.
            outer_threshold (float): Outer distance threshold.

        Returns:
            tuple: A tuple containing two lists - the inner and outer pocket residue keys.
        """
        protein_coords = np.array(
            [
                [atom.x, atom.y, atom.z]
                for molecule in self.protein_molecules.values()
                for atom in molecule.atoms.values()
            ]
        )
        protein_keys = [
            key
            for key in self.protein_molecules
            for atom in self.protein_molecules[key].atoms.values()
        ]
        assert len(protein_keys) == len(protein_coords)
        ligand_coords = self.ligand.GetConformer().GetPositions()
        distances = cdist(protein_coords, ligand_coords)
        pocket_indices = np.argwhere(distances < inner_threshold)
        larger_pocket_indices = np.argwhere(distances < outer_threshold)
        pocket_keys = []
        for i in pocket_indices:
            if protein_keys[i[0]] in pocket_keys:
                continue
            pocket_keys.append(protein_keys[i[0]])
        larger_pocket_keys = []
        for i in larger_pocket_indices:
            if protein_keys[i[0]] in larger_pocket_keys:
                continue
            larger_pocket_keys.append(protein_keys[i[0]])
        return pocket_keys, larger_pocket_keys

    def get_hetatm_pocket(self, outer_threshold):
        """Get the heteroatoms in the pocket based on distance threshold.
        
        Args:
            outer_threshold (float): Outer distance threshold.
        Returns:
            list: List of heteroatom residue keys in the pocket.
        """
        if self.heteratoms_molecules == {}:
            return []
        hetatm_coords = np.concatenate(
            [molecule.coords for molecule in self.heteratoms_molecules.values()]
        )
        hetatm_keys = [
            key
            for key in self.heteratoms_molecules
            for atom in self.heteratoms_molecules[key].atoms
        ]
        assert (
            len(hetatm_keys) == hetatm_coords.shape[0]
        ), f"{len(hetatm_keys)} != {hetatm_coords.shape[0]}, {[self.heteratoms_molecules[key].pdbblock for key in self.heteratoms_molecules]}"
        ligand_coords = self.ligand.GetConformer().GetPositions()
        distances = cdist(hetatm_coords, ligand_coords)
        pocket_indices = np.argwhere(distances < outer_threshold)
        # pocket_keys = [hetatm_keys[i[0]] for i in pocket_indices]
        pocket_keys = []
        for i in pocket_indices:
            if hetatm_keys[i[0]] in pocket_keys:
                continue
            pocket_keys.append(hetatm_keys[i[0]])
        return pocket_keys

    def expand_residue_list(self, pocket_residues, buffer):
        """Expand the list of pocket residues by a buffer.

        Args:
            pocket_residues (list): List of residue keys in the pocket.
            buffer (int): Number of residues to include in the expansion.

        Returns:
            list: List of expanded residue keys.
        """
        residue_indices = sorted(
            [
                i
                for i, res in enumerate(list(self.protein_molecules.keys()))
                if res in pocket_residues
            ]
        )
        total_residue_list = list(self.protein_molecules.keys())
        expanded_residues = []
        for i in residue_indices:
            for j in range(i - buffer, i + buffer + 1):
                if j in range(len(total_residue_list)):
                    if total_residue_list[j][3] == total_residue_list[i][3]:
                        if (
                            int(total_residue_list[j][2])
                            - int(total_residue_list[i][2])
                            < buffer
                        ):
                            expanded_residues.append(total_residue_list[j])
        return expanded_residues

    def get_new_sequence(self, residues):
        """Get the new sequence of residues after redesign.

        Args:
            residues (list): List of residue keys to include in the new sequence.

        Returns:
            dict: Dictionary mapping chain letters to their new sequences.
        """
        residue_indices = [
            (res[3], res[2], i)
            for i, res in enumerate(self.protein_molecules.keys())
            if res in residues
        ]
        breakpoints = []
        for i, j in zip(residue_indices[:-1], residue_indices[1:]):
            if i[0] != j[0] or int(j[1]) - int(i[1]) > 1:
                breakpoints.append(j[2])
        new_chains = []
        current_chain = []
        for i in residue_indices:
            if i[2] in breakpoints:
                new_chains.append(current_chain)
                current_chain = []
            current_chain.append(i[2])
        new_chains.append(current_chain)

        new_sequence = {}
        for i, chain in enumerate(new_chains):
            chain_letter = chr(65 + i)
            new_sequence[chain_letter] = ""
            for index in chain:
                new_sequence[chain_letter] += self.three_letter_to_one_letter(
                    # list(self.protein_molecules.keys())[index][1]
                    self.protein_molecules[list(self.protein_molecules.keys())[index]].resname
                )
        return new_sequence, new_chains

    def get_protein_pdbblock(self, new_chains, pocket_residues):
        """Get the protein PDB block for the redesigned pocket.

        Args:
            new_chains (list): List of new chain indices.
            pocket_residues (list): List of residue keys in the pocket.

        Returns:
            str: PDB block for the redesigned pocket.
        """
        new_pdbblock = ""
        chain_letters = []
        for indices in new_chains:
            new_pdbblock_chain = ""
            chain_letter = chr(65 + new_chains.index(indices))
            chain_letters.append(chain_letter)
            for i in indices:
                if list(self.protein_molecules.keys())[i] in pocket_residues:
                    continue
                new_pdbblock_chain += (
                    "\n".join(
                        [
                            line[:21] + chain_letter + line[22:]
                            for line in self.protein_molecules[
                                list(self.protein_molecules.keys())[i]
                            ]
                            .get_new_pdbblock()
                            .split("\n")
                        ]
                    )
                    + "\n"
                )
            new_pdbblock_chain_changed = "\n" + "\n".join(
                [
                    line[:21] + chain_letter + line[22:]
                    for line in new_pdbblock_chain.split("\n")[1:]
                    if line.startswith("ATOM") or line.startswith("HETATM")
                ]
            )
            new_pdbblock += new_pdbblock_chain_changed
        return new_pdbblock, chain_letters

    def get_hetatom_pdbblock(self, used_chain_letters, hetatom_residues):
        """Get the heteroatom PDB block for the redesigned pocket.
        
        Args:
            used_chain_letters (list): List of used chain letters.
            hetatom_residues (list): List of heteroatom residue keys in the pocket.
        Returns:
            str: PDB block for the heteroatoms in the redesigned pocket.
        """
        new_pdbblock = ""
        chain_num = 0
        if len(hetatom_residues) == 0:
            return new_pdbblock
        for res in hetatom_residues:
            chain_num += 1
            chain_letter = chr(65 + len(used_chain_letters) + chain_num)
            new_pdbblock += "\n" + "\n".join(
                [
                    line[:21] + chain_letter + line[22:]
                    for line in self.heteratoms_molecules[res]
                    .get_new_pdbblock()
                    .split("\n")
                ]
            )
        return new_pdbblock

    def get_modified_residues(self, new_chains):
        """Get the modified residues in the redesigned pocket.

        Args:
            new_chains (list): List of new chain indices.

        Returns:
            dict: Dictionary mapping chain letters to their modified residue information.
        """
        modified_residues_pocket = {}
        for indices in new_chains:
            new_chain = chr(65 + new_chains.index(indices))
            for j, i in enumerate(indices):
                res = list(self.protein_molecules.keys())[i]
                if res[1] in MODIFIED_RESIDUES:
                    if res[3] not in modified_residues_pocket:
                        modified_residues_pocket[new_chain] = {}
                    modified_residues_pocket[new_chain] = (j, res[1])
                    print(
                        f"Found modified residue {res[1]} at position {j} in chain {new_chain}"
                    )
        return modified_residues_pocket