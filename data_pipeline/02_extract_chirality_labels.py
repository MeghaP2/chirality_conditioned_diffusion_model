"""
Step 2 of the data pipeline: for every generated conformer (.sdf), determine
its RDKit chiral-center assignment and a set of continuous symmetry measures
(Cs, Ci, S4, S6, S8) via cosymlib. These become the model's chirality
conditioning label (y).

Input:  sdf_files.txt (list of .sdf paths, one per line) — output of step 1
Output: chirality_data.csv — consumed directly by MaceMolecularDataset
"""
import csv
from pathlib import Path

import numpy as np
from openbabel import openbabel
from rdkit import Chem
from rdkit.Chem import FindMolChiralCenters
from cosymlib.file_io import read_generic_structure_file


def extract_chirality_data(sdf_list_path="sdf_files.txt", out_csv="chirality_data.csv"):
    conv = openbabel.OBConversion()
    conv.SetInAndOutFormats("sdf", "xyz")

    files = np.loadtxt(sdf_list_path, dtype=str)
    chiral_info = {f: [] for f in files}

    for fname in files:
        mol = Chem.MolFromMolFile(fname, sanitize=True) # validity check on the molecule
        chiral_centers = FindMolChiralCenters(mol, includeUnassigned=True, includeCIP=True) # reports every potential stereocenter
        chirality = [c for _, c in chiral_centers]

        obmol = openbabel.OBMol()
        conv.ReadFile(obmol, fname)
        xyz_file = str(Path(fname).with_suffix(".xyz"))
        conv.WriteFile(obmol, xyz_file)

        obj = read_generic_structure_file(xyz_file)
        c1 = obj.get_symmetry_measure("cs")
        c2 = obj.get_symmetry_measure("ci")
        c3 = obj.get_symmetry_measure("s4")
        c4 = obj.get_symmetry_measure("s6")
        c5 = obj.get_symmetry_measure("s8")

        chiral_info[fname].extend([chirality, len(chirality), c1, c2, c3, c4, c5])

    with open(out_csv, "w") as f:
        writer = csv.writer(f, delimiter=" ")
        writer.writerow(["#\t", "filename", "chirality", "length_of_chirality",
                          "Cs", "Ci", "S4", "S6", "S8"])
        for fname, values in chiral_info.items():
            chirality_str = ";".join(values[0])  # handles multiple chiral centers, if present
            writer.writerow([fname, chirality_str, *values[1:]])

    return out_csv


if __name__ == "__main__":
    extract_chirality_data()
