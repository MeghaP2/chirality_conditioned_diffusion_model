"""
Step 1 of the data pipeline: generate a diverse set of nonane conformers by
scanning two dihedral angles on a grid, MMFF-relaxing everything else while
holding those two dihedrals fixed at each grid point.

SDMolSupplier returns a list-like object representing every molecule in the .sdf file, 
even if the file only contains one. [0] extracts the first molecule from that container

Input:  nonane.sdf (a single reference conformer)
Output: conf_<i>_<j>.sdf for every grid point that converged
"""
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolTransforms
import numpy as np

mol = Chem.SDMolSupplier("nonane.sdf", removeHs=False)[0] # returns the first molecule


# dihedral 1: atoms 12,9,0,2  (C5-C4-C1-C2)
# dihedral 2: atoms 2,16,19,22
DIHEDRAL_1 = (12, 9, 0, 2)
DIHEDRAL_2 = (2, 16, 19, 22)


def optimize_with_fixed_dihedral(mol, dihedral_atoms1, dihedral_atoms2,
                                  target_angle1, target_angle2, force_constant=1.0e6):
    """
    Relaxes everything EXCEPT the two given dihedrals, which are held fixed
    at target_angle1 / target_angle2 (degrees).
    Returns 0 if converged, 1 if not (RDKit's ff.Minimize convention).
    """
    ff = AllChem.MMFFGetMoleculeForceField(mol, AllChem.MMFFGetMoleculeProperties(mol))
    a, b, c, d = dihedral_atoms1
    e, f, g, h = dihedral_atoms2
    ff.MMFFAddTorsionConstraint(a, b, c, d, False, target_angle1, target_angle1, force_constant) # False: absolute dihedral value, not relative
    ff.MMFFAddTorsionConstraint(e, f, g, h, False, target_angle2, target_angle2, force_constant)
    return ff.Minimize(maxIts=500)


def generate_conformers(mol, angles1, angles2, force_constant=1.0e6):
    n_converged, n_failed = 0, 0
    failed_list = []

    for i, angle1 in enumerate(angles1):
        for j, angle2 in enumerate(angles2):
            mol_copy = Chem.Mol(mol)
            conf = mol_copy.GetConformer()
            rdMolTransforms.SetDihedralDeg(conf, *DIHEDRAL_1, angle1) # *DIHEDRAL_1 unpacks the tuple (12, 9, 0, 2)
            rdMolTransforms.SetDihedralDeg(conf, *DIHEDRAL_2, angle2)

            converged = optimize_with_fixed_dihedral(mol_copy, 
                    DIHEDRAL_1, DIHEDRAL_2, angle1, angle2,
                    force_constant=force_constant)

            if converged == 0:
                n_converged += 1
                with Chem.SDWriter(f"conf_{i}_{j}.sdf") as writer:
                    writer.write(mol_copy)
            else:
                n_failed += 1
                failed_list.append((i, j, angle1, angle2))

        print(f"outer step: {i + 1}/{len(angles1)} done - "
              f"converged so far: {n_converged}, failed: {n_failed}")

    total = n_converged + n_failed
    print("\n------ SUMMARY ------")
    print(f"Total attempted: {total}")
    print(f"Converged: {n_converged} ({100 * n_converged / total:.1f}%)")
    print(f"Failed to converge: {n_failed} ({100 * n_failed / total:.1f}%)")
    if failed_list:
        print("Failed structures are: (i, j, angle1, angle2):")
        
    return n_converged, n_failed, failed_list


if __name__ == "__main__":
    # two angle bands per dihedral -> covers the physically distinct rotamer regions
    angles1 = np.concatenate([np.random.uniform(-70, -50, size=50),
        np.random.uniform(50, 60, size=50)])
    angles2 = np.concatenate([np.random.uniform(-180, -160, size=50),
        np.random.uniform(160, 180, size=50)])
    generate_conformers(mol, angles1, angles2)

 
# NOTE: AllChem.EmbedMultipleConfs(mol_copy, numConfs=20) is a faster
# alternative for generating many conformers, but needs post-processing
# (dedup / relaxation) before use, hence not used as-is here.
