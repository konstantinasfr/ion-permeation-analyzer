# 🧪 Ion-to-Residue Distance Calculation Guide

This document describes how the `calculate_distances()` function computes the distances between potassium ions (K⁺) and selected residues across a molecular dynamics trajectory using MDAnalysis. The logic adapts based on different options to suit structural, electrostatic, or charge-based interpretations.

---

## 📌 Overview

The function calculates frame-by-frame distances between a **permeating K⁺ ion** and selected **pore-lining residues**. These distances are used to understand ion–residue interactions, such as:

- Proximity to charged or dipolar residues
- Interaction timing during permeation
- Coordination near the selectivity filter (SF)

---

## ⚙️ Input Parameters

### Flags that control the calculation:

- `use_ca_only`:  
  → Use only the alpha carbon (Cα) of each residue (geometry only)

- `use_min_distances`:  
  → Use the **minimum distance** between the ion and **any atom** of the residue (non-specific contact)

- `use_charges`:  
  → Use **biologically meaningful charge centers**:
  - For **Glu (E)**: midpoint between OE1 and OE2 (negative charge)
  - For **Asn (N)**: midpoint between OD1 and ND2 (dipole)
  - For **SF residues**: minimum atom-wise distance (as in `use_min_distances`)

> ⚠️ Only one of these flags should be `True` at a time.

---

## 🧬 Residue Categories

### 1. Glutamate Residues (Negatively Charged)
- Residues: `98, 423, 748, 1073`
- Charge center: midpoint of OE1 and OE2
- Represents the **delocalized negative charge** of the carboxylate group

### 2. Asparagine Residues (Dipolar)
- Residues: `130, 455, 780, 1105`
- Dipole center: midpoint of OD1 and ND2
- Models the dipolar amide group

### 3. Selectivity Filter Residues
- Residues: `100, 425, 750, 1075`
- Use average of **minimum atom distances** (no dipole or charge center applied)

---

## 📐 How Distance is Computed (Per Frame)

For each frame from `start_frame` to `exit_frame`:
1. Retrieve the 3D position of the ion.
2. For each residue:
   - If `use_ca_only` → compute Euclidean distance to the Cα atom
   - If `use_min_distances` → compute minimum distance to any atom
   - If `use_charges`:
     - For Glu → compute distance to OE1/OE2 midpoint
     - For Asn → compute distance to OD1/ND2 midpoint
     - For SF → same as `use_min_distances`
3. Also compute distances to ions that exist in SF and in the region between SF and HBC gate.

---

## 🧾 Output Format

Returns a dictionary:

```python
{
  ion_id: [
    {
      'frame': int,
      'residues': {
        98: dist,
        423: dist,
        ...
        'SF': avg_sf_dist
      },
      'ions': {
        other_ion_id: dist,
        ...
      }
    },
    ...
  ]
}
