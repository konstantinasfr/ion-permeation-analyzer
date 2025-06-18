# Ion Permeation Analysis: Recording Permeation Events

This module is designed to detect ion permeation events through an ion channel in an MD trajectory. The core idea is to track the movement of each ion relative to the cylindrical geometry of the channel and determine whether it enters and exits through the defined gates. The logic combines vector algebra, logical flags, and optional biochemical filters (e.g., Asn dipole check) to ensure robust and biologically meaningful detection.

## `_check_ion_position(...)`

**Purpose:**

Detect whether an ion permeates the channel in the current frame and record the event if it exits through the lower gate after entering through the upper gate.

**Algorithmic Steps:**

1.  **Project ion position along the channel axis:**

    ```python
    ion_vec = ion_pos - channel.channel_center
    ion_z = np.dot(ion_vec, channel.channel_axis)
    ```

    * This transforms the 3D ion position into a 1D value (`ion_z`) along the axis of the channel to track its progression through the pore.

2.  **Track entry and exit via boolean flags:**

    * Each ion has state flags:
        * `upper_flag`: set when the ion enters the pore
        * `lower_flag`: set when the ion exits through the lower gate
    * Frame numbers are recorded for each flag to calculate residence time.

3.  **Cylinder check:**

    * We check whether the ion is inside the channel cylinder using:

        ```python
        in_cylinder = channel.is_within_cylinder(ion_pos)
        ```

4.  **Ion Enters the Channel:**

    * If the ion is in the cylinder and has not entered before, we set `upper_flag = 1` and record the frame.

5.  **Ion Exits the Channel:**

    * If the ion is no longer in the cylinder but has `upper_flag = 1` and crosses the lower boundary, we interpret it as permeation.
    * The code optionally checks for dipole alignment with Asn residues in channel 2:
        * It ensures that the ion passed near the ND2–OD1 dipole of an Asn residue before counting it as a true permeation event.
    * If valid, the event is saved with `start_frame`, `exit_frame`, and `total_time`.

6.  **Re-entry detection:**

    * If the ion goes back toward the upper region after permeating, the flags are reset to allow detection of a second permeation.
    * The method uses `prev_ion_z` to detect the direction of motion across frames.

7.  **End-of-trajectory catch:**

    * If the ion is still in the channel at the last frame, and `upper_flag = 1`, it's treated as an exit event for completeness.

**Key Insight:**

This is a state-machine approach for each ion:

* It transitions through: Outside → Inside (upper) → Exit (lower)
* The states are updated based on geometric checks (projections, distances), and events are logged only if the full sequence is completed.

## `compute_geometry(...)`

**Purpose:**

Defines the geometric basis for analyzing ion motion — including upper/lower gate centers, channel axis, and center of the pore.

**Algorithmic Steps:**

1.  Select gate residues from topology (`resid`) and extract their atom coordinates.

2.  Z-sorting logic:

    * For each residue, atoms are sorted by their Z-coordinate (`coords[:, 2].argsort()`).
    * Either the top or bottom Z-index is used to select a key atom from each gate.
    * These selected atoms define the upper and lower gate centers by their center of mass.

3.  Compute channel geometry:

    * `channel_vector` = vector from upper to lower gate
    * `channel_length` = Euclidean norm of the vector
    * `channel_axis` = normalized vector → this defines the principal axis used to project ion positions
    * `channel_center` = midpoint of the two gate centers

## `is_within_cylinder(pos)`

**Purpose:**

Determines whether a given ion position lies inside a virtual cylinder representing the channel.

**Algorithmic Steps:**

1.  **Relative vector:**

    ```python
    rel_vector = pos - self.channel_center
    ```

2.  **Project ion onto channel axis:**

    ```python
    proj = np.dot(rel_vector, self.channel_axis) * self.channel_axis
    axial_pos = np.dot(rel_vector, self.channel_axis)
    ```

3.  **Radial distance:**

    ```python
    radial = rel_vector - proj
    radial_dist = np.linalg.norm(radial)
    ```

4.  **Final cylinder condition:**

    ```python
    return radial_dist <= self.radius and abs(axial_pos) <= self.channel_length / 2
    ```

**Key Insight:**

This method uses cylindrical projection — i.e., breaking a 3D vector into components along and perpendicular to the channel axis — to define a spatial region through which ions must pass to be counted as permeating.

## Special Handling: Asn Dipole Check

For Channel 2 only, the algorithm adds a biochemical filter:

* Asn residues (with ND2 and OD1 atoms) are used to define a dipole.
* The ion must pass within a 7 Å threshold of this dipole to be considered a true permeation.
* This models potential electrostatic gating or dipole alignment effects on ion motion.
