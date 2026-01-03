# Octave Implementation

GNU Octave scripts for computing and visualizing electromagnetic fields of rings of charge and current.

## Magnetic Field Functions

### `loop_bfield(radius, current, r, a)`

Compute the magnetic field of a current loop (ring of current) in cylindrical coordinates.

**Parameters:**
- `radius` - radius of the current loop (m)
- `current` - current through the loop (A)
- `r` - radial distance from axis (m), can be a vector/matrix
- `a` - axial distance from loop plane (m), can be a vector/matrix

**Returns:**
- `Br` - radial component of magnetic field (T)
- `Ba` - axial component of magnetic field (T)

**Example:**
```octave
[Br, Ba] = loop_bfield(1.0, 1.0, 0.5, 0.3)
```

### `loop_bfield3d(center, normal, radius, current, pos)`

Compute the 3D magnetic field of a current loop at arbitrary positions.

**Parameters:**
- `center` - 3D vector [x, y, z] of loop center (m)
- `normal` - 3D unit normal vector of the loop plane
- `radius` - radius of the current loop (m)
- `current` - current through the loop (A)
- `pos` - Nx3 matrix where each row is a position [x, y, z] (m)

**Returns:**
- `B` - Nx3 matrix where each row is the field vector [Bx, By, Bz] (T)

**Example:**
```octave
B = loop_bfield3d([0 0 0], [0 0 1], 0.15, 1.0, [0.2 0 0; 1 2 3])
```

## Electric Field Functions

### `loop_efield(radius, charge, r, a)`

Compute the electric field of a ring of charge in cylindrical coordinates.

**Parameters:**
- `radius` - radius of the ring (m)
- `charge` - total charge on the ring (C)
- `r` - radial distance from axis (m), can be a vector/matrix
- `a` - axial distance from ring plane (m), can be a vector/matrix

**Returns:**
- `Er` - radial component of electric field (V/m)
- `Ea` - axial component of electric field (V/m)

**Example:**
```octave
[Er, Ea] = loop_efield(1.0, 1e-9, 0.5, 0.3)
```

### `loop_efield3d(center, normal, radius, charge, pos)`

Compute the 3D electric field of a ring of charge at arbitrary positions.

**Parameters:**
- `center` - 3D vector [x, y, z] of ring center (m)
- `normal` - 3D unit normal vector of the ring plane
- `radius` - radius of the ring (m)
- `charge` - total charge on the ring (C)
- `pos` - Nx3 matrix where each row is a position [x, y, z] (m)

**Returns:**
- `E` - Nx3 matrix where each row is the field vector [Ex, Ey, Ez] (V/m)

**Example:**
```octave
E = loop_efield3d([0 0 0], [0 0 1], 1.0, 1e-9, [0.5 0 0.3; 0 0.5 0.3])
```

## Visualization

### `plot_bfield(radius, current, extent, output_file)`

Visualize the magnetic field with colored intensity (log scale), contour lines, and streamlines with arrowheads.

**Parameters:**
- `radius` - radius of the current loop (m), default 1.0
- `current` - current through the loop (A), default 1.0
- `extent` - plot extent in meters, default 2.0
- `output_file` - output PNG filename, or empty to display

**Example:**
```octave
plot_bfield(1.0, 1.0, 2.0, "bfield_octave.png")
```

### `plot_efield(radius, charge, extent, output_file)`

Visualize the electric field with colored intensity (log scale), contour lines, and streamlines with arrowheads.

**Parameters:**
- `radius` - radius of the ring (m), default 1.0
- `charge` - total charge on the ring (C), default 1e-9
- `extent` - plot extent in meters, default 2.0
- `output_file` - output PNG filename, or empty to display

**Example:**
```octave
plot_efield(1.0, 1e-9, 2.0, "efield_octave.png")
```

## Usage

Generate both plots using make:

```bash
make
```

Or run directly:

```octave
plot_bfield(1.0, 1.0, 2.0, "bfield_octave.png")
plot_efield(1.0, 1e-9, 2.0, "efield_octave.png")
```

## Dependencies

Requires the `specfun` package for elliptic integrals (`ellipke`).
