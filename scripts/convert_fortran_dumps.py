#!/usr/bin/env python3
"""Convert Fortran binary dump files to .npy format.

The binary format is:
  ndim (int32) — number of dimensions
  shape (ndim x int32) — array dimensions (Fortran column-major)
  data — raw float64 or int32

ndim == 0: scalar float64
ndim == -1: scalar int32
ndim >= 1: array

Complex arrays are stored as separate _re.dat and _im.dat files.
"""

import sys
import struct
from pathlib import Path
import numpy as np


def read_dump(path: Path) -> np.ndarray:
    """Read a single Fortran dump file (big-endian due to -fconvert=big-endian)."""
    with open(path, "rb") as f:
        ndim = struct.unpack(">i", f.read(4))[0]

        if ndim == -1:
            # scalar int
            val = struct.unpack(">i", f.read(4))[0]
            return np.array(val, dtype=np.int32)
        elif ndim == 0:
            # scalar float64
            val = struct.unpack(">d", f.read(8))[0]
            return np.array(val, dtype=np.float64)
        else:
            shape = []
            for _ in range(ndim):
                shape.append(struct.unpack(">i", f.read(4))[0])
            # Determine dtype from remaining data size
            remaining = f.read()
            total_elements = 1
            for s in shape:
                total_elements *= s
            if len(remaining) == total_elements * 8:
                data = np.frombuffer(remaining, dtype=">f8")
            elif len(remaining) == total_elements * 4:
                data = np.frombuffer(remaining, dtype=">i4")
            else:
                raise ValueError(
                    f"Unexpected data size {len(remaining)} for shape {shape} "
                    f"(got {len(remaining)} bytes, expected {total_elements * 8} or {total_elements * 4})"
                )
            # Convert to native endianness
            data = data.astype(data.dtype.newbyteorder("="))
            # Fortran is column-major, reshape accordingly
            if ndim == 1:
                return data.copy()
            else:
                return data.reshape(shape, order="F").copy()


def main():
    dump_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("fortran_dumps")
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("fortran_ref")
    out_dir.mkdir(parents=True, exist_ok=True)

    dat_files = sorted(dump_dir.glob("*.dat"))
    if not dat_files:
        print(f"No .dat files found in {dump_dir}")
        return

    # Group complex files (name_re.dat + name_im.dat)
    re_files = {}
    im_files = {}
    real_files = {}

    for f in dat_files:
        stem = f.stem
        if stem.endswith("_re"):
            base = stem[:-3]
            re_files[base] = f
        elif stem.endswith("_im"):
            base = stem[:-3]
            im_files[base] = f
        else:
            real_files[stem] = f

    # Convert real/int arrays
    for name, path in sorted(real_files.items()):
        arr = read_dump(path)
        np.save(out_dir / f"{name}.npy", arr)
        print(f"  {name}: shape={arr.shape}, dtype={arr.dtype}")

    # Convert complex arrays
    for name in sorted(re_files.keys()):
        if name not in im_files:
            print(f"  WARNING: {name}_re.dat without matching _im.dat")
            continue
        re = read_dump(re_files[name])
        im = read_dump(im_files[name])
        arr = re + 1j * im
        np.save(out_dir / f"{name}.npy", arr)
        print(f"  {name}: shape={arr.shape}, dtype={arr.dtype} (complex)")

    print(f"\nConverted {len(real_files) + len(re_files)} arrays to {out_dir}/")


if __name__ == "__main__":
    main()
