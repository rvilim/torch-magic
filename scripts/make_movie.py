#!/usr/bin/env python
"""Convert a MagIC movie file to mp4 via matplotlib + ffmpeg.

Usage: uv run python scripts/make_movie.py output/rvi-test-hires/Br_CMB_mov.torch
       uv run python scripts/make_movie.py output/rvi-test-hires/Vr_EQU_mov.torch --cmap seismic
"""
import argparse
import struct
import subprocess
import sys
import tempfile
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def fort_read(f, dtype='>f4'):
    marker = f.read(4)
    if not marker:
        return None
    n = struct.unpack('>i', marker)[0]
    data = f.read(n)
    f.read(4)
    return np.frombuffer(data, dtype=np.dtype(dtype))


def read_movie(path):
    """Read a MagIC movie file, return header info + list of (time, frame) tuples."""
    with open(path, 'rb') as f:
        version = fort_read(f, '>i4')[0]
        sf = fort_read(f, '>i4')
        n_surface, n_fields = int(sf[0]), int(sf[1])
        const = fort_read(f, '>f8')[0]
        ftypes = fort_read(f, '>i4')

        # runid (64 bytes)
        m = struct.unpack('>i', f.read(4))[0]
        runid = f.read(m).decode('ascii', errors='replace').strip()
        f.read(4)

        dims = fort_read(f, '>i4')
        n_r_mov_tot = int(dims[0])
        n_r_max = int(dims[1])
        n_theta_max = int(dims[3])
        n_phi_max = int(dims[4])

        phys = fort_read(f, '>f4')
        r_grid = fort_read(f, '>f4')
        theta = fort_read(f, '>f4')
        phi = fort_read(f, '>f4')

        frames = []
        while True:
            t = fort_read(f, '>f4')
            if t is None:
                break
            data = fort_read(f, '>f4')
            frames.append((t[0], data))

    return {
        'n_surface': n_surface,
        'n_r_max': n_r_max,
        'n_theta_max': n_theta_max,
        'n_phi_max': n_phi_max,
        'theta': theta,
        'phi': phi,
        'r_grid': r_grid,
        'field_type': int(ftypes[0]),
        'frames': frames,
    }


FIELD_NAMES = {1: 'Br', 2: 'Bθ', 3: 'Bφ', 4: 'Vr', 5: 'Vθ', 6: 'Vφ', 7: 'Temperature'}
SURFACE_NAMES = {1: 'CMB', 2: 'Equator'}


def main():
    parser = argparse.ArgumentParser(description='Convert MagIC movie file to mp4')
    parser.add_argument('input', help='Path to movie file (e.g., Br_CMB_mov.torch)')
    parser.add_argument('-o', '--output', help='Output mp4 path (default: same name .mp4)')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second (default 30)')
    parser.add_argument('--cmap', default='RdBu_r', help='Colormap (default RdBu_r)')
    parser.add_argument('--levels', type=int, default=64, help='Contour levels (default 64)')
    parser.add_argument('--dpi', type=int, default=150, help='DPI (default 150)')
    parser.add_argument('--symmetric', action='store_true', default=True,
                        help='Symmetric colorbar around 0 (default)')
    parser.add_argument('--no-symmetric', dest='symmetric', action='store_false')
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.splitext(args.input)[0] + '.mp4'

    print(f'Reading {args.input}...')
    mov = read_movie(args.input)
    n_frames = len(mov['frames'])
    field_name = FIELD_NAMES.get(mov['field_type'], f'Field {mov["field_type"]}')
    surface_name = SURFACE_NAMES.get(mov['n_surface'], f'Surface {mov["n_surface"]}')
    print(f'  {n_frames} frames, {field_name} at {surface_name}')

    # Determine global color range
    all_data = np.array([f[1] for f in mov['frames']])
    if args.symmetric:
        vmax = np.abs(all_data).max()
        vmin = -vmax
    else:
        vmin, vmax = all_data.min(), all_data.max()
    print(f'  Color range: [{vmin:.3e}, {vmax:.3e}]')

    with tempfile.TemporaryDirectory() as tmpdir:
        print(f'Rendering {n_frames} frames...')

        for i, (time, data) in enumerate(mov['frames']):
            if mov['n_surface'] == 1:
                # r=const (CMB) — Mollweide projection
                frame = data.reshape(mov['n_theta_max'], mov['n_phi_max'])
                lon = np.degrees(mov['phi'])
                lat = 90 - np.degrees(mov['theta'])
                lon_wrap = np.append(lon, lon[0] + 360)
                frame_wrap = np.column_stack([frame, frame[:, 0]])

                fig, ax = plt.subplots(figsize=(10, 5), subplot_kw={'projection': 'mollweide'})
                lon_rad = np.radians(lon_wrap - 180)
                lat_rad = np.radians(lat)
                LON, LAT = np.meshgrid(lon_rad, lat_rad)
                ax.contourf(LON, LAT, frame_wrap, levels=args.levels,
                            cmap=args.cmap, vmin=vmin, vmax=vmax)
                ax.grid(True, alpha=0.3)
                ax.set_title(f'{field_name} at {surface_name}   t = {time:.5f}', fontsize=13)

            elif mov['n_surface'] == 2:
                # theta=const (equator) — rectangular (phi, r) plot
                frame = data.reshape(mov['n_r_max'], mov['n_phi_max'])
                phi_deg = np.degrees(mov['phi'])
                r_norm = mov['r_grid'][:mov['n_r_max']]

                fig, ax = plt.subplots(figsize=(10, 4))
                PHI, R = np.meshgrid(phi_deg, r_norm)
                ax.contourf(PHI, R, frame, levels=args.levels,
                            cmap=args.cmap, vmin=vmin, vmax=vmax)
                ax.set_xlabel('Longitude (°)')
                ax.set_ylabel('r / r_cmb')
                ax.set_title(f'{field_name} at {surface_name}   t = {time:.5f}', fontsize=13)

            plt.tight_layout()
            plt.savefig(os.path.join(tmpdir, f'frame_{i:05d}.png'), dpi=args.dpi)
            plt.close(fig)

            if (i + 1) % 50 == 0 or i == n_frames - 1:
                print(f'  {i + 1}/{n_frames}')

        # Stitch with ffmpeg
        print(f'Encoding {args.output} at {args.fps} fps...')
        cmd = [
            'ffmpeg', '-y', '-framerate', str(args.fps),
            '-i', os.path.join(tmpdir, 'frame_%05d.png'),
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
            '-crf', '18', '-preset', 'medium',
            args.output
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f'ffmpeg failed:\n{result.stderr[-500:]}')
            sys.exit(1)

    print(f'Done: {args.output}')


if __name__ == '__main__':
    main()
