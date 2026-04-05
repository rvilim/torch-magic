#!/usr/bin/env python
"""Render MagIC movie files to mp4 using Modal for parallel frame rendering.

Reads movie files from either a local directory or a Modal volume,
renders frames in parallel on Modal (many CPUs), encodes to mp4, and
downloads the result locally.

Usage:
    # From local output directory
    uv run modal run scripts/make_movie_modal.py -- output/modal_20260405_125201

    # From Modal volume (just the run name)
    uv run modal run scripts/make_movie_modal.py -- --volume 20260405_125201

    # Options
    uv run modal run scripts/make_movie_modal.py -- --volume 20260405_125201 --fps 60 --dpi 200
"""
import os
import struct

import modal
import numpy as np

app = modal.App("magic-movie")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg")
    .pip_install("numpy", "matplotlib")
)

vol = modal.Volume.from_name("magic-output", create_if_missing=True)


# ---------------------------------------------------------------------------
# Movie file reading (same logic as make_movie.py)
# ---------------------------------------------------------------------------

def fort_read(f, dtype='>f4'):
    marker = f.read(4)
    if not marker:
        return None
    n = struct.unpack('>i', marker)[0]
    data = f.read(n)
    f.read(4)
    return np.frombuffer(data, dtype=np.dtype(dtype))


def read_movie_bytes(raw: bytes) -> dict:
    """Parse a MagIC movie file from raw bytes."""
    import io
    f = io.BytesIO(raw)
    version = fort_read(f, '>i4')[0]
    sf = fort_read(f, '>i4')
    n_surface, n_fields = int(sf[0]), int(sf[1])
    const = fort_read(f, '>f8')[0]
    ftypes = fort_read(f, '>i4')
    m = struct.unpack('>i', f.read(4))[0]
    runid = f.read(m).decode('ascii', errors='replace').strip()
    f.read(4)
    dims = fort_read(f, '>i4')
    phys = fort_read(f, '>f4')
    r_grid = fort_read(f, '>f4')
    theta = fort_read(f, '>f4')
    phi = fort_read(f, '>f4')
    # Convert big-endian to native byte order for all float arrays
    r_grid = r_grid.astype(np.float32)
    theta = theta.astype(np.float32)
    phi = phi.astype(np.float32)
    frames = []
    while True:
        t = fort_read(f, '>f4')
        if t is None:
            break
        data = fort_read(f, '>f4').astype(np.float32)
        frames.append((float(t[0]), data.tobytes()))
    return {
        'n_surface': n_surface,
        'n_r_max': int(dims[1]),
        'n_theta_max': int(dims[3]),
        'n_phi_max': int(dims[4]),
        'theta': theta.tobytes(), 'phi': phi.tobytes(),
        'r_grid': r_grid.tobytes(),
        'field_type': int(ftypes[0]),
        'frame_times': [t for t, _ in frames],
        'frame_data': [d for _, d in frames],
    }


FIELD_NAMES = {1: 'Br', 2: 'Bth', 3: 'Bph', 4: 'Vr', 5: 'Vth', 6: 'Vph', 7: 'T'}
SURFACE_NAMES = {1: 'CMB', 2: 'Equator'}


# ---------------------------------------------------------------------------
# Remote rendering: one function call per batch of frames
# ---------------------------------------------------------------------------

@app.function(image=image, timeout=600)
def render_batch(mov_info: dict, frame_indices: list, frame_data_list: list,
                 frame_times: list, vmin: float, vmax: float,
                 cmap: str, levels: int, dpi: int,
                 field_name: str, surface_name: str) -> list:
    """Render a batch of frames, return list of (index, png_bytes)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import io as _io

    theta = np.frombuffer(mov_info['theta'], dtype=np.float32)
    phi = np.frombuffer(mov_info['phi'], dtype=np.float32)
    r_grid = np.frombuffer(mov_info['r_grid'], dtype=np.float32)
    n_surface = mov_info['n_surface']
    n_theta = mov_info['n_theta_max']
    n_phi = mov_info['n_phi_max']
    n_r = mov_info['n_r_max']

    # Precompute grids
    if n_surface == 1:
        lon = np.degrees(phi)
        lat = 90 - np.degrees(theta)
        lon_wrap = np.append(lon, lon[0] + 360)
        lon_rad = np.radians(lon_wrap - 180)
        lat_rad = np.radians(lat)
        LON, LAT = np.meshgrid(lon_rad, lat_rad)
    elif n_surface == 2:
        phi_deg = np.degrees(phi)
        r_norm = r_grid[:n_r]

    level_arr = np.linspace(vmin, vmax, levels + 1)

    results = []
    for idx, raw, time in zip(frame_indices, frame_data_list, frame_times):
        data = np.frombuffer(raw, dtype=np.float32).copy()
        if idx == frame_indices[0]:
            print(f"  Frame {idx}: len={len(data)}, min={data.min():.3e}, "
                  f"max={data.max():.3e}, vmin={vmin:.3e}, vmax={vmax:.3e}")

        if n_surface == 1:
            frame = data.reshape(n_theta, n_phi)
            frame_wrap = np.column_stack([frame, frame[:, 0]])
            fig, ax = plt.subplots(figsize=(10, 5),
                                   subplot_kw={'projection': 'mollweide'})
            ax.contourf(LON, LAT, frame_wrap, levels=level_arr,
                        cmap=cmap, extend='both')
            ax.grid(True, alpha=0.3)
            ax.set_title(f'{field_name} at {surface_name}   t = {time:.5f}',
                         fontsize=13)
        elif n_surface == 2:
            frame = data.reshape(n_r, n_phi)
            fig, ax = plt.subplots(figsize=(10, 4))
            PHI, R = np.meshgrid(phi_deg, r_norm)
            ax.contourf(PHI, R, frame, levels=level_arr,
                        cmap=cmap, extend='both')
            ax.set_xlabel('Longitude (deg)')
            ax.set_ylabel('r / r_cmb')
            ax.set_title(f'{field_name} at {surface_name}   t = {time:.5f}',
                         fontsize=13)
        else:
            continue

        plt.tight_layout()
        buf = _io.BytesIO()
        plt.savefig(buf, format='png', dpi=dpi)
        plt.close(fig)
        results.append((idx, buf.getvalue()))

    print(f"  Batch done: frames {frame_indices[0]}-{frame_indices[-1]}")
    return results


@app.function(image=image, timeout=600, volumes={"/output": vol})
def encode_mp4(png_map: dict, fps: int, movie_name: str, run_name: str) -> bytes:
    """Assemble PNGs into mp4, return mp4 bytes."""
    import subprocess
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        for idx, png_bytes in sorted(png_map.items()):
            with open(os.path.join(tmpdir, f'frame_{idx:05d}.png'), 'wb') as f:
                f.write(png_bytes)

        out_path = os.path.join(tmpdir, 'out.mp4')
        cmd = [
            'ffmpeg', '-y', '-framerate', str(fps),
            '-i', os.path.join(tmpdir, 'frame_%05d.png'),
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
            '-crf', '18', '-preset', 'medium',
            out_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed:\n{result.stderr[-500:]}")

        with open(out_path, 'rb') as f:
            mp4_bytes = f.read()

        # Also save to volume if run_name provided
        if run_name:
            vol_path = f"/output/{run_name}/{movie_name}.mp4"
            os.makedirs(os.path.dirname(vol_path), exist_ok=True)
            with open(vol_path, 'wb') as f:
                f.write(mp4_bytes)
            vol.commit()

    return mp4_bytes


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(
    local_dir: str = "",
    volume: str = "",
    name: str = "",
    fps: int = 30,
    cmap: str = "RdBu_r",
    levels: int = 64,
    dpi: int = 150,
    batch_size: int = 50,
    max_frames: int = 0,
):
    if not local_dir and not volume:
        print("Error: provide either a local dir or --volume <run_name>")
        return

    # Find movie files
    if local_dir:
        movie_files = [f for f in os.listdir(local_dir) if f.endswith('_mov.torch')]
        run_name = ""
    else:
        run_name = volume
        vol_ref = modal.Volume.from_name("magic-output")
        movie_files = [e.path for e in vol_ref.listdir(f"/{run_name}")
                       if e.path.endswith('_mov.torch')]

    # Filter by name if specified (e.g. --name Br_CMB or --name Vr)
    if name:
        movie_files = [f for f in movie_files if name in f]

    if not movie_files:
        print(f"No matching *_mov.torch files found")
        return

    print(f"Found {len(movie_files)} movie file(s): {', '.join(movie_files)}")

    for movie_file in movie_files:
        print(f"\n{'='*60}")
        print(f"Processing: {movie_file}")

        # Read the movie file
        print(f"  Reading movie file...")
        if local_dir:
            with open(os.path.join(local_dir, movie_file), 'rb') as f:
                raw = f.read()
        else:
            vol_ref = modal.Volume.from_name("magic-output")
            # movie_file from listdir already includes run_name prefix
            chunks = []
            for chunk in vol_ref.read_file(f"/{movie_file}"):
                chunks.append(chunk)
            raw = b''.join(chunks)

        mov = read_movie_bytes(raw)
        n_frames = len(mov['frame_times'])
        if max_frames > 0 and n_frames > max_frames:
            mov['frame_times'] = mov['frame_times'][:max_frames]
            mov['frame_data'] = mov['frame_data'][:max_frames]
            n_frames = max_frames
        field_name = FIELD_NAMES.get(mov['field_type'], f'Field{mov["field_type"]}')
        surface_name = SURFACE_NAMES.get(mov['n_surface'], f'Surf{mov["n_surface"]}')
        print(f"  {n_frames} frames, {field_name} at {surface_name}")

        # Compute color range using percentile to ignore extreme outliers
        all_data = np.concatenate([
            np.frombuffer(d, dtype=np.float32) for d in mov['frame_data']
        ])
        p = np.percentile(np.abs(all_data), 99)
        vmax = float(p)
        vmin = -vmax
        print(f"  Color range (99th pct): [{vmin:.3e}, {vmax:.3e}]")
        print(f"  Data range: [{all_data.min():.3e}, {all_data.max():.3e}]")
        del all_data

        # Prepare mov_info (without frame data — that goes in batches)
        mov_info = {k: v for k, v in mov.items()
                    if k not in ('frame_data', 'frame_times')}

        # Split into batches and dispatch
        batches = []
        for start in range(0, n_frames, batch_size):
            end = min(start + batch_size, n_frames)
            indices = list(range(start, end))
            data_list = mov['frame_data'][start:end]
            times = mov['frame_times'][start:end]
            batches.append((indices, data_list, times))

        print(f"  Rendering {n_frames} frames in {len(batches)} batches "
              f"(batch_size={batch_size})...")

        # Launch all batches in parallel
        import time as _time
        t_render_start = _time.perf_counter()
        print(f"  Dispatching {len(batches)} batches to Modal...")
        futures = []
        for i, (indices, data_list, times) in enumerate(batches):
            futures.append(
                render_batch.spawn(
                    mov_info, indices, data_list, times,
                    vmin, vmax, cmap, levels, dpi,
                    field_name, surface_name,
                )
            )
        print(f"  All batches dispatched, waiting for results...")

        # Collect results
        png_map = {}
        done = 0
        for future in futures:
            batch_result = future.get()
            for idx, png_bytes in batch_result:
                png_map[idx] = png_bytes
            done += len(batch_result)
            elapsed = _time.perf_counter() - t_render_start
            print(f"  {done}/{n_frames} frames rendered ({elapsed:.1f}s)")

        # Encode to mp4
        movie_name = os.path.splitext(os.path.basename(movie_file))[0]
        print(f"  Encoding mp4 ({fps} fps)...")
        mp4_bytes = encode_mp4.remote(png_map, fps, movie_name, run_name)

        # Save locally
        if local_dir:
            out_path = os.path.join(local_dir, f'{movie_name}.mp4')
        else:
            os.makedirs('output', exist_ok=True)
            out_path = f'output/{movie_name}.mp4'

        with open(out_path, 'wb') as f:
            f.write(mp4_bytes)
        print(f"  Saved: {out_path} ({len(mp4_bytes)/1e6:.1f} MB)")

    print(f"\nDone!")
