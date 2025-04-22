# utils.py

import os
import numpy as np
import base64
import re

def extract_field(content, fieldname, is_position=False):
    """Extracts data from VTU file content based on field name."""
    pattern = f'<DataArray[^>]*Name="{fieldname}"[^>]*>([^<]*)</DataArray>'
    match = re.search(pattern, content)
    if not match:
        print(f"Warning: Field {fieldname} not found.")
        return np.array([])

    try:
        encoded = match.group(1).strip()
        decoded = base64.b64decode(encoded)
        decoded = np.frombuffer(decoded[4:], dtype=np.float64)  # Skip 4-byte header
        if is_position:
            return decoded.reshape(-1, 3)
        else:
            return decoded
    except Exception as e:
        print(f"Error decoding {fieldname}: {e}")
        return np.array([])

def read_stefan_vtu_snapshot(snapshot_dir, num_procs=5):
    """Reads VTU snapshot split across multiple processors and extracts all fields."""
    all_points = []
    all_Tl = []
    all_Ts = []
    all_phi = []
    all_vx = []
    all_vy = []

    for i in range(num_procs):
        vtu_file = os.path.join(snapshot_dir, f'{i:04d}.vtu')
        if not os.path.isfile(vtu_file):
            print(f"Missing file: {vtu_file}")
            continue

        with open(vtu_file, 'rb') as f:
            content = f.read().decode('utf-8', errors='ignore')

        pts = extract_field(content, 'Position', is_position=True)
        Tl = extract_field(content, 'Tl')
        Ts = extract_field(content, 'Ts')
        phi = extract_field(content, 'phi')
        vx = extract_field(content, 'v_int_x')
        vy = extract_field(content, 'v_int_y')

        all_points.append(pts)
        all_Tl.append(Tl)
        all_Ts.append(Ts)
        all_phi.append(phi)
        all_vx.append(vx)
        all_vy.append(vy)

    if not all_points:
        raise ValueError("No valid VTU files or fields found.")

    return (
        np.vstack(all_points),
        np.concatenate(all_Tl),
        np.concatenate(all_Ts),
        np.concatenate(all_phi),
        np.concatenate(all_vx),
        np.concatenate(all_vy),
    )
