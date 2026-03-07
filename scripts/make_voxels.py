"""
Make voxels from a mesh, for input to OmniPart.
"""

import open3d as o3d
import numpy as np
import argparse
import os
from rich.progress import track


def main():
    parser = argparse.ArgumentParser(description="Make voxels from a mesh.")
    parser.add_argument("input_dir", type=str, help="Path to the input directory.")
    parser.add_argument("output_dir", type=str, help="Path to the output directory.")
    args = parser.parse_args()

    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    for file in track(os.listdir(input_dir), description="Making voxels"):
        if file.endswith(".glb"):
            input_path = os.path.join(input_dir, file)
            output_path = os.path.join(output_dir, file.replace(".glb", ".npy"))
            make_voxels(input_path, output_path)


def make_voxels(input_path: str, output_path: str):
    mesh = o3d.io.read_triangle_mesh(input_path)
    # clamp vertices to the range [-0.5, 0.5]
    vertices = np.clip(np.asarray(mesh.vertices), -0.5 + 1e-6, 0.5 - 1e-6)
    assert len(vertices) > 0, "Error loading mesh, no vertices found"
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(
        mesh, voxel_size=1 / 64, min_bound=(-0.5, -0.5, -0.5), max_bound=(0.5, 0.5, 0.5)
    )
    vertices = np.array([voxel.grid_index for voxel in voxel_grid.get_voxels()])
    # flip and swap y and z axes to match the voxelization in OmniPart
    vertices[:, 2] = 63 - vertices[:, 2]
    vertices[:, [1, 2]] = vertices[:, [2, 1]]
    vertices = np.pad(vertices, ((0, 0), (1, 0)))
    assert np.all(vertices >= 0) and np.all(
        vertices < 64
    ), "Some vertices are out of bounds"
    np.save(output_path, vertices)


if __name__ == "__main__":
    main()
