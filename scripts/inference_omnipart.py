import os
import itertools
import numpy as np
import open3d as o3d
import torch
import argparse
from PIL import Image
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation as R_sci

from modules.bbox_gen.models.autogressive_bbox_gen import BboxGen
from modules.part_synthesis.process_utils import save_parts_outputs
from modules.inference_utils import load_img_mask, prepare_bbox_gen_input, prepare_part_synthesis_input, gen_mesh_from_bounds, vis_voxel_coords, merge_parts
from modules.part_synthesis.pipelines import OmniPartImageTo3DPipeline

from huggingface_hub import hf_hub_download

def main() -> None:
    device = "cuda"

    parser = argparse.ArgumentParser()
    parser.add_argument("--image_input", type=str, required=True)
    parser.add_argument("--mask_input", type=str, required=True)
    parser.add_argument("--bbox_input", type=str)
    parser.add_argument("--voxel_input", type=str)
    parser.add_argument("--align_bbox_input", action="store_true")
    parser.add_argument("--output_root", type=str, default="./output")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_inference_steps", type=int, default=25)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--simplify_ratio", type=float, default=0.3)
    parser.add_argument("--partfield_encoder_path", type=str, default="ckpt/model_objaverse.ckpt")
    parser.add_argument("--bbox_gen_ckpt", type=str, default="ckpt/bbox_gen.ckpt")
    parser.add_argument("--part_synthesis_ckpt", type=str, default="omnipart/OmniPart")
    args = parser.parse_args()

    if not os.path.exists(args.partfield_encoder_path):
        args.partfield_encoder_path = hf_hub_download(repo_id="omnipart/OmniPart_modules", filename="partfield_encoder.ckpt", local_dir="ckpt")
    if not os.path.exists(args.bbox_gen_ckpt):
        args.bbox_gen_ckpt = hf_hub_download(repo_id="omnipart/OmniPart_modules", filename="bbox_gen.ckpt", local_dir="ckpt")

    os.makedirs(args.output_root, exist_ok=True)
    torch.manual_seed(args.seed)

    # load part_synthesis model
    part_synthesis_pipeline = OmniPartImageTo3DPipeline.from_pretrained(args.part_synthesis_ckpt)
    part_synthesis_pipeline.to(device)
    print("[INFO] PartSynthesis model loaded")

    # load bbox_gen model
    bbox_gen_config = OmegaConf.load("configs/bbox_gen.yaml").model.args
    bbox_gen_config.partfield_encoder_path = args.partfield_encoder_path
    bbox_gen_model = BboxGen(bbox_gen_config)
    bbox_gen_model.load_state_dict(torch.load(args.bbox_gen_ckpt), strict=False)
    bbox_gen_model.to(device)
    bbox_gen_model.eval().half()
    print("[INFO] BboxGen model loaded")

    if args.image_input.endswith('.txt'):
        with open(args.image_input) as f:
            image_inputs = f.read().splitlines()
    else:
        image_inputs = [args.image_input]

    if args.mask_input.endswith('.txt'):
        with open(args.mask_input) as f:
            mask_inputs = f.read().splitlines()
    else:
        mask_inputs = [args.mask_input]

    if len(image_inputs) != len(mask_inputs):
        raise ValueError("The number of image inputs and mask inputs must be the same.")

    if args.bbox_input:
        if args.bbox_input.endswith('.txt'):
            with open(args.bbox_input) as f:
                bbox_inputs = f.read().splitlines()
        else:
            bbox_inputs = [args.bbox_input]
        if len(image_inputs) != len(bbox_inputs):
            raise ValueError("The number of image inputs and bbox inputs must be the same.")
    else:
        bbox_inputs = [None] * len(image_inputs)

    if args.voxel_input:
        if args.voxel_input.endswith('.txt'):
            with open(args.voxel_input) as f:
                voxel_inputs = f.read().splitlines()
        else:
            voxel_inputs = [args.voxel_input]
        if len(image_inputs) != len(voxel_inputs):
            raise ValueError("The number of image inputs and voxel inputs must be the same.")
    else:
        voxel_inputs = [None] * len(image_inputs)

    for image_input, mask_input, bbox_input, voxel_input in zip(image_inputs, mask_inputs, bbox_inputs, voxel_inputs):
        infer(image_input, mask_input, bbox_input, voxel_input, args, part_synthesis_pipeline, bbox_gen_model)


def infer(
        image_input: str, mask_input: str, bbox_input: str | None, voxel_input: str | None, args,
        part_synthesis_pipeline: OmniPartImageTo3DPipeline,
        bbox_gen_model: BboxGen,
    ) -> None:
    output_dir = os.path.join(args.output_root, image_input.split("/")[-1].split(".")[0])
    os.makedirs(output_dir, exist_ok=True)

    img_white_bg, img_black_bg, ordered_mask_input, img_mask_vis = load_img_mask(image_input, mask_input)
    img_mask_vis.save(os.path.join(output_dir, "img_mask_vis.png"))

    if voxel_input is None:
        voxel_coords = part_synthesis_pipeline.get_coords(img_black_bg, num_samples=1, seed=args.seed, sparse_structure_sampler_params={"steps": 25, "cfg_strength": 7.5})
        voxel_coords = voxel_coords.cpu().numpy()
    else:
        voxel_coords = np.load(voxel_input)

    np.save(os.path.join(output_dir, "voxel_coords.npy"), voxel_coords)
    voxel_coords_ply = vis_voxel_coords(voxel_coords)
    voxel_coords_ply.export(os.path.join(output_dir, "voxel_coords_vis.ply"))
    print("[INFO] Voxel coordinates saved")

    if bbox_input is None:
        bbox_gen_input = prepare_bbox_gen_input(os.path.join(output_dir, "voxel_coords.npy"), img_white_bg, ordered_mask_input)
        bbox_gen_output = bbox_gen_model.generate(bbox_gen_input)
        bboxes = bbox_gen_output['bboxes'][0]
    else:
        bboxes = np.load(bbox_input)
        if args.align_bbox_input:
            bboxes = align_boxes_to_voxels(voxel_coords[...,1:], bboxes)

    np.save(os.path.join(output_dir, "bboxes.npy"), bboxes)
    bboxes_vis = gen_mesh_from_bounds(bboxes)
    bboxes_vis.export(os.path.join(output_dir, "bboxes_vis.glb"))
    print("[INFO] BboxGen output saved")

    part_synthesis_input = prepare_part_synthesis_input(os.path.join(output_dir, "voxel_coords.npy"), os.path.join(output_dir, "bboxes.npy"), ordered_mask_input)
    part_synthesis_output = part_synthesis_pipeline.get_slat(
        img_black_bg, 
        part_synthesis_input['coords'], 
        [part_synthesis_input['part_layouts']], 
        part_synthesis_input['masks'],
        seed=args.seed,
        slat_sampler_params={"steps": args.num_inference_steps, "cfg_strength": args.guidance_scale},
        formats=['mesh', 'gaussian', 'radiance_field'],
        preprocess_image=False,
    )
    save_parts_outputs(
        part_synthesis_output, 
        output_dir=output_dir, 
        simplify_ratio=args.simplify_ratio, 
        save_video=False,
        save_glb=True,
        textured=False,
    )
    merge_parts(output_dir)
    print("[INFO] PartSynthesis output saved")


def align_boxes_to_voxels(
    voxels: np.ndarray, boxes: np.ndarray, num_samples_per_box: int = 1000
) -> np.ndarray:
    """
    Aligns a set of axis-aligned 3D boxes to voxel coordinates.
    Assumes rotation differences are roughly 90 degrees.
    """
    voxel_pcd, voxel_coords = _voxels_to_pcd(voxels)
    box_pcd, box_coords = _boxes_to_pcd(boxes, num_samples_per_box)

    voxel_center, box_center, voxel_scale, scale_factor = _apply_coarse_alignment(
        voxel_pcd, voxel_coords, box_pcd, box_coords
    )

    T_icp = _run_icp(voxel_pcd, box_pcd, voxel_scale)
    R_snapped, t = _get_snapped_rotation(T_icp)

    aligned_boxes = _transform_boxes(
        boxes, box_center, scale_factor, R_snapped, t, voxel_center
    )

    return aligned_boxes


def _voxels_to_pcd(
    voxels: np.ndarray, grid_dim: float = 64.0
) -> tuple[o3d.geometry.PointCloud, np.ndarray]:
    """Converts a grid of voxel coordinates to a centered Open3D PointCloud."""
    # Add 0.5 to align points to the geometric center of each voxel
    if len(voxels.shape) != 2 or voxels.shape[1] != 3:
        raise ValueError(f"Expected voxels shape to be (N, 3), but got {voxels.shape}")

    # Scale to [-0.5, 0.5] box
    coords = (voxels.astype(np.float64) + 0.5) / grid_dim - 0.5
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords)
    return pcd, coords


def _boxes_to_pcd(
    boxes: np.ndarray, num_samples: int
) -> tuple[o3d.geometry.PointCloud, np.ndarray]:
    """Uniformly samples points within bounding box volumes."""
    box_points = [np.random.uniform(b[0], b[1], (num_samples, 3)) for b in boxes]
    coords = np.vstack(box_points)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords)
    return pcd, coords


def _apply_coarse_alignment(
    voxel_pcd: o3d.geometry.PointCloud,
    voxel_coords: np.ndarray,
    box_pcd: o3d.geometry.PointCloud,
    box_coords: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Translates point clouds to origin and normalizes their scale in-place."""
    voxel_center = np.mean(voxel_coords, axis=0)
    box_center = np.mean(box_coords, axis=0)

    voxel_pcd.translate(-voxel_center)
    box_pcd.translate(-box_center)

    voxel_scale = float(np.mean(np.linalg.norm(np.asarray(voxel_pcd.points), axis=1)))
    box_scale = float(np.mean(np.linalg.norm(np.asarray(box_pcd.points), axis=1)))

    scale_factor = voxel_scale / box_scale
    box_pcd.scale(scale_factor, center=(0, 0, 0))

    return voxel_center, box_center, voxel_scale, scale_factor


def _run_icp(
    voxel_pcd: o3d.geometry.PointCloud,
    box_pcd: o3d.geometry.PointCloud,
    base_scale: float,
) -> np.ndarray:
    """Executes Point-to-Plane ICP to find the fine transformation matrix."""
    search_param = o3d.geometry.KDTreeSearchParamHybrid(
        radius=base_scale * 0.5, max_nn=30
    )
    voxel_pcd.estimate_normals(search_param=search_param)
    box_pcd.estimate_normals(search_param=search_param)

    threshold = base_scale * 0.15
    reg_p2l = o3d.pipelines.registration.registration_icp(
        box_pcd,
        voxel_pcd,
        threshold,
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000),
    )
    return reg_p2l.transformation


def _get_snapped_rotation(
    transformation_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Extracts rotation and translation, snapping rotation to nearest 90-degree axis."""
    # Use np.copy() to detach from Open3D's read-only C++ memory
    R_continuous = np.copy(transformation_matrix[:3, :3])
    t = np.copy(transformation_matrix[:3, 3])

    euler_angles = R_sci.from_matrix(R_continuous).as_euler('xyz')
    snapped_euler = np.round(euler_angles / (np.pi / 2)) * (np.pi / 2)
    R_snapped = R_sci.from_euler('xyz', snapped_euler).as_matrix()

    return R_snapped, t


def _transform_boxes(
    boxes: np.ndarray,
    box_center: np.ndarray,
    scale: float,
    R: np.ndarray,
    t: np.ndarray,
    voxel_center: np.ndarray,
) -> np.ndarray:
    """Applies the full transformation pipeline to the 8 corners of each box."""
    M = boxes.shape[0]
    transformed = np.zeros((M, 2, 3), dtype=np.float64)

    for i, (min_b, max_b) in enumerate(boxes):
        corners = np.array(
            list(
                itertools.product(
                    [min_b[0], max_b[0]], [min_b[1], max_b[1]], [min_b[2], max_b[2]]
                )
            )
        )

        # Sequentially apply transforms: center -> scale -> rotate/translate -> position
        corners = corners - box_center
        corners = corners * scale
        corners = (R @ corners.T).T + t
        corners = corners + voxel_center

        transformed[i, 0] = np.min(corners, axis=0)
        transformed[i, 1] = np.max(corners, axis=0)

    return transformed

if __name__ == "__main__":
    main()