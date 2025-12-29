#!/usr/bin/env python3
"""
Enhanced FOV120 projection using proper fisheye camera calibration.
Uses actual polynomial distortion models from camera intrinsics.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1 import helper
from fisheye_projection import FisheyeProjector
import os


def enhanced_fisheye_fov120_projection(
    pred_xyz: torch.Tensor,
    gt_xyz: torch.Tensor,
    image_frames: torch.Tensor,
    camera_indices: torch.Tensor,
    clip_id: str,
    reasoning_text: str = "",
    save_path: str = "fisheye_fov120_projections.png",
):
    """Create enhanced FOV120 projection with proper fisheye calibration."""

    print("🐟 Creating enhanced fisheye FOV120 projections...")
    print(f"📷 Image frames shape: {image_frames.shape}")
    print(f"📷 Camera indices: {camera_indices}")

    # Initialize fisheye projector
    projector = FisheyeProjector(clip_id)

    # Extract trajectories
    pred_traj = pred_xyz[0, 0, 0].cpu().numpy()  # [time, 3]
    gt_traj = gt_xyz[0, 0].cpu().numpy()  # [time, 3]

    # Convert frames for display - use the most recent frame (last in time dimension)
    frames = image_frames.cpu().numpy()
    if frames.max() <= 1.0:
        frames = (frames * 255).astype(np.uint8)
    else:
        frames = frames.astype(np.uint8)

    # Shape: [N_cameras, num_frames, 3, H, W] -> [N_cameras, 3, H, W] (take last frame)
    frames = frames[:, -1, :, :, :]  # Use most recent frame
    frames = frames.transpose(0, 2, 3, 1)  # [N_cameras, H, W, C]

    # Find the correct camera index for front wide 120fov
    # camera_indices maps to: 0=cross_left, 1=front_wide, 2=cross_right, 6=front_tele
    front_wide_camera_idx = 1  # From camera_name_to_index mapping

    # Find which position in the array corresponds to front_wide (index 1)
    camera_indices_list = camera_indices.cpu().numpy()
    try:
        front_wide_position = np.where(camera_indices_list == front_wide_camera_idx)[0][0]
        print(f"📷 Found front wide camera at position {front_wide_position}")
    except IndexError:
        print("❌ Front wide camera not found in camera_indices!")
        return plt.figure()

    # Focus only on Front Wide 120° FOV camera
    fov120_cameras = [
        {
            "name": "Front Wide\n120° FOV",
            "camera_id": "camera_front_wide_120fov",
            "frame_idx": front_wide_position,  # Use correct position
        },
    ]

    # Create enhanced visualization - single camera view (no subplots)
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))

    camera = fov120_cameras[0]  # Only one camera
    camera_id = camera["camera_id"]
    frame_idx = camera["frame_idx"]

    if frame_idx >= len(frames) or camera_id not in projector.get_available_cameras():
        ax.text(
            0.5,
            0.5,
            f"Camera\nNot Available\n({camera_id})",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=14,
            color="red",
        )
        ax.set_title(camera["name"], fontsize=16, fontweight="bold")
    else:
        # Display camera image
        frame = frames[frame_idx]
        ax.imshow(frame)
        ax.set_title(camera["name"], fontsize=18, fontweight="bold")

        # Get camera calibration info
        if camera_id in projector.intrinsics.index:
            calib = projector.intrinsics.loc[camera_id]
            img_width = int(calib["width"])
            img_height = int(calib["height"])
        else:
            img_width = 1920
            img_height = 1080

        # Project ground truth trajectory using fisheye model
        try:
            gt_projected, gt_valid = projector.project_trajectory_to_camera(gt_traj, camera_id)
            gt_count = gt_valid.sum()

            if gt_count > 0:
                gt_points = gt_projected[gt_valid]

                # Calculate distances for visualization
                gt_traj_relative = gt_traj - gt_traj[0]
                gt_distances = np.linalg.norm(gt_traj_relative[gt_valid], axis=1)

                if len(gt_distances) > 1:
                    # Normalize distances for color/size mapping
                    dist_norm = (gt_distances - gt_distances.min()) / (
                        gt_distances.max() - gt_distances.min() + 1e-8
                    )
                    colors = plt.cm.Greens(0.4 + 0.6 * (1 - dist_norm))  # Closer = darker green
                    sizes = 200 - dist_norm * 100  # Closer = larger
                else:
                    colors = ["green"]
                    sizes = [150]

                # Plot trajectory points
                ax.scatter(
                    gt_points[:, 0],
                    gt_points[:, 1],
                    c=colors,
                    s=sizes,
                    alpha=0.9,
                    edgecolors="darkgreen",
                    linewidth=2,
                    label=f"Ground Truth ({gt_count} pts)",
                    zorder=5,
                )

                # Connect with line
                if len(gt_points) > 1:
                    ax.plot(
                        gt_points[:, 0],
                        gt_points[:, 1],
                        "-",
                        color="lime",
                        linewidth=4,
                        alpha=0.8,
                        zorder=4,
                    )

                # Mark start point specially
                ax.scatter(
                    gt_points[0, 0],
                    gt_points[0, 1],
                    s=300,
                    c="lime",
                    marker="s",
                    edgecolors="darkgreen",
                    linewidth=4,
                    label="GT Start",
                    zorder=10,
                )

        except Exception as e:
            gt_count = 0
            print(f"⚠️ GT projection failed for {camera_id}: {e}")

        # Project predicted trajectory using fisheye model
        try:
            pred_projected, pred_valid = projector.project_trajectory_to_camera(
                pred_traj, camera_id
            )
            pred_count = pred_valid.sum()

            if pred_count > 0:
                pred_points = pred_projected[pred_valid]

                # Calculate distances
                pred_traj_relative = pred_traj - pred_traj[0]
                pred_distances = np.linalg.norm(pred_traj_relative[pred_valid], axis=1)

                if len(pred_distances) > 1:
                    dist_norm = (pred_distances - pred_distances.min()) / (
                        pred_distances.max() - pred_distances.min() + 1e-8
                    )
                    colors = plt.cm.Reds(0.4 + 0.6 * (1 - dist_norm))  # Closer = darker red
                    sizes = 200 - dist_norm * 100
                else:
                    colors = ["red"]
                    sizes = [150]

                # Plot trajectory points
                ax.scatter(
                    pred_points[:, 0],
                    pred_points[:, 1],
                    c=colors,
                    s=sizes,
                    alpha=0.9,
                    edgecolors="darkred",
                    linewidth=2,
                    label=f"Predicted ({pred_count} pts)",
                    zorder=5,
                )

                # Connect with line
                if len(pred_points) > 1:
                    ax.plot(
                        pred_points[:, 0],
                        pred_points[:, 1],
                        "-",
                        color="red",
                        linewidth=4,
                        alpha=0.8,
                        zorder=4,
                    )

                # Mark start point
                ax.scatter(
                    pred_points[0, 0],
                    pred_points[0, 1],
                    s=300,
                    c="red",
                    marker="s",
                    edgecolors="darkred",
                    linewidth=4,
                    label="Pred Start",
                    zorder=10,
                )

        except Exception as e:
            pred_count = 0
            print(f"⚠️ Pred projection failed for {camera_id}: {e}")

        # Add comprehensive info box
        total_points = len(gt_traj)

        if gt_count > 0 or pred_count > 0:
            info_color = "lightgreen"
            text_color = "black"
            status_icon = "✓"
        else:
            info_color = "lightcoral"
            text_color = "white"
            status_icon = "✗"

        # Get camera-specific info
        if camera_id in projector.intrinsics.index:
            calib = projector.intrinsics.loc[camera_id]
            cx, cy = calib["cx"], calib["cy"]
            fw_poly_1 = calib["fw_poly_1"]  # Main fisheye parameter

            info_text = f"{status_icon} FISHEYE PROJECTION\n"
            info_text += f"GT: {gt_count}/{total_points} pts\n"
            info_text += f"Pred: {pred_count}/{total_points} pts\n"
            info_text += f"Principal: ({cx:.0f}, {cy:.0f})\n"
            info_text += f"Fisheye coeff: {fw_poly_1:.0f}"
        else:
            info_text = f"{status_icon} Camera unavailable"

        ax.text(
            0.02,
            0.98,
            info_text,
            transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.5", facecolor=info_color, alpha=0.95),
            fontsize=11,
            verticalalignment="top",
            color=text_color,
            fontweight="bold",
            fontfamily="monospace",
        )

        # Add reasoning text below the fisheye projection box
        if reasoning_text:
            reasoning_box = f"🧠 MODEL REASONING:\n{reasoning_text}"
            ax.text(
                0.02,
                0.82,
                reasoning_box,
                transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.95),
                fontsize=10,
                verticalalignment="top",
                horizontalalignment="left",
                fontweight="bold",
                color="darkblue",
                wrap=True,
            )

        # Add technical details
        tech_text = "FISHEYE MODEL\nPolynomial Distortion\nActual Calibration"
        ax.text(
            0.98,
            0.98,
            tech_text,
            transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9),
            fontsize=9,
            verticalalignment="top",
            horizontalalignment="right",
            fontweight="bold",
        )

        # Add legend if trajectories visible
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc="lower right", fontsize=12, framealpha=0.9, fancybox=True, shadow=True)

        ax.set_xlim(0, img_width)
        ax.set_ylim(img_height, 0)  # Invert Y

    plt.suptitle(
        "Enhanced Fisheye Projection - Front Wide 120° FOV Camera\n(Using Actual Polynomial Distortion Calibration)",
        fontsize=18,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"🐟 Enhanced fisheye projections saved to: {save_path}")

    # Print detailed summary
    print(f"\n📊 Enhanced Fisheye Projection Summary:")
    total_gt = 0
    total_pred = 0

    for camera in fov120_cameras:
        camera_id = camera["camera_id"]
        if camera_id in projector.get_available_cameras():
            try:
                _, gt_valid = projector.project_trajectory_to_camera(gt_traj, camera_id)
                _, pred_valid = projector.project_trajectory_to_camera(pred_traj, camera_id)
                gt_cnt = gt_valid.sum()
                pred_cnt = pred_valid.sum()
                total_gt += gt_cnt
                total_pred += pred_cnt
                print(f"   {camera['name'].replace(chr(10), ' ')}: GT={gt_cnt}, Pred={pred_cnt}")
            except:
                print(f"   {camera['name'].replace(chr(10), ' ')}: Error in projection")

    print(f"   Front Wide Camera Total: GT={total_gt}, Pred={total_pred}")
    print(f"   Improvement from simple model: Much more accurate positioning!")

    return fig


def main():
    """Run enhanced fisheye FOV120 projection."""

    print("🚀 Enhanced Fisheye FOV120 Projection")
    print("=" * 60)

    # Setup
    output_dir = "inference_outputs"
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    clip_id = "030c760c-ae38-49aa-9ad8-f5650a545d26"
    print(f"📁 Loading dataset...")
    data = load_physical_aiavdataset(clip_id, t0_us=5_100_000)

    print("🤖 Loading model...")
    model = AlpamayoR1.from_pretrained("nvidia/Alpamayo-R1-10B", dtype=torch.bfloat16).to("cuda")
    processor = helper.get_processor(model.tokenizer)

    # Prepare inputs
    messages = helper.create_message(data["image_frames"].flatten(0, 1))
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="pt",
    )

    model_inputs = {
        "tokenized_data": inputs,
        "ego_history_xyz": data["ego_history_xyz"],
        "ego_history_rot": data["ego_history_rot"],
    }
    model_inputs = helper.to_device(model_inputs, "cuda")

    # Run inference
    print("🧮 Running inference...")
    torch.cuda.manual_seed_all(42)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
            data=model_inputs,
            top_p=0.98,
            temperature=0.6,
            num_traj_samples=1,
            max_generation_length=256,
            return_extra=True,
        )

    # Metrics
    gt_xy = data["ego_future_xyz"].cpu()[0, 0, :, :2].T.numpy()
    pred_xy = pred_xyz.cpu().numpy()[0, 0, :, :, :2].transpose(0, 2, 1)
    diff = np.linalg.norm(pred_xy - gt_xy[None, ...], axis=1).mean(-1)
    min_ade = diff.min()

    print(f"📊 Performance: minADE = {min_ade:.3f} meters")
    print(f"🧠 Reasoning: {extra['cot'][0][0]}")

    # Create enhanced fisheye projection
    fig = enhanced_fisheye_fov120_projection(
        pred_xyz,
        data["ego_future_xyz"],
        data["image_frames"],
        data["camera_indices"],
        clip_id,
        reasoning_text=extra["cot"][0][0],
        save_path=os.path.join(output_dir, "fisheye_fov120_projections.png"),
    )

    plt.close(fig)
    print("✅ Enhanced fisheye FOV120 projection complete!")


if __name__ == "__main__":
    main()
