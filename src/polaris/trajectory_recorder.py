"""
Trajectory recorder for Polaris DROID environments.

Records trajectories in HDF5 format for conversion to LeRobot dataset.
Adapted from SimEval DROID recorder with Polaris-specific observation structure.

Multi-environment support:
    Each env_id writes to its own HDF5 file to avoid locking and enable
    fully parallel recording. Files are saved in the same output directory
    with naming: trajectory_{timestamp}_env{env_id}.h5

Streaming image writes:
    Camera images are written to HDF5 incrementally during record_step(),
    NOT accumulated in memory. This avoids:
    - OOM on long episodes (492 steps × 720×1280×3 × 2 cameras ≈ 2.7 GB)
    - Blocking np.stack + compression at end_episode()
    - Blocking the asyncio event loop in the async pipeline

    Non-image data (joint_pos, ee_pose, etc.) is tiny and still accumulated
    in lists, then bulk-written at end_episode().

Data fields recorded (for GS splat post-rendering):
    observations/
        external_cam: (T, 720, 1280, 3) uint8  -- sim camera RGB
        wrist_cam:    (T, 720, 1280, 3) uint8  -- sim camera RGB
        joint_position:    (T, 7)  float32
        joint_velocity:    (T, 7)  float32
        gripper_position:  (T, 1)  float32
        gripper_velocity:  (T, 1)  float32
        ee_pose:           (T, 7)  float32  -- xyz + wxyz quaternion
        ee_velocity:       (T, 6)  float32  -- linear + angular
        timestamp:         (T,)    float64
    actions/
        joint_position_command: (T, 7) float32
        gripper_command:        (T, 1) float32
    metadata/
        instruction, task_id, env_id, fps, episode_length,
        success, progress
"""

import h5py
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from typing import Optional


class PolarisTrajectoryRecorder:
    """Records trajectories from Polaris environment to HDF5 format.

    Supports multi-environment recording: each env_id writes to its own
    HDF5 file, enabling lock-free parallel recording without contention.

    Image data is written incrementally to avoid memory issues and blocking.
    """

    # Camera image dimensions (must match Isaac Lab camera config)
    _CAM_H = 720
    _CAM_W = 1280
    _CAM_C = 3
    _IMAGE_KEYS = ("external_cam", "wrist_cam")
    _CAM_EXTRINSIC_KEYS = ("external_cam_pos", "external_cam_rot",
                           "wrist_cam_pos", "wrist_cam_rot")
    _CAM_INTRINSICS_KEY = "_cam_intrinsics"

    def __init__(self, output_dir: str):
        """Initialize trajectory recorder.

        Args:
            output_dir: Directory to save trajectory files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Shared timestamp for this session
        self._timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Per-env HDF5 file paths (created lazily on first start_episode)
        self._filepaths: dict[int, Path] = {}

        # Per-env episode counters (each env has independent numbering)
        self._episode_counts: dict[int, int] = {}

        # Per-env recording state
        self._is_recording: dict[int, bool] = {}
        self._current_episode: dict[int, Optional[int]] = {}
        self._start_time: dict[int, Optional[float]] = {}

        # Per-env non-image episode data (accumulated in lists, small)
        self._episode_data: dict[int, dict] = {}

        # Per-env open HDF5 file handles and image dataset refs
        self._h5_files: dict[int, h5py.File] = {}
        self._h5_img_datasets: dict[int, dict[str, h5py.Dataset]] = {}
        self._h5_step_counts: dict[int, int] = {}

    @property
    def filepath(self) -> Path:
        """Primary filepath (env_0) for backward compatibility."""
        return self._get_filepath(0)

    @property
    def episode_count(self) -> int:
        """Total episodes across all envs."""
        return sum(self._episode_counts.values())

    def _get_filepath(self, env_id: int) -> Path:
        """Get or create the HDF5 filepath for a specific env_id."""
        if env_id not in self._filepaths:
            self._filepaths[env_id] = (
                self.output_dir / f"trajectory_{self._timestamp}_env{env_id}.h5"
            )
        return self._filepaths[env_id]

    def is_recording(self, env_id: int = 0) -> bool:
        """Check if a specific environment is currently recording."""
        return self._is_recording.get(env_id, False)

    def start_episode(self, instruction: str, task_id: str, env_id: int = 0):
        """Start recording a new episode for a specific environment.

        Opens (or creates) the HDF5 file and pre-allocates expandable
        image datasets so that record_step() can write frames incrementally.
        """
        if self._is_recording.get(env_id, False):
            raise RuntimeError(
                f"Episode already in progress for env {env_id}. "
                f"Call end_episode() or discard_episode() first."
            )

        # Assign env-local episode number
        if env_id not in self._episode_counts:
            self._episode_counts[env_id] = 0
        assigned_episode = self._episode_counts[env_id]

        self._is_recording[env_id] = True
        self._start_time[env_id] = datetime.now().timestamp()
        self._current_episode[env_id] = assigned_episode
        self._h5_step_counts[env_id] = 0

        # Initialize non-image episode data (accumulated in lists)
        self._episode_data[env_id] = {
            "observations": {
                "joint_position": [],
                "joint_velocity": [],
                "gripper_position": [],
                "gripper_velocity": [],
                "ee_pose": [],
                "ee_velocity": [],
                "timestamp": [],
                # GS splat 后渲染所需数据
                "object_pose": [],          # (7,) xyz + wxyz per frame
                "external_cam_pos": [],     # (3,) camera world position
                "external_cam_rot": [],     # (3,3) camera rotation matrix
                "wrist_cam_pos": [],        # (3,)
                "wrist_cam_rot": [],        # (3,3)
            },
            "actions": {
                "joint_position_command": [],
                "gripper_command": [],
            },
            "metadata": {
                "instruction": instruction,
                "task_id": task_id,
                "env_id": env_id,
                "fps": 15.0,
            }
        }

        # Open HDF5 file and create expandable image datasets
        filepath = self._get_filepath(env_id)
        h5f = h5py.File(filepath, "a")
        ep_group = h5f.create_group(f"episode_{assigned_episode}")
        obs_group = ep_group.create_group("observations")

        img_datasets = {}
        for cam_key in self._IMAGE_KEYS:
            ds = obs_group.create_dataset(
                cam_key,
                shape=(0, self._CAM_H, self._CAM_W, self._CAM_C),
                maxshape=(None, self._CAM_H, self._CAM_W, self._CAM_C),
                dtype=np.uint8,
                chunks=(1, self._CAM_H, self._CAM_W, self._CAM_C),
                compression="lzf",
            )
            img_datasets[cam_key] = ds

        self._h5_files[env_id] = h5f
        self._h5_img_datasets[env_id] = img_datasets

        print(
            f"[Recorder] Started episode {assigned_episode} (env {env_id}): "
            f"'{instruction}' ({task_id}) -> {filepath.name}"
        )

    def record_step(self, env_id: int, obs: dict, action: np.ndarray):
        """Record a single timestep for a specific environment.

        Image data is written directly to HDF5 (streaming).
        Non-image data is appended to in-memory lists.

        Args:
            env_id: Environment index
            obs: Observation dictionary from environment
                - obs["splat"]: Camera images (external_cam, wrist_cam)
                - obs["policy"]: Proprioceptive state
                - obs["splat_meta"] (optional): GS splat post-rendering data
                    - object_pose: (7,) xyz + wxyz quaternion
                    - external_cam_pos: (3,) camera world position
                    - external_cam_rot: (3,3) rotation matrix
                    - wrist_cam_pos: (3,)
                    - wrist_cam_rot: (3,3)
            action: Action array (8D: 7 joint commands + 1 gripper)
        """
        if not self._is_recording.get(env_id, False):
            raise RuntimeError(
                f"No episode in progress for env {env_id}. "
                f"Call start_episode() first."
            )

        episode_data = self._episode_data[env_id]
        step_idx = self._h5_step_counts[env_id]
        img_datasets = self._h5_img_datasets[env_id]

        # ── Stream camera images directly to HDF5 ──
        splat_obs = obs["splat"]
        for cam_key in self._IMAGE_KEYS:
            img = self._to_numpy(splat_obs[cam_key])  # (H, W, 3) uint8
            # Ensure correct shape
            if img.ndim == 3 and img.shape == (self._CAM_H, self._CAM_W, self._CAM_C):
                ds = img_datasets[cam_key]
                ds.resize(step_idx + 1, axis=0)
                ds[step_idx] = img
            elif img.ndim == 3:
                # Unexpected shape — try to handle gracefully
                ds = img_datasets[cam_key]
                ds.resize(step_idx + 1, axis=0)
                # Crop or pad to expected shape
                h = min(img.shape[0], self._CAM_H)
                w = min(img.shape[1], self._CAM_W)
                c = min(img.shape[2], self._CAM_C)
                ds[step_idx, :h, :w, :c] = img[:h, :w, :c]

        self._h5_step_counts[env_id] = step_idx + 1

        # ── Accumulate non-image data in lists (small) ──
        policy_obs = obs["policy"]
        joint_pos = self._to_numpy(policy_obs["arm_joint_pos"])
        joint_vel = self._to_numpy(policy_obs["arm_joint_vel"])
        gripper_pos = self._to_numpy(policy_obs["gripper_pos"])
        gripper_vel = self._to_numpy(policy_obs["gripper_vel"])
        ee_pose = self._to_numpy(policy_obs["ee_pose"])
        ee_vel = self._to_numpy(policy_obs["ee_vel"])
        timestamp = datetime.now().timestamp() - self._start_time[env_id]

        episode_data["observations"]["joint_position"].append(joint_pos)
        episode_data["observations"]["joint_velocity"].append(joint_vel)
        episode_data["observations"]["gripper_position"].append(gripper_pos)
        episode_data["observations"]["gripper_velocity"].append(gripper_vel)
        episode_data["observations"]["ee_pose"].append(ee_pose)
        episode_data["observations"]["ee_velocity"].append(ee_vel)
        episode_data["observations"]["timestamp"].append(timestamp)

        # GS splat 后渲染元数据（object pose + camera extrinsics）
        splat_meta = obs.get("splat_meta", {})
        if splat_meta:
            for meta_key in ("object_pose",
                             "external_cam_pos", "external_cam_rot",
                             "wrist_cam_pos", "wrist_cam_rot"):
                if meta_key in splat_meta:
                    episode_data["observations"][meta_key].append(
                        self._to_numpy(splat_meta[meta_key])
                    )

            # 相机内参（仅首帧保存一次，存入 episode metadata）
            if step_idx == 0 and self._CAM_INTRINSICS_KEY in splat_meta:
                cam_intrinsics = splat_meta[self._CAM_INTRINSICS_KEY]
                episode_data["metadata"]["_cam_intrinsics"] = {
                    name: self._to_numpy(K) for name, K in cam_intrinsics.items()
                }

        action = np.asarray(action, dtype=np.float32)
        episode_data["actions"]["joint_position_command"].append(action[:7])
        episode_data["actions"]["gripper_command"].append(action[7:8])

    def discard_episode(self, env_id: int = 0):
        """Discard the current episode without saving."""
        if not self._is_recording.get(env_id, False):
            return

        discarded_ep = self._current_episode.get(env_id)

        # Remove the episode group from HDF5 and close file
        h5f = self._h5_files.pop(env_id, None)
        if h5f is not None:
            ep_name = f"episode_{discarded_ep}"
            if ep_name in h5f:
                del h5f[ep_name]
            h5f.close()

        self._h5_img_datasets.pop(env_id, None)
        self._h5_step_counts.pop(env_id, None)

        print(f"[Recorder] Discarded episode {discarded_ep} (env {env_id})")

        self._is_recording[env_id] = False
        self._current_episode[env_id] = None
        self._start_time[env_id] = None
        self._episode_data.pop(env_id, None)

    def end_episode(self, rubric_result: Optional[dict] = None, env_id: int = 0) -> bool:
        """End the current episode and finalize HDF5 data.

        Image data has already been written incrementally during record_step().
        This method only writes the (small) non-image data and metadata,
        then closes the file — typically completing in < 1 second.
        """
        if not self._is_recording.get(env_id, False):
            print(f"[Recorder] Warning: No episode in progress for env {env_id}.")
            return False

        import time as _time
        t0 = _time.time()

        episode_data = self._episode_data[env_id]
        episode_number = self._current_episode[env_id]
        episode_length = self._h5_step_counts.get(env_id, 0)

        if episode_length == 0:
            print(
                f"[Recorder] Warning: Episode {episode_number} (env {env_id}) "
                f"has no data. Skipping."
            )
            self._cleanup_env_state(env_id)
            return False

        print(
            f"[Recorder] Ending episode {episode_number} "
            f"(env {env_id}, {episode_length} steps)"
        )

        h5f = self._h5_files.get(env_id)
        if h5f is None:
            print(f"[Recorder] Error: HDF5 file not open for env {env_id}")
            self._cleanup_env_state(env_id)
            return False

        ep_group = h5f[f"episode_{episode_number}"]
        obs_group = ep_group["observations"]

        # ── Write non-image observations (small, fast) ──
        for key, arr_list in episode_data["observations"].items():
            if len(arr_list) > 0:
                stacked = np.stack(arr_list, axis=0)
                obs_group.create_dataset(key, data=stacked, compression="gzip")

        # ── Write actions (small, fast) ──
        action_group = ep_group.create_group("actions")
        for key, arr_list in episode_data["actions"].items():
            if len(arr_list) > 0:
                stacked = np.stack(arr_list, axis=0)
                action_group.create_dataset(key, data=stacked, compression="gzip")

        # ── Write metadata ──
        meta_group = ep_group.create_group("metadata")
        episode_data["metadata"]["episode_length"] = episode_length
        if rubric_result:
            episode_data["metadata"]["success"] = rubric_result.get("success", False)
            episode_data["metadata"]["progress"] = rubric_result.get("progress", 0.0)

        # 相机内参作为 dataset 存储（numpy array 不能存为 HDF5 attr）
        cam_intrinsics = episode_data["metadata"].pop("_cam_intrinsics", None)
        if cam_intrinsics:
            intrinsics_group = meta_group.create_group("camera_intrinsics")
            for cam_name, K in cam_intrinsics.items():
                intrinsics_group.create_dataset(cam_name, data=K)

        for key, value in episode_data["metadata"].items():
            meta_group.attrs[key] = value

        # ── Check if image datasets have real data ──
        for cam_key in self._IMAGE_KEYS:
            if cam_key in obs_group:
                ds = obs_group[cam_key]
                if ds.shape[0] > 0:
                    # Quick check: is the first frame all-zero? (placeholder)
                    first_frame = ds[0]
                    if first_frame.max() == 0:
                        print(
                            f"[Recorder] 警告: {cam_key} 首帧全零（可能是占位符），"
                            f"但已写入 {ds.shape[0]} 帧"
                        )

        # ── Close file ──
        h5f.close()

        elapsed = _time.time() - t0
        filepath = self._get_filepath(env_id)
        print(
            f"[Recorder] HDF5 写入完成: episode_{episode_number} -> "
            f"{filepath.name} ({elapsed:.1f}s, {episode_length} steps)"
        )

        # Increment counter and cleanup
        self._episode_counts[env_id] = episode_number + 1
        self._cleanup_env_state(env_id)
        return True

    def _cleanup_env_state(self, env_id: int):
        """Reset per-env state after episode ends or is discarded."""
        self._is_recording[env_id] = False
        self._current_episode[env_id] = None
        self._start_time[env_id] = None
        self._episode_data.pop(env_id, None)
        self._h5_files.pop(env_id, None)
        self._h5_img_datasets.pop(env_id, None)
        self._h5_step_counts.pop(env_id, None)

    def save(self):
        """Finalize and close all HDF5 files."""
        active_envs = [
            eid for eid, recording in self._is_recording.items() if recording
        ]
        for env_id in active_envs:
            print(
                f"[Recorder] Warning: Episode still in progress for env {env_id}. "
                f"Ending it."
            )
            self.end_episode(env_id=env_id)

        # Close any remaining open files
        for env_id, h5f in list(self._h5_files.items()):
            try:
                h5f.close()
            except Exception:
                pass
        self._h5_files.clear()

        # Print summary per env
        total = 0
        for env_id, count in sorted(self._episode_counts.items()):
            filepath = self._filepaths.get(env_id)
            print(f"[Recorder] env {env_id}: {count} episodes -> {filepath}")
            total += count
        print(f"[Recorder] Total: {total} episodes in {self.output_dir}")

    def _to_numpy(self, tensor):
        """Convert torch tensor to numpy array, handling batch dimension."""
        if isinstance(tensor, torch.Tensor):
            arr = tensor.cpu().numpy()
            if arr.shape[0] == 1:
                arr = arr[0]
            return arr
        return np.asarray(tensor)
