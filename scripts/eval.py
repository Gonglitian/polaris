import tyro
import mediapy

import tqdm
import gymnasium as gym
import torch
import numpy as np
import argparse
import pandas as pd


from pathlib import Path
from isaaclab.app import AppLauncher

from polaris.config import EvalArgs


def _load_start_poses(env, eval_args) -> tuple[str, list[dict]]:
    """加载 initial conditions（优先使用 StartPoseSampler，否则回退到 initial_conditions.json）

    Returns:
        (language_instruction, initial_conditions): 语言指令和初始条件列表
        每个 initial_condition 是 {obj_name: [x, y, z, qw, qx, qy, qz]} 格式的 dict
    """
    from my_regrasp.configs.polaris_tasks_cfg import extract_task_name_from_env_id, get_task_config

    task_name = extract_task_name_from_env_id(eval_args.environment)

    if task_name is not None:
        task_cfg = get_task_config(task_name)

        # 尝试使用 StartPoseSampler 生成多样化 start poses
        if (
            task_cfg.start_pose_sampler_cfg is not None
            and task_cfg.start_pose_sampler_cfg.enabled
            and task_cfg.placement_cfg is not None
            and task_cfg.placement_cfg.save_path
        ):
            try:
                from my_regrasp.components.start_pose_sampler import StartPoseSampler
                from my_regrasp.utils.transforms import pose_to_position_quat_np

                sampler = StartPoseSampler(task_cfg.start_pose_sampler_cfg)

                # 获取 initial_conditions.json 路径（用于读取原始 x, y 坐标）
                ic_path = Path(env.usd_file).parent / "initial_conditions.json"

                start_poses_4x4 = sampler.sample(
                    placement_cache_path=task_cfg.placement_cfg.save_path,
                    initial_conditions_path=str(ic_path),
                    target_object_name=task_cfg.primary_target,
                )

                if start_poses_4x4:
                    # 将 4x4 矩阵转换为 {obj_name: [x, y, z, qw, qx, qy, qz]} 格式
                    initial_conditions = []
                    for pose_4x4 in start_poses_4x4:
                        pos, quat = pose_to_position_quat_np(pose_4x4, quat_format="wxyz")
                        pose_7 = np.concatenate([pos, quat]).tolist()
                        initial_conditions.append({task_cfg.primary_target: pose_7})

                    # 限制 rollout 数量
                    if eval_args.rollouts is not None:
                        initial_conditions = initial_conditions[: eval_args.rollouts]

                    instruction = task_cfg.instruction
                    print(f"[eval] StartPoseSampler 生成 {len(initial_conditions)} 个 start poses")
                    return instruction, initial_conditions
                else:
                    print("[eval] StartPoseSampler 返回空列表，回退到 initial_conditions.json")

            except Exception as e:
                print(f"[eval] StartPoseSampler 失败: {e}，回退到 initial_conditions.json")

    # 回退：使用 initial_conditions.json
    from polaris.utils import load_eval_initial_conditions

    return load_eval_initial_conditions(
        usd=env.usd_file,
        initial_conditions_file=eval_args.initial_conditions_file,
        rollouts=eval_args.rollouts,
    )


def main(eval_args: EvalArgs):
    # This must be done before importing anything from IsaacLab
    # Inside main function to avoid launching IsaacLab in global scope
    # >>>> Isaac Sim App Launcher <<<<
    parser = argparse.ArgumentParser()
    args_cli, _ = parser.parse_known_args()
    args_cli.enable_cameras = True
    args_cli.headless = eval_args.headless
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app
    # >>>> Isaac Sim App Launcher <<<<

    # 注册 my_regrasp 环境（Polaris-Regrasp-{Task}-v0 等）
    import my_regrasp.envs  # noqa: F401, E402

    from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
    from polaris.policy import InferenceClient

    # 统一接口：接受任务名（PlayingCardsKitchen）或完整 env ID
    from my_regrasp.configs.polaris_tasks_cfg import resolve_env_id
    eval_args.environment = resolve_env_id(eval_args.environment)
    print(f"[eval] 环境 ID: {eval_args.environment}")

    env_cfg = parse_env_cfg(
        eval_args.environment,
        device="cuda",
        num_envs=1,
        use_fabric=True,
    )
    env = gym.make(eval_args.environment, cfg=env_cfg)

    language_instruction, initial_conditions = _load_start_poses(env, eval_args)
    rollouts = len(initial_conditions)

    # Resume CSV logging
    run_folder = Path(eval_args.run_folder)
    run_folder.mkdir(parents=True, exist_ok=True)
    csv_path = run_folder / "eval_results.csv"
    if csv_path.exists():
        episode_df = pd.read_csv(csv_path)
    else:
        episode_df = pd.DataFrame(
            {
                "episode": pd.Series(dtype="int"),
                "episode_length": pd.Series(dtype="int"),
                "success": pd.Series(dtype="bool"),
                "progress": pd.Series(dtype="float"),
            }
        )
    episode = len(episode_df)
    if episode >= rollouts:
        print("All rollouts have been evaluated. Exiting.")
        env.close()
        simulation_app.close()
        return

    policy_client: InferenceClient = InferenceClient.get_client(eval_args.policy)

    video = []
    horizon = env.max_episode_length
    bar = tqdm.tqdm(range(horizon))
    obs, info = env.reset(
        object_positions=initial_conditions[episode % len(initial_conditions)]
    )
    policy_client.reset()
    print(f" >>> Starting eval job from episode {episode + 1} of {rollouts} <<< ")
    while True:
        action, viz = policy_client.infer(obs, language_instruction)
        if viz is not None:
            video.append(viz)
        obs, rew, term, trunc, info = env.step(
            torch.tensor(action).reshape(1, -1), expensive=policy_client.rerender
        )

        bar.update(1)
        if term[0] or trunc[0] or bar.n >= horizon:
            policy_client.reset()

            # Save video and metadata
            filename = run_folder / f"episode_{episode}.mp4"
            mediapy.write_video(filename, video, fps=15)

            # Log episode results to CSV
            episode_data = {
                "episode": episode,
                "episode_length": bar.n,
                "success": info["rubric"]["success"],
                "progress": info["rubric"]["progress"],
            }
            episode_df = pd.concat(
                [episode_df, pd.DataFrame([episode_data])], ignore_index=True
            )
            episode_df.to_csv(csv_path, index=False)

            bar.close()
            print(f"Episode {episode} finished. Episode length: {bar.n}")
            bar = tqdm.tqdm(range(horizon))
            obs, info = env.reset(
                object_positions=initial_conditions[episode % len(initial_conditions)]
            )

            episode += 1
            video = []
            if episode >= rollouts:
                break

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    args: EvalArgs = tyro.cli(EvalArgs)
    main(args)
