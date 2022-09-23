import os
import os.path as osp
import time
import argparse
from copy import deepcopy
from collections import OrderedDict

import numpy as np
import gym
import cv2
import torch
import open3d as o3d
import pyrealsense2 as rs

from .xmate3_robot import RealRobotController, Robot
from .realsense import RealSense

from maniskill2_learn.utils.meta import Config, DictAction, get_logger
from maniskill2_learn.networks.builder import build_model
from maniskill2_learn.networks.utils import (
    get_kwargs_from_shape,
    replace_placeholder_with_args,
)
from maniskill2_learn.utils.lib3d.mani_skill2_contrib import (
    apply_pose_to_points,
    apply_pose_to_point,
)
from maniskill2_learn.utils.data import GDict
from real_robot_eval_examples.env_utils import build_env


def plan_joint_traj(init_qpos: np.ndarray, target_qpos: np.ndarray, steps: int):
    """
    uniformly interpolate in joint space
    init_qpos: [7, ]
    target_qpos: [7, ]
    steps: int
    """
    plan = {}
    plan["position"] = np.linspace(init_qpos, target_qpos, steps)
    plan["velocity"] = np.ones_like(plan["position"]) * 0.1
    return plan


def gripper_action_sim2real(qpos):
    return int(np.clip(qpos / 0.068 * 255, 0, 255))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate pick cube visual RL policy trained in simulation"
    )
    # Parameters for log dir
    parser.add_argument("--output-dir", required=True, help="output directory")
    parser.add_argument("--config", required=True, help="config file path")
    parser.add_argument("--model-path", required=True, help="checkpoint path")
    parser.add_argument(
        "--control-freq", required=True, help="real robot control frequency"
    )
    parser.add_argument(
        "--sub-step", required=True, type=int, help="number of substep for real robot"
    )

    args = parser.parse_args()

    # Merge cfg with args.cfg_options
    cfg = Config.fromfile(args.config)

    return args, cfg


if __name__ == "__main__":
    np.set_printoptions(suppress=True, precision=4)

    args, cfg = parse_args()
    torch.cuda.set_device(0)
    torch.set_num_threads(1)
    device = "cuda"
    args.timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    args.output_dir = f"{args.output_dir}/{args.timestamp}"
    os.makedirs(osp.abspath(args.output_dir), exist_ok=True)

    logger = get_logger(
        name="Sim2Real EVAL",
        log_file=osp.join(args.output_dir, f"{args.timestamp}.log"),
    )

    logger.info(f"Args: \n{args}")
    logger.info(f"Config:\n{cfg.pretty_text}")

    #### simulation env set up ####
    # since we do evaluation, we use eval_env_cfg to override env_cfg
    env_cfg = cfg["env_cfg"]
    eval_env_cfg = cfg["eval_cfg"]["env_cfg"]
    if eval_env_cfg:
        for k, v in eval_env_cfg:
            env_cfg[k] = v

    env = build_env(env_cfg)
    unwrapped_env = env.unwrapped
    scene = unwrapped_env._scene
    obs = env.reset()
    viewer = unwrapped_env._viewer
    robot = unwrapped_env.agent.robot
    init_sim_qpos = robot.get_qpos()

    def wait_to_continue(key: str):
        print(f'Press "{key}" to continue')
        while not viewer.closed:
            scene.update_render()
            viewer.render()
            if viewer.window.key_down(key):
                break

    def follow_traj(traj):
        n_step = traj["position"].shape[0]
        for i in range(n_step):
            qf = robot.compute_passive_force(
                gravity=True, coriolis_and_centrifugal=True
            )
            robot.set_qf(qf)
            for j in range(7):
                robot.get_active_joints()[j].set_drive_target(traj["position"][i][j])
                robot.get_active_joints()[j].set_drive_velocity_target(
                    traj["velocity"][i][j]
                )
            for i in range(
                int(unwrapped_env.control_timestep / unwrapped_env.sim_timestep)
            ):
                unwrapped_env.step()
                unwrapped_env.render()

    logger.info(f"Observation space {env.observation_space}")
    logger.info(f"Action space {env.action_space}")
    logger.info(f"Control mode {env.control_mode}")
    logger.info(f"Reward mode {env.reward_mode}")

    actor_cfg = cfg.agent_cfg.actor_cfg
    actor_cfg.pop("optim_cfg")
    actor_cfg["obs_shape"] = GDict(obs).shape
    actor_cfg["action_shape"] = len(env.action_space.sample())
    actor_cfg["action_space"] = deepcopy(env.action_space)
    replaceable_kwargs = get_kwargs_from_shape(
        GDict(obs).list_shape, env.action_space.sample().shape[0]
    )
    cfg = replace_placeholder_with_args(cfg, **replaceable_kwargs)
    logger.info(f"Final actor config:\n{cfg.agent_cfg.actor_cfg}")
    actor = build_model(cfg.agent_cfg.actor_cfg).to(device)

    # load weight
    state_dict = torch.load(args.model_path)["state_dict"]
    logger.info(f"State dict keys in model checkpoint: {state_dict.keys()}")
    actor_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k[:6] == "actor.":
            actor_state_dict[k[6:]] = v

    assert (
        actor.state_dict().keys() == actor_state_dict.keys()
    ), f"Actor keys: {actor.state_dict().keys()}\nweight keys: {actor_state_dict.keys()}"
    actor.load_state_dict(actor_state_dict)
    actor.eval()

    #### real env set up ####
    real_robot_controller = RealRobotController(
        ROS_rate=args.control_freq, joint_stiffness=[1000] * 3 + [200] * 4
    )
    # move the simulated robot to the same configuration as the real robot.
    print("waiting for state msg to set the initial configuration in the simulation...")
    init_real_state = real_robot_controller.get_realstate()
    robot.set_qpos(np.concatenate([init_real_state.q_m, np.zeros(2)]))

    print("Initial configuration set. Press [f] to play traj in simulation.")
    wait_to_continue("f")
    # move to init qpos in simulation env
    init_move_plan = plan_joint_traj(init_real_state.q_m[:7], init_sim_qpos[:7], 600)
    follow_traj(init_move_plan)
    print("sim qpos:", robot.get_qpos(), " target sim qpos: ", init_sim_qpos)

    print("Press [r] to control the real robot: joint align.")
    wait_to_continue("r")
    real_robot_controller.exec_trajs(joint_move_plan)
    init_real_state = real_robot_controller.get_realstate()
    print("real qpos: ", init_real_state.q_m)

    real_realsense = RealSense(SN="105422061051")

    def get_real_obs() -> dict:
        # hyper parameters
        num_points = 2048
        aabb = o3d.geometry.AxisAlignedBoundingBox(
            min_bound=np.array([-0.35, -0.15, -0.05]),
            max_bound=np.array([0.15, 0.15, 1.0]),
        )
        # acquire real observations
        _, scene_pts = real_realsense.get_depth_and_pointcloud()
        real_robot_state = real_robot_controller.get_realstate()
        real_gripper_state = real_robot_controller.gripper_listener.gPO
        real_gripper_state = np.repeat(real_gripper_state / 255 * 0.068, 2)
        real_qpos = np.concatenate((real_robot_state.q_m, real_gripper_state))

        # compose to policy input
        real_obs = {}
        scene_pcd = o3d.geometry.PointCloud()
        scene_pcd.points = o3d.utility.Vector3dVector(scene_pts)
        scene_pcd = scene_pcd.crop(aabb)
        scene_pcd = scene_pcd.random_down_sample(
            num_points / np.asarray(scene_pcd.points).shape[0]
        )
        scene_pts = np.asarray(scene_pcd.points)

        # Append green points near the goal to the point cloud, which serves as visual goal indicator,
        # if self.n_goal_points is given and the environment returns goal information
        # Also, transform these points to self.obs_frame
        if env.n_goal_points > 0:
            assert (
                env.goal_pos is not None
            ), "n_goal_points should only be used if goal_pos(e) is contained in the environment observation"
            goal_pts = (
                np.random.uniform(low=-1.0, high=1.0, size=(env.n_goal_points, 3))
                * 0.01
            )
            goal_pts = goal_pts + env.goal_pos[None, :]
            scene_pts = np.concatenate([scene_pts, goal_pts], axis=0)

        # Calculate coordinate transformations that transforms poses in the world to self.obs_frame
        # These "to_origin" coordinate transformations are formally T_{self.obs_frame -> world}^{self.obs_frame}
        assert env.obs_frame == "ee"
        robot.set_qpos(real_qpos)  # align sim to real
        tcp_pose = unwrapped_env.tcp.pose
        to_origin = Pose(p=tcp_pose.p, q=tcp_pose.q).inv()
        xyz = apply_pose_to_points(scene_pts, to_origin)
        real_obs["xyz"] = xyz

        frame_related_states = []
        base_info = apply_pose_to_point(robot.pose, to_origin)
        frame_related_states.append(base_info)
        tcp_info = apply_pose_to_point(tcp_pose.p, to_origin)
        frame_related_states.append(tcp_info)
        goal_info = apply_pose_to_point(env.goal_pos, to_origin)
        frame_related_states.append(goal_info)
        tcp_to_goal_pos = env.goal_pos - tcp_pose.p
        tcp_to_goal_info = apply_pose_to_point(tcp_to_goal_pos, to_origin)
        frame_related_states.append(tcp_to_goal_info)
        frame_related_states = np.stack(frame_related_states, axis=0)

        real_obs["state"] = np.concatenate(
            [real_qpos, frame_goal_related_poses.flatten()]
        )

        return real_obs

    # begin evaluation
    done = False
    step = 0
    with torch.no_grad():
        while (not done) and step < 200:
            obs = get_real_obs()
            obs = (
                GDict(obs)
                .unsqueeze(0)
                .to_torch(dtype="float32", wrapper=False, device=device)
            )
            action = actor(obs)[0]
            action = action.cpu().numpy()
            _, rew, done, info = env.step(action)
            env.render()
            logger.info(f"STEP: {step:5d}, rew: {rew}, done: {done}, info: {info}")
            step += 1

            # translate to real action
            curr_qpos = np.asarray(real_robot_controller.get_realstate().q_m)
            real_arm_qpos_residual = action[:7]/10  # [-1, 1] -> [-0.1, 0.1]
            real_gripper_qpos = (action[7]+1)/2*0.078
            print("step in real")
            wait_to_continue('r')
            for substep in range(args.sub_step):
                real_robot_controller.exec_action(curr_qpos+(substep/args.sub_step)*real_arm_qpos_residual)
                real_robot_controller.exec_gripper(gripper_action_sim2real(real_gripper_qpos))

