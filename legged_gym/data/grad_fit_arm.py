import glob
import os
import sys
import pdb
import os.path as osp
sys.path.append(os.getcwd())

from smpl_sim.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonMotion, SkeletonState
from scipy.spatial.transform import Rotation as sRot
import numpy as np
import torch
from phc.smpllib.smpl_parser import (
    SMPL_Parser,
    SMPLH_Parser,
    SMPLX_Parser,
    SMPL_BONE_ORDER_NAMES, 
)
import joblib
from phc.utils.rotation_conversions import axis_angle_to_matrix
from phc.utils.torch_arm_humanoid_batch import Humanoid_Batch
from torch.autograd import Variable
from tqdm import tqdm
import argparse

def load_amass_data(data_path):
    entry_data = dict(np.load(open(data_path, "rb"), allow_pickle=True))

    if not 'mocap_framerate' in  entry_data:
        return 
    framerate = entry_data['mocap_framerate']


    root_trans = entry_data['trans']
    pose_aa = np.concatenate([entry_data['poses'][:, :66], np.zeros((root_trans.shape[0], 6))], axis = -1)
    betas = entry_data['betas']
    gender = entry_data['gender']
    N = pose_aa.shape[0]
    return {
        "pose_aa": pose_aa,
        "gender": gender,
        "trans": root_trans, 
        "betas": betas,
        "fps": framerate
    }
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--amass_root", type=str, default="data/AMASS/KIT/0")
    args = parser.parse_args()
    
    device = torch.device("cpu")

    h1_rotation_axis = torch.tensor([[
        [0, 1, 0], # left_shoulder_pitch_joint
        [1, 0, 0], # left_shoulder_roll_joint
        [0, 0, 1], # left_shoulder_yaw_joint
        [0, 1, 0], # left_elbow_joint
        [1, 0, 0], # left_wrist_roll_joint
        [0, 0, 1], # left_wrist_yaw_joint
        [0, 1, 0], # left_wrist_pitch_joint
        
        [0, 1, 0], # right_shoulder_pitch_joint
        [1, 0, 0], # right_shoulder_roll_joint
        [0, 0, 1], # right_shoulder_yaw_joint
        [0, 1, 0], # right_elbow_joint
        [1, 0, 0], # right_wrist_roll_joint
        [0, 0, 1], # right_wrist_yaw_joint
        [0, 1, 0], # right_wrist_pitch_joint
    ]]).to(device)

    h1_joint_names = [ 'base_link',
                    'left_shoulder_pitch_link', 'left_shoulder_roll_link', 'left_shoulder_yaw_link', 'left_elbow_link', "left_wrist_roll_link", "left_wrist_yaw_link","left_wrist_pitch_link",
                    'right_shoulder_pitch_link', 'right_shoulder_roll_link', 'right_shoulder_yaw_link', 'right_elbow_link', "right_wrist_roll_link", "right_wrist_yaw_link","right_wrist_pitch_link"]

    # h1_joint_names_augment = h1_joint_names
    # h1_joint_pick = ['base_link', "left_shoulder_roll_link", "left_elbow_link", "left_wrist_roll_link", "right_shoulder_roll_link", "right_elbow_link", "right_wrist_roll_link"]
    # smpl_joint_pick = ["Pelvis", "L_Shoulder", "L_Elbow", "L_Hand", "R_Shoulder", "R_Elbow", "R_Hand"]
    h1_joint_names_augment = h1_joint_names + ["head_link"]
    h1_joint_pick = ['base_link', "left_shoulder_roll_link", "left_elbow_link", "left_wrist_pitch_link", "right_shoulder_roll_link", "right_elbow_link", "right_wrist_pitch_link", "head_link"]
    smpl_joint_pick = ["Pelvis", "L_Shoulder", "L_Elbow", "L_Hand", "R_Shoulder", "R_Elbow", "R_Hand", "Head"]

    h1_joint_pick_idx = [ h1_joint_names_augment.index(j) for j in h1_joint_pick]
    smpl_joint_pick_idx = [SMPL_BONE_ORDER_NAMES.index(j) for j in smpl_joint_pick]

    smpl_parser_n = SMPL_Parser(model_path="data/smpl", gender="neutral")
    smpl_parser_n.to(device)


    shape_new, scale = joblib.load("data/h1/shape_optimized_v1.pkl")
    shape_new = shape_new.to(device)
    scale = scale.mean()  # 确保 scale 是标量

    amass_root = args.amass_root
    all_pkls = glob.glob(f"{amass_root}/*.npz", recursive=True)
    split_len = len(amass_root.split("/"))
    key_name_to_pkls = {"0-" + "_".join(data_path.split("/")[split_len:]).replace(".npz", ""): data_path for data_path in all_pkls}
    
    if len(key_name_to_pkls) == 0:
        raise ValueError(f"No motion files found in {amass_root}")

    h1_fk = Humanoid_Batch(device = device)
    data_dump = {}
    pbar = tqdm(key_name_to_pkls.keys())


    for data_key in pbar:
        amass_data = load_amass_data(key_name_to_pkls[data_key])
        skip = int(amass_data['fps']//30)
        trans = torch.from_numpy(amass_data['trans'][::skip]).float().to(device)
        N = trans.shape[0]
        pose_aa_walk = torch.from_numpy(np.concatenate((amass_data['pose_aa'][::skip, :66], np.zeros((N, 6))), axis = -1)).float().to(device)


        verts, joints = smpl_parser_n.get_joints_verts(pose_aa_walk, torch.zeros((1, 10)).to(device), trans)
        offset = joints[:, 0] - trans
        root_trans_offset = trans + offset

        pose_aa_h1 = np.repeat(np.repeat(sRot.identity().as_rotvec()[None, None, None, ], 22, axis = 2), N, axis = 1)
        pose_aa_h1[..., 0, :] = (sRot.from_rotvec(pose_aa_walk.cpu().numpy()[:, :3]) * sRot.from_quat([0.5, 0.5, 0.5, 0.5]).inv()).as_rotvec()
        pose_aa_h1 = torch.from_numpy(pose_aa_h1).float().to(device)
        gt_root_rot = torch.from_numpy((sRot.from_rotvec(pose_aa_walk.cpu().numpy()[:, :3]) * sRot.from_quat([0.5, 0.5, 0.5, 0.5]).inv()).as_rotvec()).float().to(device)

        dof_pos = torch.zeros((1, N, 14, 1)).to(device)

        dof_pos_new = Variable(dof_pos, requires_grad=True)
        optimizer_pose = torch.optim.Adadelta([dof_pos_new],lr=100) #100
        # optimizer_pose = torch.optim.Adam([dof_pos_new],lr=10)

        verts, joints = smpl_parser_n.get_joints_verts(pose_aa_walk, shape_new, trans)
        joints = (joints - joints[:, 0:1]) * scale + joints[:, 0:1]
        pose_aa_h1_new = torch.cat([gt_root_rot[None, :, None], h1_rotation_axis * dof_pos_new, torch.zeros((1, N, 2, 3)).to(device)], axis = 2).to(device)
        fk_return = h1_fk.fk_batch(pose_aa_h1_new, root_trans_offset[None, ])
        # -----------------------------vis-----------------------------
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
        import matplotlib.pyplot as plt

        # j3d = fk_return.global_translation_extend[0, :, h1_joint_pick_idx, :] .detach().numpy()
        j3d = fk_return.global_translation[0, :, h1_joint_pick_idx, :] .detach().numpy()
        j3d = j3d - j3d[:, 0:1]
        j3d_joints = joints[:, smpl_joint_pick_idx].detach().numpy()
        j3d_joints = j3d_joints - j3d_joints[:, 0:1]
        idx = 0
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(90, 0)
        ax.scatter(j3d[idx, :,0], j3d[idx, :,1], j3d[idx, :,2], label='Humanoid Shape', c='blue')
        ax.scatter(j3d_joints[idx, :,0], j3d_joints[idx, :,1], j3d_joints[idx, :,2], label='Fitted Shape', c='red')

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        drange = 1
        ax.set_xlim(-drange, drange)
        ax.set_ylim(-drange, drange)
        ax.set_zlim(-drange, drange)
        ax.legend()
        plt.show()

        for iteration in range(1000):
            verts, joints = smpl_parser_n.get_joints_verts(pose_aa_walk, shape_new, trans)
            joints = (joints - joints[:, 0:1]) * scale + joints[:, 0:1]
            pose_aa_h1_new = torch.cat([gt_root_rot[None, :, None], h1_rotation_axis * dof_pos_new, torch.zeros((1, N, 2, 3)).to(device)], axis = 2).to(device)
            fk_return = h1_fk.fk_batch(pose_aa_h1_new, root_trans_offset[None, ])
            
            # diff = fk_return['global_translation_extend'][:, :, h1_joint_pick_idx] - joints[:, smpl_joint_pick_idx]
            diff = fk_return['global_translation'][:, :, h1_joint_pick_idx] - joints[:, smpl_joint_pick_idx]
            loss_g = diff.norm(dim = -1).mean() 
            loss = loss_g
            
            
            pbar.set_description_str(f"{iteration} {loss.item() * 1000}")

            optimizer_pose.zero_grad()
            loss.backward()
            optimizer_pose.step()
            
            dof_pos_new.data.clamp_(h1_fk.joints_range[:, 0, None], h1_fk.joints_range[:, 1, None])
            
        dof_pos_new.data.clamp_(h1_fk.joints_range[:, 0, None], h1_fk.joints_range[:, 1, None])
        pose_aa_h1_new = torch.cat([gt_root_rot[None, :, None], h1_rotation_axis * dof_pos_new, torch.zeros((1, N, 2, 3)).to(device)], axis = 2)
        fk_return = h1_fk.fk_batch(pose_aa_h1_new, root_trans_offset[None, ])

        root_trans_offset_dump = root_trans_offset.clone()

        # root_trans_offset_dump[..., 2] -= fk_return.global_translation_extend[..., 2].min().item() - 0.08
        root_trans_offset_dump[..., 2] -= fk_return.global_translation[..., 2].min().item() - 0.08
        # 添加全局位置到 data_dump
        global_positions = fk_return.global_translation.squeeze().cpu().detach().numpy()  # [T, J, 3]

        data_dump[data_key]={
                "root_trans_offset": root_trans_offset_dump.squeeze().cpu().detach().numpy(),
                "pose_aa": pose_aa_h1_new.squeeze().cpu().detach().numpy(),   
                "dof": dof_pos_new.squeeze().detach().cpu().numpy(), 
                "root_rot": sRot.from_rotvec(gt_root_rot.cpu().numpy()).as_quat(),
                "global_position": global_positions,  # 新增：所有关节的全局位置
                "fps": 30
                }
        
        print(f"dumping {data_key} for testing, remove the line if you want to process all data")
        # import ipdb; ipdb.set_trace()
        joblib.dump(data_dump, "data/h1/test.pkl")


    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
    import matplotlib.pyplot as plt

    # j3d = fk_return.global_translation_extend[0, :, h1_joint_pick_idx, :] .detach().numpy()
    j3d = fk_return.global_translation[0, :, h1_joint_pick_idx, :] .detach().numpy()
    j3d = j3d - j3d[:, 0:1]
    j3d_joints = joints[:, smpl_joint_pick_idx].detach().numpy()
    j3d_joints = j3d_joints - j3d_joints[:, 0:1]
    idx = 0
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(90, 0)
    ax.scatter(j3d[idx, :,0], j3d[idx, :,1], j3d[idx, :,2], label='Humanoid Shape', c='blue')
    ax.scatter(j3d_joints[idx, :,0], j3d_joints[idx, :,1], j3d_joints[idx, :,2], label='Fitted Shape', c='red')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    drange = 1
    ax.set_xlim(-drange, drange)
    ax.set_ylim(-drange, drange)
    ax.set_zlim(-drange, drange)
    ax.legend()
    plt.show()
        
    # import ipdb; ipdb.set_trace()
    # joblib.dump(data_dump, "data/h1/amass_all.pkl")