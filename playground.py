import argparse
import os
import time
import numpy as np
import torch
import json

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
print('Current working directory：',os.getcwd())
import json
from src.models_dual_inter_traj_big.utils import Get_RC_Data,visuaulize,visuaulize_bianhao,seed_set,get_dct_matrix,gen_velocity,predict,update_metric,getRandomPermuteOrder,getRandomRotatePoseTransform
from src.baseline_3dpw_big.config import config
from src.models_dual_inter_traj_big.model import siMLPe as Model
from src.baseline_3dpw_big.lib.dataset.dataset_3dpw import get_3dpw_dataloader
from src.baseline_3dpw_big.lib.utils.logger import get_logger, print_and_log_info
from src.baseline_3dpw_big.lib.utils.pyt_utils import  ensure_dir
from src.baseline_3dpw_big.test import vim_test,random_pred,mpjpe_vim_test
import shutil
import warnings
# Ignore all warnings
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--exp-name', type=str, default="online_inference", help='=exp name')
parser.add_argument('--dataset', type=str, default="others", help='=exp name')
parser.add_argument('--seed', type=int, default=888, help='=seed')
parser.add_argument('--temporal-only', action='store_true', help='=temporal only')
parser.add_argument('--layer-norm-axis', type=str, default='spatial', help='=layernorm axis')
parser.add_argument('--with-normalization', type=bool,default=True, help='=use layernorm')# Unused
parser.add_argument('--spatial-fc', action='store_true', help='=use only spatial fc')
parser.add_argument('--normalization',type=bool,default=True, help='Normalize data')
parser.add_argument('--norm_way',type=str,default='first', help='=use only spatial fc')
parser.add_argument('--rc',type=bool,default=True, help='=use only spatial fc')
parser.add_argument('--permute_p', type=bool, default=True, help='Permute P dimension')
parser.add_argument('--random_rotate', type=bool, default=True, help='Random rotation around world center')
parser.add_argument('--num', type=int, default=64, help='=num of blocks')
parser.add_argument('--hd', type=int, default=256, help='=num of blocks')
parser.add_argument('--interaction_interval', type=int, default=16, help='Interval between local and Global interactions, must be divisible by num')
parser.add_argument('--weight', type=float, default=1., help='=loss weight')
parser.add_argument('--device', type=str, default="cuda:0")
parser.add_argument('--debug', type=bool, default=False)
parser.add_argument('--n_p', type=int, default=2)
parser.add_argument('--model_path', type=str, default='pt_ckpts/pt_rc.pth')
parser.add_argument('--vis_every', type=int, default=250000000000)
parser.add_argument('--save_every', type=int, default=100)
parser.add_argument('--print_every', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=128)
args = parser.parse_args()

# Create folder
expr_dir = os.path.join('exprs', args.exp_name)
if os.path.exists(expr_dir):
    shutil.rmtree(expr_dir)
os.makedirs(expr_dir, exist_ok=True)

# --- Define Logging Path ---
POSE_LOG_FILE = os.path.join(expr_dir, 'pose_log.jsonl')
print(f"Prediction log will be saved to: {POSE_LOG_FILE}")

# Configuration
config.rc=args.rc
config.norm_way=args.norm_way
config.normalization=args.normalization
config.batch_size = args.batch_size
config.dataset = args.dataset
config.n_p = args.n_p
config.vis_every = args.vis_every
config.save_every = args.save_every
config.print_every = args.print_every
config.debug = args.debug
config.device = args.device
config.expr_dir=expr_dir
config.motion_fc_in.temporal_fc = args.temporal_only
config.motion_fc_out.temporal_fc = args.temporal_only
config.motion_mlp.norm_axis = args.layer_norm_axis
config.motion_mlp.spatial_fc_only = args.spatial_fc
config.motion_mlp.with_normalization = args.with_normalization
config.motion_mlp.num_layers = args.num
config.motion_mlp.n_p=args.n_p
config.motion_mlp.interaction_interval = args.interaction_interval
config.motion_mlp.hidden_dim = args.hd
config.snapshot_dir=os.path.join(expr_dir, 'snapshot')
ensure_dir(config.snapshot_dir)# Create folder
config.vis_dir=os.path.join(expr_dir, 'vis')
ensure_dir(config.vis_dir)# Create folder
config.log_file=os.path.join(expr_dir, 'log.txt')
config.model_pth=args.model_path


dct_m,idct_m = get_dct_matrix(config.dct_len)
dct_m = torch.tensor(dct_m).float().to(config.device).unsqueeze(0)
idct_m = torch.tensor(idct_m).float().to(config.device).unsqueeze(0)
config.dct_m=dct_m
config.idct_m=idct_m

def log_prediction_data(past_poses_np, future_poses_np):
    """
    Saves the input and output data to the log file in JSON Lines format.
    Uses 'a' mode for append and immediate flush to ensure durability.
    """
    log_entry = {
        "timestamp": time.time(),
        "input_frames": past_poses_np.squeeze(0).tolist(), # Convert NumPy array to nested list
        "prediction_frames": future_poses_np.squeeze(0).tolist(), # Convert NumPy array to nested list
    }
    
    try:
        with open(POSE_LOG_FILE, 'a') as f:
            # Write a single line JSON object followed by a newline
            json.dump(log_entry, f)
            f.write('\n')
    except Exception as e:
        print(f"Error during log write: {e}")


def run_single_inference(latest_16_frames, model, config):
    # 1. Shape and Flatten (Matches the input preparation in your code)
    # Input shape: (16, P,  13, 3) -> target flat shape: (1, P, 16, 39)
    num_people = config.n_p

    # 1. Convert list of T frames to a PyTorch Tensor
    # input_np shape: (T, P, 13, 3)
    input_np = np.array(latest_16_frames) 
    print("input shape", input_np.shape)
    
    # Flatten the joint data (13*3=39)
    # Shape: (T, P, 39)
    input_flat = input_np.reshape(input_history_length, num_people, config.n_joint * 3)

    #Convert (T, P, D) to model-expected (B=1, P, T, D)
    # 1. Transpose T and P: (P, T, D)
    input_transposed = np.transpose(input_flat, (1, 0, 2))
    
    # 2. Add Batch dimension (B=1)
    # Final Shape: (1, P, 16 (T), 39)
    h36m_motion_input = torch.from_numpy(input_transposed).unsqueeze(0).float().to(device)

    # 3. Apply Transformations for RC (Root centering)
    if config.rc:
        h36m_motion_input = Get_RC_Data_Inference(h36m_motion_input)

    with torch.no_grad():
        # 4. Predict
        # motion_pred shape: (1, P, 14, 39)
        motion_pred = predict(model, h36m_motion_input, config)

    # 5. Reshape and Return
    b, p, n, c = motion_pred.shape
    # Reshape to (1, P, 14, 13, 3) for visualization/use
    h36m_motion_input_3d = h36m_motion_input.reshape(b, p, config.t_his, config.n_joint, 3)
    future_poses_3d = motion_pred.reshape(b, p, n, config.n_joint, 3) #n = T = number of frames pred

    return h36m_motion_input_3d, future_poses_3d

# 1. Root Correction (RC) Function for Inference
def Get_RC_Data_Inference(motion_input):
    """
    Applies Root Correction (RC) and velocity integration to a single sequence.
    This replaces the original Get_RC_Data which required both input and target.
    
    motion_input shape: (B, P, T, JK) -> (1, 1, 16, 39)
    """
    b, p, t, jk = motion_input.shape
    k = 3
    j = jk // k
    
    # 1. Reshape and calculate velocity
    motion = motion_input.reshape(b, p, t, j, k) # B, P, T, J, K
    
    # Calculate velocity: Vel_t = Pos_t+1 - Pos_t
    vel_data = torch.zeros((b, p, t, j, k)).to(motion.device) 
    vel_data[:, :, :-1, :, :] = motion[:, :, 1:, :, :] - motion[:, :, :-1, :, :]
    
    data = torch.cat((motion, vel_data), dim=-1) # B, P, T, J, 6 (Pos + Vel)
    data = data.transpose(1, 2) # B, T, P, J, 6
    
    # 2. Global Translation Correction
    # Estimate global movement from average velocity of all joints/people
    # Only use the non-zero velocity data (1:t)
    camera_vel = data[:, :t-1, :, :, 3:].mean(dim=(1, 2, 3)) # B, 3
    
    # Subtract camera velocity from ALL velocity vectors
    data[:, 1:, ..., 3:] -= camera_vel[:, None, None, None]
    
    # 3. Velocity-to-Position Reconversion (Integration)
    # Corrected Pos = Starting Pos + Cumulative Sum of Corrected Velocities
    data[..., :3] = data[:, 0:1, ..., :3] + data[..., 3:].cumsum(dim=1)
    
    # 4. Reshape and return corrected positions
    data = data.transpose(1, 2)[..., :3].reshape(b, p, t, jk) 
    
    return data

# 2. Checkpoint Loading Utility
def load_checkpoint(model, model_path, device):
    try:
        if model_path.endswith('.pth'):
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict, strict=True) 
            print(f"Successfully loaded checkpoint from {model_path}")
    except FileNotFoundError:
        print(f"WARNING: Checkpoint file not found at {model_path}. Starting with uninitialized weights.")
    except Exception as e:
        print(f"ERROR loading checkpoint: {e}")

def generate_rand():
    #Frame - people - joints - 3D coordinate
    # random_joints= np.random.rand(16,2,13,3)
    random_joints=np.random.rand(1,2,13,3)
    return random_joints

def prepare_dynamic_batch(keypoints_list):
    """
    Args:
        keypoints_list: List of numpy arrays, where each is (13, 3)
                        Length of list = Number of people detected (N)
    Returns:
        input_tensor: Torch tensor of shape (Batch, 2, 13, 3)
    """
    num_people = len(keypoints_list)
    
    # --- Case 0: No one detected ---
    if num_people == 0:
        return None

    # --- Case 1: Single Person (Ghost Padding) ---
    if num_people == 1:
        real_p = keypoints_list[0] # Shape (13, 3)
        ghost_p = np.zeros_like(real_p)
        # Combine to shape (1, 2, 13, 3)
        batch = np.array([[real_p, ghost_p]])
        return torch.tensor(batch).float()

    # --- Case 2: Exactly Two People ---
    if num_people == 2:
        # Combine to shape (1, 2, 13, 3)
        batch = np.array([[keypoints_list[0], keypoints_list[1]]])
        return torch.tensor(batch).float()

    # --- Case 3: More than 2 (Chunking Strategy) ---
    # We will group them into pairs: (P1,P2), (P3,P4), ...
    batch_items = []
    
    for i in range(0, num_people, 2):
        person_a = keypoints_list[i]
        
        # Check if there is a partner for this chunk
        if i + 1 < num_people:
            person_b = keypoints_list[i+1]
        else:
            # Odd number of people? Last person gets a ghost partner
            person_b = np.zeros_like(person_a)
            
        batch_items.append([person_a, person_b])
        
    # Stack into final batch
    # Result shape: (Num_Pairs, 2, 13, 3)
    final_batch = np.array(batch_items)
    return torch.tensor(final_batch).float()


'''-------------This is the beginning of the main code---------------------'''
model = Model(config).to(device=config.device)
print(">>> total params: {:.2f}M".format(
    sum(p.numel() for p in list(model.parameters())) / 1000000.0))
load_checkpoint(model, config.model_pth, config.device)

device = config.device
model.eval() 
device = config.device


input_history_length=config.t_his
past_joints = [] 
frame_counter = 0

'''Loading a pre-recorded JSON file from zed camera'''
with open('bodies.json','r') as file:
    data = json.load(file)

keypoints_stored = []
for timestamp, frame_data in data.items():
    try:
        body_list = frame_data['body_list']

        for body_data in body_list:
            keypoints = body_data['keypoint']
            # indices=[0,2,3,4,5,6,7,8,9,10,11,12,13]
            zed_to_lsp_indices = [10, 9, 8, 11, 12, 13, 4, 3, 2, 5, 6, 7, 0]
            keypoints_13 = [keypoints[x] for x in zed_to_lsp_indices]
            
            keypoints_stored.append(keypoints_13)

    except KeyError as e:
        print(f"Key error accessing data : {e}")
        continue
# print(json.dumps(data))
print(len(keypoints_stored))

for i in range(20):
    # test_input = generate_rand()
    test_input = np.array(keypoints_stored[i])

    input_tensor = prepare_dynamic_batch(test_input)
    model_input = input_tensor.to(config.device)

    # ghost = np.copy(test_input)

    # model_input=np.concatenate((test_input,ghost), axis=1)



    # print("test input looks like", np.shape(test_input))

    if len(past_joints) == input_history_length:
        past_joints.pop(0)
    
    past_joints.append(model_input)

    if len(past_joints) == input_history_length:
        print("Entering prediction code")
        input_joints , output_joints = run_single_inference(past_joints,model,config)
        print("Prediction ended", output_joints.shape)

        motion=torch.cat([input_joints,output_joints],dim=2).cpu().detach().numpy()
        visuaulize(motion,"trial",config.vis_dir,input_len=15,dataset='mupots')

        # Convert tensors to NumPy arrays for logging
        past_poses_np = input_joints.cpu().numpy()
        future_poses_np = output_joints.cpu().numpy()

        # --- LOGGING STEP (Ensures durability) ---
        log_prediction_data(past_poses_np, future_poses_np)
    
    i+=1