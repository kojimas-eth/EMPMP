import argparse
import os
import time
import numpy as np
import torch
import json
from collections import deque, defaultdict


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

class PersonHistoryManager:
    def __init__(self,history_length = 16):
        self.history_length = history_length
        self.buffers = defaultdict(lambda: deque(maxlen=history_length))

    def update_and_get_valid_batch(self,zed_body_list):
        """
        Args:
            zed_body_list: The 'body_list' from the current live ZED frame.
        Returns:
            valid_ids: List of IDs included in the batch
            batch_data: List of trajectories (16, 13, 3) ready for inference
        """
        current_frame_ids=set()
        ready_trajectories=[]
        ready_ids=[]

        # 1. Update buffers for visible people
        for body in zed_body_list:
            # if body.tracking_state != sl.OBJECT_TRACKING_STATE.OK:
            #     continue
            raw_kp=body.keypoint

            if np.isnan(raw_kp).any() or np.isinf(raw_kp).any():
                continue
            
            uid = body.id
            current_frame_ids.add(uid)
            
            #reorder into right index and add into history
            # zed_to_lsp_indices = [8,11,9,12,10,13,0,2,5,3,4,6,7] 
            zed_to_lsp_indices = [11, 8, 12, 9, 13, 10, 0, 5, 2, 6, 3, 7, 4]

            kp_13 = np.array([raw_kp[x] for x in zed_to_lsp_indices])
            self.buffers[uid].append(kp_13)


        # 2. Clean up people who left the frame
        active_id = list(self.buffers.keys())
        for uid in active_id:
            if uid not in current_frame_ids:
                # User left the FOV. Clear their buffer or keep it briefly?
                # Simplest approach: Delete immediately to save memory
                del self.buffers[uid]

        # 3. Check whether there is enough history
        for uid in current_frame_ids:
            if len(self.buffers[uid]) == self.history_length:
                traj = np.array(self.buffers[uid])
                ready_trajectories.append(traj)
                ready_ids.append(uid)
                print(f"{uid} is ready for prediction")
            #Check for FPS issues
            # full_buffer = list(self.buffers[uid])
            # if len(full_buffer) >= 32:
            #     downsampled_traj = np.array(full_buffer)[::2]
            #     ready_trajectories.append(downsampled_traj)
            #     print(f"{uid} is ready for prediction")
            # else:
                pass

        return ready_ids, ready_trajectories

            
class MockBody:
    """
    A dummy class that mimics the structure of sl.BodyData
    so the PersonHistoryManager thinks it's looking at live camera data.
    """
    def __init__(self, json_data):
        # Copy the ID (Standard integer ID)
        self.id = json_data.get('id')
        
        # Copy the Unique UUID (String) - vital for tracking across frames
        self.unique_object_id = str(json_data.get('unique_object_id', self.id))
        
        # Load the raw keypoints as a numpy array
        self.keypoint = np.array(json_data['keypoint'])
        
        # Handle Tracking State
        # The manager ignores bodies if state != OK. 
        # Since this is recorded data, we force it to OK so the manager accepts it.
        # self.tracking_state = sl.OBJECT_TRACKING_STATE.OK
        
        # Optional: If you need head position later
        if 'head_position' in json_data:
            self.head_position = np.array(json_data['head_position'])



def log_prediction_data(past_poses_np, future_poses_np):
    """
    Saves the input and output data to the log file in JSON Lines format.
    Uses 'a' mode for append and immediate flush to ensure durability.
    """
    log_entry = {
        "timestamp": time.time(),
        "input_frames": past_poses_np.tolist(), # Convert NumPy array to nested list
        "prediction_frames": future_poses_np.tolist(), # Convert NumPy array to nested list
    }
    
    try:
        with open(POSE_LOG_FILE, 'a') as f:
            # Write a single line JSON object followed by a newline
            json.dump(log_entry, f)
            f.write('\n')
    except Exception as e:
        print(f"Error during log write: {e}")


def run_live_inference(zed_body_list, model, config):
    #1. Update hisotry and get list of people with all histories
    valid_ids, valid_trajectories = history_manager.update_and_get_valid_batch(zed_body_list)

    num_people = len(valid_ids)
    print(f"making prediction for {num_people}")

    if num_people ==0:
        return None
    
    #2. Batch into groups of two to match model
    batched_inputs = []

    for i in range(0,num_people,2):
        person_a = valid_trajectories[i]

        if i+1 < num_people:
            person_b = valid_trajectories[i+1]
        else:
            person_b=np.zeros_like(person_a)
        
        batched_inputs.append([person_a,person_b])
    
    #3. shape the input to correct shape. Current Shape: (Num_Pairs, 2, 16, 13, 3)
    input_tensor = torch.tensor(np.array(batched_inputs)).float().to(config.device)
    b, p, t, j, c = input_tensor.shape
    model_input = input_tensor.reshape(b, p, t, j*c) # (Num_Pairs, 2, 16, 39)
    
    # 4. Root Correction (if needed)
    camera_vel= None
    if config.rc:
        model_input, camera_vel = Get_RC_Data_Inference(model_input,frame_index)
    
    #5. Do the prediction
    # motion_pred is (Num_Pairs, 2, Future_Frames, 39)
    motion_pred = predict(model,model_input,config)

    #add back the RC velocity
    if camera_vel is not None:
        future_frames = motion_pred.shape[2]
        time_steps = torch.arange(1,future_frames+1, device=config.device).float()
        #separate the last dim back into xyz and joints to only change xyz
        mp_b, mp_p, mp_t, mp_d = motion_pred.shape
        motion_pred_reshaped = motion_pred.reshape(mp_b,mp_p,mp_t,-1,3)

        drift = camera_vel.view(mp_b, 1, 1, 1, 3) * time_steps.view(1, 1, mp_t, 1, 1)
        # ADD THE DRIFT BACK
        motion_pred_reshaped += drift

        # Real History Last Frame (Center of mass or Root)
        real_last_pos = input_tensor[:, :, -1, :, :].mean(dim=2, keepdim=True) # (B, P, 1, 3)
        
        # Treadmill History Last Frame (Center of mass or Root)
        # We have to reshape model_input back to read it
        treadmill_last_pos = model_input.reshape(b, p, t, j, c)[:, :, -1, :, :].mean(dim=2, keepdim=True)
        
        # The positional offset caused by RC
        rc_pos_offset = real_last_pos.unsqueeze(2) - treadmill_last_pos.unsqueeze(2) # (B, P, 1, 1, 3)
        
        # Add the positional offset to the whole prediction
        motion_pred_reshaped += rc_pos_offset

        # Flatten back to original shape for your existing post-processing
        motion_pred = motion_pred_reshaped.reshape(mp_b, mp_p, mp_t, mp_d)
  


    pred_flat = motion_pred.reshape(-1, motion_pred.shape[2], config.n_joint, 3)
    pred_flat = pred_flat[:num_people]

    # model_input is (Num_Pairs, 2, 16, 39)
    # 1. Reshape (39) -> (13, 3)
    # Shape becomes: (Num_Pairs, 2, 16, 13, 3)
    input_reshaped = model_input.reshape(model_input.shape[0], 2, config.t_his, config.n_joint, 3)
    
    # 2. Flatten pairs to list of people 
    # Shape becomes: (Num_Pairs*2, 16, 13, 3)
    input_flat = input_reshaped.view(-1, config.t_his, config.n_joint, 3)
    
    # Shape becomes: (Total_People, 16, 13, 3)
    input_flat = input_flat[:num_people]

    #6. return the results into each uid
    results = {}
    for idx, uid in enumerate(valid_ids):
        results[uid] = pred_flat[idx].detach().cpu().numpy()
        
    return input_flat, pred_flat


# 1. Root Correction (RC) Function for Inference
def Get_RC_Data_Inference(motion_input,frame_index):
    """
    Applies Root Correction (RC) and velocity integration to a single sequence.
    This replaces the original Get_RC_Data which required both input and target.
    
    motion_input shape: (B, P, T, JK) -> (1, 1, 16, 39)
    camera_vel: The removed global velocity vector (B, 3)
    """
    b, p, t, jk = motion_input.shape
    k = 3
    j = jk // k
    
    # 1. Reshape
    motion = motion_input.reshape(b, p, t, j, k)
    
    # 2. Velocity Calculation
    vel_data = torch.zeros((b, p, t, j, k)).to(motion.device) 
    vel_data[:, :, :-1, :, :] = motion[:, :, 1:, :, :] - motion[:, :, :-1, :, :]
    
    data = torch.cat((motion, vel_data), dim=-1)
    
    # Transpose to (B, T, P, J, 6)
    data = data.transpose(1, 2) 
    
    # 3. Calculate Global Velocity (The "Drift")
    # Average over Time(1), Person(2), and Joint(3) to get (B, 3)
    # Note: We use t-1 because velocity is 0 for the last frame in the way we calculated it

    #Trying to average JUST HIP
    # camera_vel = data[:, :t-1, :, :, 3:].mean(dim=(1, 2, 3)) 
    camera_vel = data[:, :t-1, :, 0, 3:].mean(dim=(1, 2)) # Shape (B, 3)
    
    # 4. Remove Drift
    # FORCE the shape for broadcasting: (Batch, Time=1, Person=1, Joint=1, Coords=3)
    # This aligns perfectly with data shape (B, T, P, J, 3)
    camera_vel_broadcast = camera_vel.view(b, 1, 1, 1, 3)
    
    # Subtract from all frames
    data[:, 1:, ..., 3:] -= camera_vel_broadcast
    
    # 5. Integrate back to positions (Treadmill positions)
    data[..., :3] = data[:, 0:1, ..., :3] + data[..., 3:].cumsum(dim=1)
    
    # 6. Reshape back to original format
    data = data.transpose(1, 2)[..., :3].reshape(b, p, t, jk) 
    print(f"velocity for frame{frame_index} IS {camera_vel}")
    
    return data, camera_vel
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
    print(f"think number of people is {num_people}")
    
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

model.eval() 
device = config.device


input_history_length=config.t_his
past_joints = [] 
frame_counter = 0

history_manager = PersonHistoryManager(history_length=input_history_length)

# --- Define Logging Path ---
POSE_LOG_FILE = os.path.join(expr_dir, 'pose_log3.jsonl')
print(f"Prediction log will be saved to: {POSE_LOG_FILE}")


# --- Load the Json file ---
with open('15fps_1.json','r') as file:
    data = json.load(file)

processed_frames_store = []
ordered_timestamps = list(data.keys())
print(f"Loaded {len(ordered_timestamps)} frames.")

#Loading the data into a format readable by code
for timestamp_key in ordered_timestamps:
    frame_data=data[timestamp_key]

    try:
        current_frame_people = []
        for body_dict in frame_data['body_list']:
            fake_body_object = MockBody(body_dict)
            current_frame_people.append(fake_body_object)

        processed_frames_store.append(current_frame_people)

    except KeyError as e:
        print(f"Key error accesing data {e}")
        continue

#Debug
if len(processed_frames_store) > 0:
    print(f"Type of first item: {type(processed_frames_store[0])}") 

#Main Loop
for frame_index, frame_bodies in enumerate(processed_frames_store):

    inference_results = run_live_inference(frame_bodies,model,config)
    
    #makes sure None is not assigned (when not enough history)
    if inference_results is None:
        print (f"  -> Frame {frame_index}: Not enough history yet (Buffering...)")
        continue

    #Extract the right information
    input_joints, output_joints = inference_results
    

    print(f"  -> Prediction successful for Frame {frame_index}!")
    # input_joints: (N, 16, 13, 3)
    # output_joints: (N, 14, 13, 3)
    # Result: (N, 30, 13, 3)
    motion=torch.cat([input_joints,output_joints],dim=1).cpu().detach().numpy()
 

    if frame_index % 10 ==0:
        motion_5d = motion[np.newaxis, ...]
        visuaulize(motion_5d,f"iter:{frame_index}",config.vis_dir,input_len=15,dataset='mupots')

    # Convert tensors to NumPy arrays for logging
    past_poses_np = input_joints.cpu().detach().numpy()
    future_poses_np = output_joints.cpu().detach().numpy()

    # --- LOGGING STEP (Ensures durability) ---
    log_prediction_data(past_poses_np, future_poses_np)
    