import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D 

# --- CONFIGURATION ---
# LOG_FILE = 'exprs/online_inference/pose_log2.jsonl'
LOG_FILE = 'exprs/15fps/pose_log.jsonl'
# LOG_FILE = 'exprs/3dpw_single/pose_log.jsonl'
GAP_THRESHOLD = 0.9  # seconds


# MuPoTS 13-joint connections (Indices match your 13-joint array)
# 0:Pelvis, 1:RHip, 2:LHip, 3:RKnee, 4:LKnee, 5:RAnk, 6:LAnk, 
# 7:Spine, 8:Neck, 9:RSho, 10:LSho, 11:RElb, 12:LElb
# Connectivity for your specific 13-joint list
SKELETON_EDGES = [
    # Legs
    (0, 2), (2, 4),       # LHip -> LKnee -> LAnk
    (1, 3), (3, 5),       # RHip -> RKnee -> RAnk
    (0, 1),               # LHip -> RHip (Pelvis connector)

    # Arms
    (7, 9), (9, 11),      # LSho -> LElb -> LHand
    (8, 10), (10, 12),    # RSho -> RElb -> RHand
    (7, 8),               # LSho -> RSho (Shoulder connector)

    # Torso (Connecting Hips to Shoulders)
    (0, 7),               # LHip -> LSho
    (1, 8),               # RHip -> RSho
    
    # Head (Connecting Head to Shoulders)
    (6, 7), (6, 8)        # Head -> LSho & Head -> RSho (Triangle Neck)
]

JOINT_NAMES = {
    0: "Lhip", 1: "RHip", 2: "Lknee", 3: "RKnee", 4: "LAnk", 
    5: "RAnk", 6: "Head", 7: "Lsho", 8: "Rsho", 9: "Lelb", 
    10: "Relb", 11: "Lhand", 12: "Rhand"
}
def load_log_data(filepath):
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                pass # Skip broken lines
    return data

def set_consistent_axes(ax, all_data_np):
    """
    ax: The matplotlib 3D axis
    all_data_np: A numpy array containing ALL frames you want to plot 
                 Shape: (Total_Frames, Num_People, 13, 3)
    """
    # Flatten data to list of coordinates
    # We care about the global Min/Max of X, Y, Z across the whole file
    all_x = all_data_np[..., 0].flatten()
    all_y = all_data_np[..., 2].flatten() # ZED Z (Depth)
    all_z = all_data_np[..., 1].flatten() # ZED Y (Height)

    # 1. Find the centers
    mid_x = (all_x.max() + all_x.min()) * 0.5
    mid_y = (all_y.max() + all_y.min()) * 0.5
    mid_z = (all_z.max() + all_z.min()) * 0.5

    # 2. Find the maximum range to make a CUBE
    max_range = max(
        all_x.max() - all_x.min(),
        all_y.max() - all_y.min(),
        all_z.max() - all_z.min()
    )
    
    # Add some padding (e.g. 10%)
    half_size = (max_range * 0.5) * 1.1

    # 3. Set Limits centered on the data
    ax.set_xlim(mid_x - half_size, mid_x + half_size)
    ax.set_ylim(mid_y - half_size, mid_y + half_size)
    ax.set_zlim(mid_z - half_size, mid_z + half_size)
    
    # 4. FORCE EQUAL ASPECT RATIO (Crucial for 3D)
    # This prevents the person from looking stretched
    ax.set_box_aspect([1, 1, 1])


def compute_mpjpe(predicted, ground_truth):
    """
    predicted: numpy array of shape (Num_Frames, 13, 3)
    ground_truth: numpy array of shape (Num_Frames, 13, 3)
    Calculates the error of joints at specific indices only.
    """
    # print(f"predicted shape: {predicted.shape}, ground_truth shape: {ground_truth.shape}")
    assert predicted.shape == ground_truth.shape, "Shape mismatch between predicted and ground truth"
    
    # Compute Euclidean distances per joint per frame
    diffs = predicted - ground_truth
    dists = np.linalg.norm(diffs, axis=-1)  # Shape: (Num_Frames, 13)
    
    # Average over all joints and frames
    mpjpe = np.mean(dists)
    return mpjpe

def compute_avg_mpjpe(predicted_data, ground_truth_data):
    """
    predicted_data: numpy array of shape (Num_People, Num_Frames, 13, 3)
    ground_truth_data: numpy array of shape (Num_People, Num_Frames, 13, 3)
    Computes average MPJPE across all people.
    """
    num_people = predicted_data.shape[0]
    total_mpjpe = 0.0
    
    for person_idx in range(num_people):
        pred = predicted_data[person_idx]
        gt = ground_truth_data[person_idx]
        mpjpe = compute_mpjpe(pred, gt)
        total_mpjpe += mpjpe
    
    avg_mpjpe = total_mpjpe / num_people
    return avg_mpjpe


# --- MAIN ---
if __name__ == "__main__":
    data = load_log_data(LOG_FILE)
    # raw_data = load_log_data(RAW_DATA)
    print(f"Loaded {len(data)} log entries.")
    
    ##For Debugging plot just single frame
    # if len(data) > 0:
    #     current_data = data[0] # Just look at the first log entry
        
    #     # Get the MOST RECENT frame of history (Index -1)
    #     # Shape: (Num_People, 16, 13, 3) -> Select Person 0, Last Frame
    #     past_data = np.array(current_data["input_frames"])
        
    #     # Handle missing person dimension if it exists
    #     if past_data.ndim == 3:
    #         past_data = past_data[:, np.newaxis, :, :]

    #     # Select: Last Frame (-1), First Person (0)
    #     # Shape: (13, 3)
    #     current_pose = past_data[-1, 0, :, :]

    #     fig = plt.figure(figsize=(10, 8))
    #     ax = fig.add_subplot(111, projection='3d')

    #     xs = current_pose[:, 0]
    #     ys = current_pose[:, 1]
    #     zs = current_pose[:, 2]

    #     # 1. Plot Joints (Red)
    #     ax.scatter(xs, zs, ys, c='r', s=50, depthshade=False)

    #     # 2. Plot Bones (Blue Lines)
    #     for p1, p2 in SKELETON_EDGES:
    #         x_line = [xs[p1], xs[p2]]
    #         y_line = [ys[p1], ys[p2]] # ZED Y -> Plot Z (Up)
    #         z_line = [zs[p1], zs[p2]] # ZED Z -> Plot Y (Depth)
            
    #         # Swap Y and Z for Matplotlib plotting preference
    #         ax.plot(x_line, z_line, y_line, c='blue', linewidth=2)

    #     # 3. Label Joints (Text)
    #     # This is crucial! Check if "0" is at the hips and "8" is at the neck.
    #     for i in range(len(xs)):
    #         label = f"{i}: {JOINT_NAMES.get(i, '?')}"
    #         ax.text(xs[i], zs[i], ys[i], label, fontsize=9)

    #     ax.set_xlabel('X (Lateral)')
    #     ax.set_ylabel('Z (Depth)')
    #     ax.set_zlabel('Y (Height)')
    #     ax.set_title('Debug: Single Frame Skeleton')
        
    #     # Force equal aspect ratio so the human doesn't look stretched
    #     ax.set_box_aspect([1,1,1]) 

    #     plt.show()


    #choose instance to study
    print(len(data))
    for i in range(len(data)-1):
        time_diff = data[i+1]['timestamp'] - data[i]['timestamp']
        if time_diff > GAP_THRESHOLD:
            print(f"Large gap of {time_diff:.2f}s between frames {i} and {i+1}")
    #Find the intervals of data that are continuous 
    frame = 19
    future = 14

    current_data =data[frame]
    future_data = data[frame+future]
    highlight_pred = 0 #frame to study
    error_frames=[0,4,13] #frames to compute error on

    past_data= np.array(current_data["input_frames"])
    predicted_data = np.array(current_data["prediction_frames"])
    truth_data = np.array(future_data["input_frames"])

    print(f"past = {past_data[0][future][0]}")
    print(f"truth = {truth_data[0][0][0]}")
    # print(f"past_data = {past_data.shape}")

    #reshape into correct format (to handle 1 person case)
    if predicted_data.ndim ==3:
        predicted_data = predicted_data[np.newaxis,:,:,:]
        past_data = past_data[np.newaxis,:,:,:]
        print(f"fixed predicted shape into {predicted_data.shape}")
    
    if truth_data.ndim ==3:
        truth_data = truth_data[np.newaxis,:,:,:]
    
    num_past = past_data.shape[1]
    num_predicted = predicted_data.shape[1]
    people = predicted_data.shape[0]

    fig = plt.figure(figsize=(14,8))
    ax = fig.add_subplot(111,projection='3d')

    error_summary = []
    for person_idx in range(people):
        #Scatter each past frame
        for time in range(num_past):
            alpha = 0.1
            pose = past_data[person_idx,time,:,:]
            xs,ys,zs = pose[:,0], pose[:,1] , pose[:,2]
            ax.scatter(xs,zs,ys, c='blue', alpha=alpha, s=10)
            for p1, p2 in SKELETON_EDGES:
                x_line = [xs[p1], xs[p2]]
                y_line = [ys[p1], ys[p2]] # ZED Y -> Plot Z (Up)
                z_line = [zs[p1], zs[p2]] # ZED Z -> Plot Y (Depth)
                
                if time == future:
                    ax.plot(x_line, z_line, y_line, c='blue', alpha=1, linewidth=1)
                else:
                    ax.plot(x_line, z_line, y_line, c='blue', alpha=alpha, linewidth=1)
                            
        # #Now scatter predictions
        for time in range(num_predicted):
                alpha = 0.10
                pose = predicted_data[person_idx,time,:,:]
                xs, ys, zs = pose[:, 0], pose[:, 1], pose[:, 2]
                for p1,p2 in SKELETON_EDGES:
                    x_line = [xs[p1], xs[p2]]
                    y_line = [ys[p1], ys[p2]] # ZED Y -> Plot Z (Up)
                    z_line = [zs[p1], zs[p2]] # ZED Z -> Plot Y (Depth)

                    if time == highlight_pred:
                        ax.plot(x_line, z_line, y_line, c='red', alpha=1, linewidth=1)
                    else:
                        ax.plot(x_line, z_line, y_line, c='red', alpha=alpha, linewidth=1)

        #Scatter the true future frames
        for time in range(num_predicted):
            alpha = 0.10
            pose = truth_data[person_idx,time,:,:]
            xs, ys, zs = pose[:, 0], pose[:, 1], pose[:, 2]
            for p1,p2 in SKELETON_EDGES:
                x_line = [xs[p1], xs[p2]]
                y_line = [ys[p1], ys[p2]] # ZED Y -> Plot Z (Up)
                z_line = [zs[p1], zs[p2]] # ZED Z -> Plot Y (Depth)

                if time == highlight_pred:
                    ax.plot(x_line, z_line, y_line, c='green', alpha=1, linewidth=1)
                else:
                    ax.plot(x_line, z_line, y_line, c='green', alpha=alpha, linewidth=1)
        
        #compute error

        for err_frame in error_frames:
            mpjpe = compute_mpjpe(predicted_data[person_idx, err_frame], truth_data[person_idx, err_frame])
            error_summary.append(f"P: {person_idx} T+{err_frame+1}: {mpjpe:.3f}m")

    # Join into one string
    info_text = f"Prediction Errors (MPJPE):\n" + "\n".join(error_summary)
    # Place text in 2D coordinates (0,0 is bottom-left, 1,1 is top-right)
    # transform=ax.transAxes makes it relative to the axes box
    # ax.text2D(0.00, 0.2, info_text, transform=ax.transAxes, 
    #           color='black', bbox=dict(facecolor='white', alpha=0.7))
    fig.text(0.75, 0.60, info_text, 
             fontsize=10, 
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    all_frames = np.concatenate((past_data,predicted_data, truth_data), axis=1)
    set_consistent_axes(ax,all_frames)

    ax.set_xlabel('X (Lateral)')
    ax.set_ylabel('Z (Depth)')
    ax.set_zlabel('Y (Height)')
    ax.set_title(f'Studying frame {frame} {num_past} Past Frames + {num_predicted} Predicted Frames with prediction #{highlight_pred} Highlighted')
    legend_elements = [
    Line2D([0], [0], color='blue', lw=2, label='Past Frame (History)'),
    Line2D([0], [0], color='green', lw=2, label='Ground Truth (Real)'),
    Line2D([0], [0], color='red', lw=2, label='Prediction (Model)')
    ]

    # Add the legend to the plot
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 0.8), borderaxespad=0.)

    plt.show()

