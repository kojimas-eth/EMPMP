import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- CONFIGURATION ---
LOG_FILE = 'exprs/online_inference/pose_log2.jsonl'
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

# --- MAIN ---
if __name__ == "__main__":
    data = load_log_data(LOG_FILE)
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
    frame = 26
    current_data =data[frame]
    future_data = data[frame + 10]

    past_data= np.array(current_data["input_frames"])
    predicted_data = np.array(current_data["prediction_frames"])
    truth_data = np.array(future_data["input_frames"])

    print(f"past_data = {past_data.shape}")

    #reshape into correct format (to handle 1 person case)
    if predicted_data.ndim ==3:
        predicted_data = predicted_data[np.newaxis,:,:,:]
        past_data = past_data[np.newaxis,:,:,:]
        print(f"fixed predicted shape into {predicted_data.shape}")
    
    if truth_data.ndim ==3:
        truth_data = truth_data[np.newaxis,:,:,:]
    
    num_past = past_data.shape[1]
    num_predicted = predicted_data.shape[1]
    person_idx = 0

    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111,projection='3d')

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
            
            if time == num_past -1:
                ax.plot(x_line, z_line, y_line, c='blue', alpha=1, linewidth=1)
            else:
                ax.plot(x_line, z_line, y_line, c='blue', alpha=alpha, linewidth=1)
                         
    # #Now scatter predictions
    for time in range(num_predicted):
            alpha = 0.1
            pose = predicted_data[person_idx,time,:,:]
            xs, ys, zs = pose[:, 0], pose[:, 1], pose[:, 2]
            for p1,p2 in SKELETON_EDGES:
                x_line = [xs[p1], xs[p2]]
                y_line = [ys[p1], ys[p2]] # ZED Y -> Plot Z (Up)
                z_line = [zs[p1], zs[p2]] # ZED Z -> Plot Y (Depth)

                if time == num_predicted -1:
                    ax.plot(x_line, z_line, y_line, c='red', alpha=1, linewidth=1)
                else:
                    ax.plot(x_line, z_line, y_line, c='red', alpha=alpha, linewidth=1)
    
    for time in range(num_predicted):
        alpha = 0.1
        pose = truth_data[person_idx,time,:,:]
        xs, ys, zs = pose[:, 0], pose[:, 1], pose[:, 2]
        for p1,p2 in SKELETON_EDGES:
            x_line = [xs[p1], xs[p2]]
            y_line = [ys[p1], ys[p2]] # ZED Y -> Plot Z (Up)
            z_line = [zs[p1], zs[p2]] # ZED Z -> Plot Y (Depth)

            if time == num_predicted -1:
                ax.plot(x_line, z_line, y_line, c='green', alpha=1, linewidth=1)
            else:
                ax.plot(x_line, z_line, y_line, c='green', alpha=alpha, linewidth=1)


    all_frames = np.concatenate((past_data,predicted_data, truth_data), axis=1)
    set_consistent_axes(ax,all_frames)

    ax.set_xlabel('X (Lateral)')
    ax.set_ylabel('Z (Depth)')
    ax.set_zlabel('Y (Height)')
    ax.set_title(f'Trajectory: {num_past} Past Frames + {num_predicted} Predicted Frames')
    ax.legend()

    plt.show()

