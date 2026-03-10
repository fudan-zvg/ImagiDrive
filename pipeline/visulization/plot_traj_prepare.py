import os
import json
import cv2
import pickle
import numpy as np
from tqdm import tqdm
from plot_traj import draw_traj, read_cam_params,CamParams
from scipy.spatial.transform import Rotation as R

def read_cam_params(data, stream_id, cam_name, ref_frame):
    """Read camera parameters from json data

    Args:
        data (dict): Json data read from file
        stream_id (str): Data stream id, e.g. "113bdf9278314c72b7f8988a67f0dff9".
        cam_name (str): Camera name, e.g. "CAM_FRONT".
        ref_frame (str): Reference frame.
    """
    if ref_frame == "ego":
        sensor2ref_t = data["cams"][cam_name]["sensor2ego_translation"]
        # sensor2ref_t = [0, 0, 0]
        sensor2ref_q = data["cams"][cam_name]["sensor2ego_rotation"]
        sensor2ref_R = R.from_quat(sensor2ref_q, scalar_first=True).as_matrix()
    elif ref_frame == "lidar":
        sensor2ref_t = data["cams"][cam_name]["sensor2lidar_translation"]
        sensor2ref_R = data["cams"][cam_name]["sensor2lidar_rotation"]
    else:
        print(f"WARNING: Reference frame '{ref_frame}' is undefined.")
        return None

    cam_intrinsic = data["cams"][cam_name]["cam_intrinsic"]
    return CamParams(np.array(cam_intrinsic), np.array(sensor2ref_R), np.array(sensor2ref_t), ref_frame)

def process_data(data_path, json_path, image_path):
    data = pickle.load(open(data_path, 'rb'))
    search_map = {str(item["token"]): i for i, item in enumerate(data)}
    with open(json_path, "r")as f:
        json_data = json.load(f)
    for i, item in tqdm(enumerate(json_data), total=len(json_data)):
        token = item["token"]
        json_idx = item['data_id'] 
        data_id = search_map[token]
        info = data[data_id]
        image = cv2.imread(os.path.join(image_path, f"NUSCENES_{json_idx:06}_{0:04}.png"))
        image = cv2.resize(image, (1600, 900))
        plot_traj(info, token, image, item, json_idx)
    return

def plot_traj(info, token, original_img, item,json_idx):
    current_dir = "/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/lijingyu-240108540149/vla_gen/pipeline/vis_traj/"
    car_width = 0  # Renault Zoe
    car_length = 4.084  # Renault Zoe
    cam_ego_params = read_cam_params(info, token, "CAM_FRONT", "ego")
    gt_traj = np.array(item['gt']).reshape(6,2)
    img = original_img.copy()
    img = draw_traj(img, gt_traj, cam_ego_params,
                    car_width=car_width, car_length=car_length)
    cv2.imwrite(os.path.join(current_dir, f"NUSCENES_{json_idx:06}_gt.png"), img)
    base_traj = np.array(item['base_traj']).reshape(6,2)
    img = original_img.copy()
    img = draw_traj(img, base_traj, cam_ego_params,
                    car_width=car_width, car_length=car_length)
    cv2.imwrite(os.path.join(current_dir, f"NUSCENES_{json_idx:06}_base.png"), img)
    for i in range(5):
        traj_pred = list(np.array(item[f'traj_pred_{i}']).reshape(6,2))
        img = original_img.copy()
        img = draw_traj(img, traj_pred, cam_ego_params,
                    car_width=car_width, car_length=car_length)
        cv2.imwrite(os.path.join(current_dir, f"NUSCENES_{json_idx:06}_{i+1:04}.png"), img)
        
    return

if __name__ =="__main__":
    data_path = "/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/lijingyu-240108540149/vla_gen/vlm_intervl/data/nuscenes_infos_temporal_val_v1.pkl"
    json_path = "/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/lijingyu-240108540149/vla_gen/pipeline/v2_selected_merged.json"
    image_path = "/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/lijingyu-240108540149/vla_gen/pipeline/outputs/virtual_test/images"
    process_data(data_path, json_path, image_path)