import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R


class CamParams:
    def __init__(self, K, R, t, frame_id=None):
        """Construct camera parameter object

        Args:
            K (3x3 matrix): Camera intrinsics
            R (3x3 matrix): Camera rotation
            t (3x1 vector): Camera translation
            frame_id: Reference frame id
        """
        self.K = K
        self.R = R
        self.t = t.reshape(3, 1)
        # Affine Transform
        self.A = np.column_stack((self.R.T, - self.R.T @ self.t))
        # Projection Matrix
        self.P = K @ self.A
        self.frame_id = frame_id


def read_cam_params(data, stream_id, cam_name, ref_frame):
    """Read camera parameters from json data

    Args:
        data (dict): Json data read from file
        stream_id (str): Data stream id, e.g. "113bdf9278314c72b7f8988a67f0dff9".
        cam_name (str): Camera name, e.g. "CAM_FRONT".
        ref_frame (str): Reference frame.
    """
    if ref_frame == "ego":
        sensor2ref_t = data[stream_id]["cams"][cam_name]["sensor2ego_translation"]
        # sensor2ref_t = [0, 0, 0]
        sensor2ref_q = data[stream_id]["cams"][cam_name]["sensor2ego_rotation"]
        sensor2ref_R = R.from_quat(sensor2ref_q, scalar_first=True).as_matrix()
    elif ref_frame == "lidar":
        sensor2ref_t = data[stream_id]["cams"][cam_name]["sensor2lidar_translation"]
        sensor2ref_R = data[stream_id]["cams"][cam_name]["sensor2lidar_rotation"]
    else:
        print(f"WARNING: Reference frame '{ref_frame}' is undefined.")
        return None

    cam_intrinsic = data[stream_id]["cams"][cam_name]["cam_intrinsic"]
    return CamParams(np.array(cam_intrinsic), np.array(sensor2ref_R), np.array(sensor2ref_t), ref_frame)


def proj_3d_point(point, cam_params):
    point = np.append(point, 1)
    # print(cam_params.A @ point)
    point = cam_params.P @ point
    if point[2] > 0:
        point = point[:2]/point[2]
        point = point.astype(np.int16)
        return point
    else:
        return None  # Point is behind the camera


def draw_3d_pts(img, point, cam_params, color):
    """Take in 3d points and plot them on image as red circles
    """
    point = proj_3d_point(point, cam_params)
    if point is not None:
        cv2.circle(img, (point[0], point[1]), 3, color, thickness=-1)


def draw_3d_bbox(img, bbox, cam_params, color=(0, 255, 0), thickness=2):
    """Draw a 3D bounding box on the image.


    Args:
        img (cv2.image): Camera image.
        bbox (list): The bounding box in the format [center_x, center_y, center_z, dim_x, dim_y, dim_z, rot_deg].
        cam_params (CamParams): Camera paremeters object.
        color (tuple, optional): _description_. Defaults to (0, 255, 0).
        thickness (int, optional): _description_. Defaults to 2.

    Returns:
        _type_: _description_
    """

    # Extract bounding box parameters
    center = np.array(bbox[:3])
    dim = np.array(bbox[3:6])
    rot_deg = bbox[6]

    if proj_3d_point(center, cam_params) is None:
        print("WARNING: The bounding box center is behind the camera!")
        return

    # Compute the rotation matrix around the Y-axis
    rot_rad = np.deg2rad(rot_deg)
    cos_r, sin_r = np.cos(rot_rad), np.sin(rot_rad)
    R = np.array([[cos_r, 0, sin_r], [0, 1, 0], [-sin_r, 0, cos_r]])

    # Define the 8 corners of the bounding box in its local coordinate system
    dx, dy, dz = dim / 2
    corners = np.array([
        [dx, dy, dz],
        [-dx, dy, dz],
        [-dx, -dy, dz],
        [dx, -dy, dz],
        [dx, dy, -dz],
        [-dx, dy, -dz],
        [-dx, -dy, -dz],
        [dx, -dy, -dz]
    ])

    # Rotate and translate the corners to the bounding box center
    rotated_corners = (R @ corners.T).T + center

    # Project the 3D corners to image
    projected_corners = []
    for corner in rotated_corners:
        _pt = proj_3d_point(corner, cam_params)
        if _pt is None:
            print("WARNING: The bounding box corner is behind the camera!")
            return
        projected_corners.append(_pt)
    projected_corners = np.array(projected_corners, dtype=int)

    # Define the edges of a 3D box in terms of the corner indices
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # front face
        (4, 5), (5, 6), (6, 7), (7, 4),  # back face
        (0, 4), (1, 5), (2, 6), (3, 7)   # connecting edges
    ]

    # Draw the edges on the image
    for start, end in edges:
        pt1 = tuple(projected_corners[start])
        pt2 = tuple(projected_corners[end])
        cv2.line(img, pt1, pt2, color, thickness)

    return img


def draw_traj(img, traj, cam_params, color=(0, 255, 0), car_width=-1, car_length=-1, thickness=2):
    projected_traj = []
    traj_3d = []
    polygon_sides = [[], []]
    for i, pt in enumerate(traj):
        pt3d_corrected = np.array([pt[1], -pt[0], 0])
        traj_3d.append(pt3d_corrected)
        _pt = proj_3d_point(pt3d_corrected, cam_params)
        if _pt is not None:
            projected_traj.append(_pt)
            cv2.circle(img, (_pt[0], _pt[1]), thickness*3, color, thickness=-1)

            if car_width > 0:
                n = traj_3d[i] - \
                    traj_3d[i-1] if i > 0 else np.array([1.0, 0])
                n = n / np.linalg.norm(n)
                n_rt = np.array([n[1], -n[0], 0])
                polygon_sides[0].append(proj_3d_point(
                    pt3d_corrected + n_rt * car_width / 2.0, cam_params))
                polygon_sides[1].append(proj_3d_point(
                    pt3d_corrected - n_rt * car_width / 2.0, cam_params))
                if i == len(traj) - 1 and car_length > 0:
                    polygon_sides[0].append(proj_3d_point(
                        pt3d_corrected + n_rt * car_width / 2.0 + n * car_length / 2, cam_params))
                    polygon_sides[1].append(proj_3d_point(
                        pt3d_corrected - n_rt * car_width / 2.0 + n * car_length / 2, cam_params))
        else:
            print("WARNING: Traj point behind!")

    projected_traj = np.array(projected_traj, dtype=int)

    for i in range(len(projected_traj)-1):
        cv2.line(img, tuple(projected_traj[i]), tuple(
            projected_traj[i+1]), color, thickness)

    if car_width > 0:
        occupied_polygon = np.array(
            polygon_sides[0] + list(reversed(polygon_sides[1])), dtype=int)
        frame = np.zeros_like(img)
        cv2.fillPoly(frame, [occupied_polygon], color)
        alpha = 0.5
        mask = frame.astype(bool)
        img[mask] = cv2.addWeighted(img, alpha, frame, 1 - alpha, 0)[mask]

    return img


if __name__ == "__main__":
    import json
    import os

    car_width = 1.73  # Renault Zoe
    car_length = 4.084  # Renault Zoe

    current_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(current_dir, "samples", "output.json")
    img_path = os.path.join(current_dir, "samples", "CAM_FRONT.jpg")
    img = cv2.imread(img_path)

    with open(json_path, 'r') as file:
        data = json.load(file)
        stream_id = list(data.keys())[0]
        cam_lidar_params = read_cam_params(
            data, stream_id, "CAM_FRONT", "lidar")
        bboxes = data[stream_id]["gt_boxes"] # right, up, front 
        bboxes_names = data[stream_id]["gt_names"]

        for i, bbox in enumerate(bboxes):
            if proj_3d_point(bbox[:3], cam_lidar_params) is None:
                continue
            if bboxes_names[i] == "movable_object.trafficcone":
                draw_3d_bbox(img, bbox, cam_lidar_params, color=(0, 165, 255))
            if bboxes_names[i] == "vehicle.car":
                draw_3d_bbox(img, bbox, cam_lidar_params, color=(0, 0, 255))

        cam_ego_params = read_cam_params(data, stream_id, "CAM_FRONT", "ego")
        traj_future = data[stream_id]["gt_ego_fut_trajs"]
        draw_traj(img, traj_future, cam_ego_params,
                  car_width=car_width, car_length=car_length)
        cv2.imwrite(os.path.join(current_dir, "output.png"), img)
