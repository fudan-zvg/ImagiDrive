import os
import glob
import argparse
from tqdm import tqdm
import pickle
import json

import cv2
import numpy as np
from PIL import Image

from bev_render import BEVRender
from cam_render import CamRender

plot_choices = dict(
    draw_pred = True, # True: draw gt and pred; False: only draw gt
    det = True,
    track = True, # True: draw history tracked boxes
    motion = True,
    map = True,
    planning = True,
)
START = 0
END = 544
INTERVAL = 1

token_list = [
    # 'ebd128c8164943538f360f11c10e26d5',
    # 'c06a5e8ca3694889a25d3143d4dca9d5',
    # 'c8151359de4e4c76aab4e3e73a54fcfe',
    # '8a75ea80dd4646d7bf2775ccfd012db6',
    # '6060fa4c3b1e443994967da28e86132b',
    # "d7f6efdc518b48b9aae17102636a6921",
    # "12db2c7ea88e4708b422985835988fc7"


    # "ebd128c8164943538f360f11c10e26d5"
    # "b16c13a33031451e97116a3149783fa9",
    # "34346fc6dbc54931913847339b48abc2"
    # "b731e3bfd7b64247b9d9708e67847933"
    # "50e62add1dd7476c9ab223655ffb9f5e"
    # "493effd2c5cb494792b5f1c64cb01050"

    # supp
    # "7c21c98a10a04997975b8d527e15bc81"
    # "e31dad83431d497eb58c285b85453a40"
    # "f4550267cd0240e1a1ceb844e33e97d4"

    # failure
    "fa0391dcc3dd4dbbaa29cbafbea7bbf0"


]
class Visualizer:
    def __init__(
        self,
        args,
        plot_choices,
    ):
        self.out_dir = args.out_dir
        self.combine_dir = os.path.join(self.out_dir, 'combine')
        os.makedirs(self.combine_dir, exist_ok=True)
        
        # cfg = Config.fromfile(args.config)
        # self.dataset = build_dataset(cfg.data.val)
        # self.results = mmcv.load(args.result_path)/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/lijingyu-240108540149/vla_gen/vlm_intervl/data/nuscenes_infos_val_hrad_planing_scene.pkl
        data_path = "/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/lijingyu-240108540149/vla_gen/pipeline/visulization/nuscenes_infos_val.pkl"
        json_path = "/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/lijingyu-240108540149/vla_gen/pipeline/v1_selected_merged.json"
        self.data = pickle.load(open(data_path, 'rb'))['infos']

        self.search_map = {str(item["token"]): i for i, item in enumerate(self.data)}
        with open(json_path, "r")as f:
            json_data = json.load(f)
        results = list()
        for i , item in enumerate(json_data):
            # cur = dict()
            # cur['token'] = item['token']
            # cur['base_traj'] = item['base_traj']
            results.append(item)
        self.results = results
        self.bev_render = BEVRender(plot_choices, self.out_dir)
        self.cam_render = CamRender(plot_choices, self.out_dir)

    def add_vis(self, index):
        #index 理解为宜json为基准
        data = self.data[self.search_map[self.results[index]['token']]]
        if self.results[index]['token'] not in token_list:
            
            return
        result = self.results[index]
        print(index, self.results[index]['token'])

        bev_gt_path, bev_pred_path = self.bev_render.render(data, result, index)
        # cam_pred_path = self.cam_render.render(data, result, index)
        # self.combine(bev_gt_path, bev_pred_path, cam_pred_path, index)
    
    def combine(self, bev_gt_path, bev_pred_path, cam_pred_path, index):
        bev_gt = cv2.imread(bev_gt_path)
        bev_image = cv2.imread(bev_pred_path)
        cam_image = cv2.imread(cam_pred_path)
        merge_image = cv2.hconcat([cam_image, bev_image, bev_gt])
        save_path = os.path.join(self.combine_dir, str(index).zfill(4) + '.jpg')
        cv2.imwrite(save_path, merge_image)

    def image2video(self, fps=12, downsample=4):
        imgs_path = glob.glob(os.path.join(self.combine_dir, '*.jpg'))
        imgs_path = sorted(imgs_path)
        img_array = []
        for img_path in tqdm(imgs_path):
            img = cv2.imread(img_path)
            height, width, channel = img.shape
            img = cv2.resize(img, (width//downsample, height //
                             downsample), interpolation=cv2.INTER_AREA)
            height, width, channel = img.shape
            size = (width, height)
            img_array.append(img)
        out_path = os.path.join(self.out_dir, 'video.mp4')
        out = cv2.VideoWriter(
            out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()


def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize groundtruth and results')
    parser.add_argument('--config', default=None, help='config file path')
    parser.add_argument('--result-path', 
        default=None,
        help='prediction result to visualize'
        'If submission file is not provided, only gt will be visualized')
    parser.add_argument(
        '--out-dir', 
        default='vis',
        help='directory where visualize results will be saved')
    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    visualizer = Visualizer(args, plot_choices)

    for idx in tqdm(range(START, END, INTERVAL)):
        if idx > len(visualizer.results):
            break
        visualizer.add_vis(idx)
    
    visualizer.image2video()

if __name__ == '__main__':
    main()