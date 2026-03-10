import os
import json
import torch
from multiprocessing import Process
import multiprocessing as mp
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
import numpy as np
from torchvision.transforms.functional import InterpolationMode
from vlm_intervl.internvl.model.internvl_chat import (InternVisionConfig,
                                          InternVisionModel,
                                          InternVLChatConfig,
                                          InternVLChatModel)
from vlm_intervl.internvl.inference.inference_extratoken import load_image

class InterVLPredictor(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.model, self.tokenizer = self.load_model(cfg)
        self.rank = cfg.rank
        
    def load_model(self, cfg):
        self.config = InternVLChatConfig.from_pretrained(cfg.model_path)
        self.config.output_hidden_states = True
        if cfg.is_train:
            model = InternVLChatModel.from_pretrained(
                cfg.model_path,
                torch_dtype=torch.float16,
                device_map=cfg.rank,
                config=self.config
            )
        else:
            model = InternVLChatModel.from_pretrained(
                cfg.model_path,
                torch_dtype=torch.float16,
                device_map=cfg.rank,
                config=self.config
            ).eval()
        print("******Finished initializing Model *****")
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_path,
            trust_remote_code=True,
            use_fast=False
        )
        print("******Finished initializing Tokenizer *****")
        return model, tokenizer

    def test(self, item):
        dtype = torch.float16
        id = item["id"]
        token = item["token"]
        image_paths = item['image'][0]
        # question = "<image>\n<ego_state>\n Please describe the scenario."+ "<traj_query>" * 6
        # question = item['conversations'][0]["value"] + "<traj_query>" * 6
        nav_cmd = item['nav_cmd']
        if nav_cmd == 0:
            command_text = "Turn Right"
        elif nav_cmd == 1:
            command_text = "Turn Left"
        else:
            command_text = "Go Straight"
        question = (
                "<image>\n" +
                "<ego_state>\n" +
                f"The command is {command_text}. "
                "This is the current frame."
                "Predict the waypoints of ego vehicle for the next 3 seconds based on the current frame and command. "
                "Do not generate any textual response."
            )
        question = question + "<traj_query>" * 6
        ego_states = torch.tensor(item['gt_vel'],dtype=dtype).to(f"cuda:{self.rank}")
        gt = np.array(item["gt_traj"]).reshape(6,2)

        # 加载图像
        pixel_values_list = []
        if type(image_paths) == list:
            for image_path in image_paths:
                image_path = "/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/lijingyu-240108540149/vla_gen/vlm_intervl/data/nuscenes/" +  image_path
                pixel_values = load_image(image_path, max_num=12).to(dtype).to(f"cuda:{self.rank}")
                pixel_values_list.append(pixel_values)
        else:
            image_path = "/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/lijingyu-240108540149/vla_gen/vlm_intervl/data/nuscenes/" +  image_paths
            pixel_values = load_image(image_path, max_num=12).to(dtype).to(f"cuda:{self.rank}")
            pixel_values_list.append(pixel_values)
        

        # 合并多个图像的特征
        pixel_values = torch.cat(pixel_values_list, dim=0)
        num_patches_list = [pv.size(0) for pv in pixel_values_list]

        # 推理
        generation_config = dict(max_new_tokens=4096, do_sample=False, num_beams=1, pad_token_id=self.tokenizer.eos_token_id)
        response, traj_pred, traj_score = self.model.chat(
            self.tokenizer,
            pixel_values,
            question,
            generation_config,
            num_patches_list=num_patches_list,
            ego_states = ego_states
        )

        # 保存结果
        results = {
            'id': id,
            'token': token,
            'question': question,
            # 'traj_score': traj_score.to(torch.float32).cpu().numpy(),
            'gt': gt,
            'traj_pred': traj_pred.to(torch.float32).cpu().numpy()[0]
        }
        return results
    
    def test_v1(self, item, image_paths):
        dtype = torch.float16
        id = item["id"]
        token = item["token"]
        # image_paths = item['image']
        image_path_0 = item['image'][0]
        image_paths.insert(0,image_path_0)
        nav_cmd = item['nav_cmd']
        if nav_cmd == 0:
            command_text = "Turn Right"
        elif nav_cmd == 1:
            command_text = "Turn Left"
        else:
            command_text = "Go Straight"
        question = (
                "<image>\n" * 3 +
                "<ego_state>\n" +
                f"The command is {command_text}. "
                "There are three front-view images: the first is the current frame, and the second is a future frame. Predict the waypoints of ego vehicle for the next 3 seconds based on the frames and command. Use the future frame only for context and do not generate any textual response."
                # "Three front-view images are provided: the first is the current frame, and the others are future frames. "
                # "Predict the ego vehicle's waypoints for the next 3 seconds based on these images and the command. "
                # "Use future frames only for additional context. Do not generate any textual response."
            )

        question = question + "<traj_query>" * 6
        ego_states = torch.tensor(item['gt_vel'],dtype=dtype)
        

        # 加载图像
        pixel_values_list = []
        if type(image_paths) == list:
            for idx, image_path in enumerate(image_paths):
                # image_path = "/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/lijingyu-240108540149/vla_gen/vlm_intervl/data/nuscenes/" +  image_path
                if idx==0:
                    image_path = "/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/lijingyu-240108540149/vla_gen/vlm_intervl/data/nuscenes/" +  image_path
                else:
                    image_path = "/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/lijingyu-240108540149/vla_gen/pipeline/" +  image_path
                pixel_values = load_image(image_path, max_num=12).to(dtype).to('cuda')
                pixel_values_list.append(pixel_values)
        else:
            image_path = image_paths
            pixel_values = load_image(image_path, max_num=12).to(dtype).to('cuda')
            pixel_values_list.append(pixel_values)
        

        # 合并多个图像的特征
        pixel_values = torch.cat(pixel_values_list, dim=0)
        num_patches_list = [pv.size(0) for pv in pixel_values_list]

        # 推理
        generation_config = dict(max_new_tokens=4096, do_sample=False, num_beams=1, pad_token_id=self.tokenizer.eos_token_id)
        response, traj_pred, traj_score = self.model.chat(
            self.tokenizer,
            pixel_values,
            question,
            generation_config,
            num_patches_list=num_patches_list,
            ego_states = ego_states
        )
        

        # 保存结果
        results = {
            'id': id,
            'token': token,
            'question': question,
            # 'traj_score': traj_score.to(torch.float32).cpu().numpy(),
            'traj_pred': traj_pred.to(torch.float32).cpu().numpy()[0]
        }
        return results