import argparse
import json
import random

from pytorch_lightning import seed_everything
from torchvision import transforms
import torch.nn as nn

import Vista.init_proj_path
from Vista.sample_utils import *
from Vista.sample import load_img

VERSION2SPECS = {
    "vwm": {
        "config": "../Vista/configs/inference/vista.yaml",
        "ckpt": "../Vista/ckpts/vista.safetensors"
    }
}
DATASET2SOURCES = {
    "NUSCENES": {
        "data_root": "../Vista/data/nuscenes",
        "anno_file": "../Vista/data/nuScenes.json"
    },
    "IMG": {
        "data_root": "image_folder"
    }
}

class VistaGenerator(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        # set_lowvram_mode(cfg.low_vram)
        self.model = init_model(VERSION2SPECS["vwm"], rank = cfg.rank)
        self.unique_keys = set([x.input_key for x in self.model.conditioner.embedders])
        self.search_map = self.construct_search_map(cfg.gen_data_path)
        self.base_data_path = cfg.base_data_path
        self.config = cfg
        
        print("***** Finished initializing Vista *****")

    def test(self, item):
        if item['token'] not in self.search_map:
            return None 
        with open(DATASET2SOURCES['NUSCENES']["anno_file"], "r") as anno_json:
            all_samples = json.load(anno_json)
        token = item["token"]
        idx = self.search_map[token]
        sample_dict = all_samples[idx]
        assert token == sample_dict['tokens']                                                 
        frame_list  = self.get_frames(item,sample_dict)
        # action_dict = self.get_action(item, base_traj) if 'action_mode' in item else None
        action_dict = dict()
        action_dict["trajectory"] = torch.tensor(item['traj'].flatten())

        img_seq = list()
        for each_path in frame_list[-self.config.n_conds:]:
            img = load_img(each_path, 576, 1024)
            img_seq.append(img)
        images = torch.stack(img_seq)
        value_dict = init_embedder_options(self.unique_keys)
        cond_img = img_seq[0][None]
        value_dict["cond_frames_without_noise"] = cond_img
        value_dict["cond_aug"] = 0.0
        value_dict["cond_frames"] = cond_img + 0.0 * torch.randn_like(cond_img)
        if action_dict is not None:
            for key, value in action_dict.items():
                value_dict[key] = value

        sampler = init_sampling(guider='VanillaCFG', steps=50, cfg_scale=2.5, num_frames=25)

        uc_keys = ["cond_frames", "cond_frames_without_noise", "command", "trajectory", "speed", "angle", "goal"]
        out = self.do_sample(
            images,
            self.model,
            sampler,
            value_dict,
            num_rounds=1,
            num_frames=self.config.num_frames,
            force_uc_zero_embeddings=uc_keys,
            initial_cond_indices=[index for index in range(self.config.n_conds)]
        ) # samples, samples_z, inputs
        virtual_path = os.path.join("outputs", "virtual_test")
        # real_path = os.path.join("outputs", "real")
        perform_save_locally(virtual_path, out[0], "videos", "NUSCENES", item['id'])
        # perform_save_locally(real_path, images, "videos", "NUSCENES", item['id'])
        perform_save_locally(virtual_path, out[0][[self.config.n_conds-1,12,18]], "images", "NUSCENES", item['id'])
        image_paths = list()
        image_paths.append(os.path.join(virtual_path, "images", f"NUSCENES_{item['id']:06}_{1:04}.png"))
        image_paths.append(os.path.join(virtual_path, "images", f"NUSCENES_{item['id']:06}_{2:04}.png"))
        # image_path = os.path.join(virtual_path, "images", f"NUSCENES_{item['id']:06}_{0:04}.png")
        return out[0], image_paths
    
    #结合intervl——wrapper——v2
    def test_v2(self, item):
        if item['token'] not in self.search_map:
            return None 
        with open(DATASET2SOURCES['NUSCENES']["anno_file"], "r") as anno_json:
            all_samples = json.load(anno_json)
        token = item["token"]
        idx = self.search_map[token]
        sample_dict = all_samples[idx]
        assert token == sample_dict['tokens']                                                 
        frame_list  = self.get_frames(item,sample_dict)
        # action_dict = self.get_action(item, base_traj) if 'action_mode' in item else None
        action_dict = dict()
        action_dict["trajectory"] = torch.tensor(item['traj'].flatten())

        img_seq = list()
        for each_path in frame_list[-self.config.n_conds:]:
            img = load_img(each_path, 576, 1024)
            img_seq.append(img)
        images = torch.stack(img_seq)
        value_dict = init_embedder_options(self.unique_keys)
        cond_img = img_seq[0][None]
        value_dict["cond_frames_without_noise"] = cond_img
        value_dict["cond_aug"] = 0.0
        value_dict["cond_frames"] = cond_img + 0.0 * torch.randn_like(cond_img)
        if action_dict is not None:
            for key, value in action_dict.items():
                value_dict[key] = value

        sampler = init_sampling(guider='TrianglePredictionGuider', steps=50, cfg_scale=2.5, num_frames=25)

        uc_keys = ["cond_frames", "cond_frames_without_noise", "command", "trajectory", "speed", "angle", "goal"]
        out = self.do_sample(
            images,
            self.model,
            sampler,
            value_dict,
            num_rounds=1,
            num_frames=self.config.num_frames,
            force_uc_zero_embeddings=uc_keys,
            initial_cond_indices=[index for index in range(self.config.n_conds)]
        ) # samples, samples_z, inputs
        virtual_path = os.path.join("outputs", "virtual_test")
        # real_path = os.path.join("outputs", "real")
        perform_save_locally(virtual_path, out[0], "videos", "NUSCENES", item['id'])
        # perform_save_locally(real_path, images, "videos", "NUSCENES", item['id'])
        perform_save_locally(virtual_path, out[0][[self.config.n_conds-1,8,14]], "images", "NUSCENES", item['id'])
        image_paths = list()
        image_paths.append(os.path.join(virtual_path, "images", f"NUSCENES_{item['id']:06}_{1:04}.png"))
        image_paths.append(os.path.join(virtual_path, "images", f"NUSCENES_{item['id']:06}_{2:04}.png"))
        # image_path = os.path.join(virtual_path, "images", f"NUSCENES_{item['id']:06}_{0:04}.png")
        return out[0], image_paths
    
    def test_test(self, item):
        if item['token'] not in self.search_map:
            return None 
        with open("/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/lijingyu-240108540149/vla_gen/Vista/data/nuScenes_val_back.json", "r") as anno_json:
            all_samples = json.load(anno_json)
        token = item["token"]
        idx = self.search_map[token]
        sample_dict = all_samples[idx]
        assert token == sample_dict['tokens']                                                  
        frame_list  = self.get_frames(item, sample_dict)
        action_dict = dict()
        action_dict["trajectory"] = torch.tensor(item['traj'].flatten())

        img_seq = list()
        for each_path in frame_list:
            img = load_img(each_path, 576, 1024)
            img_seq.append(img)
        images = torch.stack(img_seq)
        value_dict = init_embedder_options(self.unique_keys)
        cond_img = img_seq[0][None]
        value_dict["cond_frames_without_noise"] = cond_img
        value_dict["cond_aug"] = 0.0
        value_dict["cond_frames"] = cond_img + 0.0 * torch.randn_like(cond_img)
        if action_dict is not None:
            for key, value in action_dict.items():
                value_dict[key] = value

        sampler = init_sampling(guider='VanillaCFG', steps=50, cfg_scale=2.5, num_frames=25)

        uc_keys = ["cond_frames", "cond_frames_without_noise", "command", "trajectory", "speed", "angle", "goal"]
        out = self.do_sample_v1(
            images,
            self.model,
            sampler,
            value_dict,
            num_rounds=1,
            num_frames=self.config.num_frames,
            force_uc_zero_embeddings=uc_keys,
            initial_cond_indices=[index for index in range(1)]
        ) # samples, samples_z, inputs
        virtual_path = os.path.join("outputs", "virtual")
        real_path = os.path.join("outputs", "real")
        perform_save_locally(virtual_path, out[0], "videos", "NUSCENES", item['id'])
        perform_save_locally(real_path, images, "videos", "NUSCENES", item['id'])
        perform_save_locally(virtual_path, out[0][[6,12]], "images", "NUSCENES", item['id'])
        image_paths = list()
        image_paths.append(os.path.join(virtual_path, "images", f"NUSCENES_{item['id']:06}_{1:04}.png"))
        image_paths.append(os.path.join(virtual_path, "images", f"NUSCENES_{item['id']:06}_{2:04}.png"))
        print(item['id'])
        return out[0], image_paths
    
    
    @torch.no_grad()
    def do_sample(
            self,
            images,
            model,
            sampler,
            value_dict,
            num_rounds,
            num_frames,
            force_uc_zero_embeddings: Optional[List] = None,
            initial_cond_indices: Optional[List] = None,
            device="cuda"
    ):
        if initial_cond_indices is None:
            initial_cond_indices = [0]

        force_uc_zero_embeddings = default(force_uc_zero_embeddings, list())
        precision_scope = autocast

        with torch.no_grad(), precision_scope(device), model.ema_scope("Sampling"):
            c, uc = get_condition(model, value_dict, num_frames, force_uc_zero_embeddings, device)

            load_model(model.first_stage_model)
            z = model.encode_first_stage(images)
            unload_model(model.first_stage_model)

            samples_z = torch.zeros((num_rounds * (num_frames - 3) + 3, *z.shape[1:])).to(device)

            sampling_progress = tqdm(total=num_rounds, desc="Dreaming")

            def denoiser(x, sigma, cond, cond_mask):
                return model.denoiser(model.model, x, sigma, cond, cond_mask)

            load_model(model.denoiser)
            load_model(model.model)

            initial_cond_mask = torch.zeros(num_frames).to(device)
            initial_cond_mask[initial_cond_indices] = 1

            filled_latent = fill_latent(z, num_frames, initial_cond_indices, device)
            noise = torch.randn_like(filled_latent)
            sample = sampler(
                denoiser,
                noise,  #25, 4, 72, 128
                cond=c,
                uc=uc,
                cond_frame=filled_latent,  # cond_frame will be rescaled when calling the sampler
                cond_mask=initial_cond_mask
            )
            sampling_progress.update(1)
            samples_z = sample

            unload_model(model.model)
            unload_model(model.denoiser)

            load_model(model.first_stage_model)
            samples_x = model.decode_first_stage(samples_z)
            unload_model(model.first_stage_model)

            samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0)
           
            return samples, samples_z, images
    @torch.no_grad()
    def do_sample_v1(
            self,
            images,
            model,
            sampler,
            value_dict,
            num_rounds,
            num_frames,
            force_uc_zero_embeddings: Optional[List] = None,
            initial_cond_indices: Optional[List] = None,
            device="cuda"
    ):
        if initial_cond_indices is None:
            initial_cond_indices = [0]

        force_uc_zero_embeddings = default(force_uc_zero_embeddings, list())
        precision_scope = autocast

        with torch.no_grad(), precision_scope(device), model.ema_scope("Sampling"):
            c, uc = get_condition(model, value_dict, num_frames, force_uc_zero_embeddings, device)

            load_model(model.first_stage_model)
            z = model.encode_first_stage(images)
            unload_model(model.first_stage_model)

            samples_z = torch.zeros((num_rounds * (num_frames - 3) + 3, *z.shape[1:])).to(device)

            sampling_progress = tqdm(total=num_rounds, desc="Dreaming")

            def denoiser(x, sigma, cond, cond_mask):
                return model.denoiser(model.model, x, sigma, cond, cond_mask)

            load_model(model.denoiser)
            load_model(model.model)

            initial_cond_mask = torch.zeros(num_frames).to(device)
            initial_cond_mask[initial_cond_indices] = 1

            noise = torch.randn_like(z)
            sample = sampler(
                denoiser,
                noise,  #25, 4, 72, 128
                cond=c,
                uc=uc,
                cond_frame=z,  # cond_frame will be rescaled when calling the sampler
                cond_mask=initial_cond_mask
            )
            sampling_progress.update(1)
            samples_z = sample

            unload_model(model.model)
            unload_model(model.denoiser)

            load_model(model.first_stage_model)
            samples_x = model.decode_first_stage(samples_z)
            unload_model(model.first_stage_model)

            samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0)
           
            return samples, samples_z, images

    def construct_search_map(self, gen_data_path):
        with open(gen_data_path, "r") as f1:
            gen_data = json.load(f1) 
        search_map = {str(item["tokens"]): i for i, item in enumerate(gen_data)}
        return search_map

    
    def get_frames(self, item, sample_dict):
        path_list = list()
        for index in range(10):
            image_path = os.path.join(self.base_data_path, sample_dict["frames"][index])
            path_list.append(image_path)
        return path_list

    def get_frames(self, item, sample_dict):
        path_list = list()
        for index in range(len(sample_dict["frames"])):
            image_path = os.path.join(self.base_data_path, sample_dict["frames"][index])
            path_list.append(image_path)
        return path_list

    def get_action(self, item, base_traj):
        action_mode = item.get('action_mode', None)
        action_dict = dict()
        if action_mode == "traj" or action_mode == "trajectory":
            pred_traj = torch.tensor(item["traj"])
            pred_rel = pred_traj - base_traj[-1]
            base_rel = base_traj - base_traj[-1]
            action_dict["trajectory"] = np.concatenate([base_rel[:1], pred_rel], axis=0).flatten()
        elif action_mode == "cmd" or action_mode == "command":
            action_dict["command"] = torch.tensor(item["cmd"])
        elif action_mode == "steer":
            # scene might be empty
            if item["speed"]:
                action_dict["speed"] = torch.tensor(item["speed"][1:])
            # scene might be empty
            if item["angle"]:
                action_dict["angle"] = torch.tensor(item["angle"][1:]) / 780
        elif action_mode == "goal":
            # point might be invalid
            if item["z"] > 0 and 0 < item["goal"][0] < 1600 and 0 < item["goal"][1] < 900:
                action_dict["goal"] = torch.tensor([
                    item["goal"][0] / 1600,
                    item["goal"][1] / 900
                ])
        else:
            raise ValueError(f"Unsupported action mode {action_mode}")