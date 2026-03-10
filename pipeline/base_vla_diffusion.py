
import os
import torch
import torch.nn as nn
import numpy as np

from internvl_wrapper import InterVLPredictor
from vista_wrapper import VistaGenerator


class Vla_Diffusion_model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.mllm = InterVLPredictor(cfg)
        self.generator = VistaGenerator(cfg)
        self.l2_1_all = np.zeros((6,),dtype=np.float32)
        self.l2_0_all = np.zeros((6,),dtype=np.float32)
        self.count = 0

    
    def updata(self, l2_0, l2_1):
        self.l2_0_all += l2_0
        self.l2_1_all += l2_1
        self.count +=1
    def result(self):
        return self.l2_0_all/self.count, self.l2_1_all/self.count

    def compute_diff(self, pred_traj_0, pred_traj_1, gt):
        # print(f"pred_traj_0:{pred_traj_0}, pred_traj_1:{pred_traj_1},gt:{gt}")
        l2_0 = np.linalg.norm(pred_traj_0 - gt, axis=1)
        l2_1 = np.linalg.norm(pred_traj_1 - gt, axis=1)
        diff  = l2_1 - l2_0
        self.updata(l2_0, l2_1)

        return diff
    def forward(self, data):
        mllm_result_0 = self.mllm.test(data)
        # print(mllm_result)
        gt = np.array(data['gt_traj']).reshape(-1,2)
        his_traj = np.array(data['his_traj']).reshape(-1,2)
        his_traj = np.cumsum(his_traj, axis=-2)
        # print(his_traj)
        item = dict()
        item['id'] = data['id']
        item['token'] = mllm_result_0['token']
        item['action_mode'] = "traj"
        traj_pred = mllm_result_0['traj_pred']

        #过去6帧生成未来19帧，取0.5s和1s,取第12和18张
        item['traj'] = traj_pred[0:4] + his_traj[0,:] #0-2s
        _, image_paths = self.generator.test(item) #25
        # image_paths = ["samples/CAM_FRONT/n015-2018-07-24-10-42-41+0800__CAM_FRONT__1532400346162460.jpg", "samples/CAM_FRONT/n015-2018-07-24-10-42-41+0800__CAM_FRONT__1532400346612460.jpg", "samples/CAM_FRONT/n015-2018-07-24-10-42-41+0800__CAM_FRONT__1532400347112460.jpg"]
        mllm_result_1 = self.mllm.test_v1(data, image_paths)

        diff = self.compute_diff(mllm_result_0['traj_pred'], mllm_result_1['traj_pred'], gt)
        
        # print(diff)
        # print(diff.sum())

        #过去6帧0.5s+预测未来1.5s18帧
        # item['traj'] = traj_pred[:3] # 只需要前2s
        # print(item['traj'].shape)
        # gen_result, image_path = self.generator.test(item)
        # if gen_result == None:
        #     print("no use diffusion")
        # else:
        #     mllm_result = self.mllm.test_v1(data, image_path)
        #     print(mllm_result)
        # gt_traj = mllm_result['answer']
        # ori_l2 = np.linalg.norm(traj_pred - gt_traj, axis=1)  # shape: (6,)
        # new_traj = mllm_result['traj_pred'][:4] + traj_pred[1]
        # new_traj = np.concatenate([traj_pred[:2], new_traj], axis=0)
        # new_l2 = np.linalg.norm(new_traj - gt_traj, axis=1) 
        # print(f"ori_traj:{traj_pred}")
        # print(f"new_traj:{new_traj}")
        # print(ori_l2)
        # print(new_l2)
        # print(new_l2-ori_l2)
        
        #直接未来帧做参考帧生成
        # item['traj'] = traj_pred[:4] # 只需要前2s
        # gen_result, image_path = self.generator.test_v1(item)
        # mllm_result_1 = self.mllm.test_v1(data, image_path)
        # diff = self.compute_diff(mllm_result_0['traj_pred'], mllm_result_1['traj_pred'], gt)
        
        # print(diff)
        # print(diff.sum())
        


        return diff, mllm_result_0['traj_pred'], mllm_result_1['traj_pred'], gt
    
    def forward_test(self, data):
        mllm_result_0 = self.mllm.test(data)
        # print(mllm_result)
        gt = np.array(data['gt_traj']).reshape(-1,2)
        his_traj = np.array(data['his_traj']).reshape(-1,2)
        his_traj = np.cumsum(his_traj, axis=-2)
        # print(his_traj)
        item = dict()
        item['id'] = data['id']
        item['token'] = mllm_result_0['token']
        item['action_mode'] = "traj"
        traj_pred = mllm_result_0['traj_pred']
        result = dict()
        result['gt'] = gt.flatten().tolist()
        result['base_traj'] = traj_pred.flatten().tolist()
        # traj_score = mllm_result_0['traj_score']
        # result['base_score'] = traj_score.flatten().tolist()
        # for idx in range(5):
        import time
        start = time.perf_counter()
        for idx in range(5):
            #过去6帧生成未来19帧，取0.5s和1s,取第12和18张
            item['traj'] = traj_pred[0:4] + his_traj[0,:] #0-2s
            _, image_paths = self.generator.test(item) #25
            # image_paths = ["samples/CAM_FRONT/n015-2018-07-24-10-42-41+0800__CAM_FRONT__1532400346162460.jpg", "samples/CAM_FRONT/n015-2018-07-24-10-42-41+0800__CAM_FRONT__1532400346612460.jpg", "samples/CAM_FRONT/n015-2018-07-24-10-42-41+0800__CAM_FRONT__1532400347112460.jpg"]
            mllm_result_1 = self.mllm.test_v1(data, image_paths)
            traj_pred = mllm_result_1['traj_pred']
            diff = self.compute_diff(mllm_result_0['traj_pred'], mllm_result_1['traj_pred'], gt)
            result[f'traj_pred_{idx}'] = traj_pred.flatten().tolist()
            # traj_score = mllm_result_1['traj_score']
            # result[f'traj_score_{idx}'] = traj_score.flatten().tolist()
            result[f'diff_{idx}'] = diff.tolist()
            result[f'diff_sum_{idx}'] = diff.sum()
        

        return result
    
    #start from current frame
    def forward_test_v2(self, data):
        mllm_result_0 = self.mllm.test(data)
        # print(mllm_result)
        gt = np.array(data['gt_traj']).reshape(-1,2)
        # his_traj = np.array(data['his_traj']).reshape(-1,2)
        # his_traj = np.cumsum(his_traj, axis=-2)
        # print(his_traj)
        item = dict()
        item['id'] = data['id']
        item['token'] = mllm_result_0['token']
        item['action_mode'] = "traj"
        traj_pred = mllm_result_0['traj_pred']
        result = dict()
        result['gt'] = gt.flatten().tolist()
        result['base_traj'] = traj_pred.flatten().tolist()
        # traj_score = mllm_result_0['traj_score']
        # result['base_score'] = traj_score.flatten().tolist()
        # for idx in range(5):
        import time
        start = time.perf_counter()
        for idx in range(5):
            #过去3帧生成未来22帧，取0.5s和1s,取第8和14张
            # item['traj'] = traj_pred[0:4] + his_traj[0,:] #0-2s
            item['traj'] = traj_pred[0:4]
            _, image_paths = self.generator.test_v2(item) #25
            # image_paths = ["samples/CAM_FRONT/n015-2018-07-24-10-42-41+0800__CAM_FRONT__1532400346162460.jpg", "samples/CAM_FRONT/n015-2018-07-24-10-42-41+0800__CAM_FRONT__1532400346612460.jpg", "samples/CAM_FRONT/n015-2018-07-24-10-42-41+0800__CAM_FRONT__1532400347112460.jpg"]
            mllm_result_1 = self.mllm.test_v1(data, image_paths)
            traj_pred = mllm_result_1['traj_pred']
            diff = self.compute_diff(mllm_result_0['traj_pred'], mllm_result_1['traj_pred'], gt)
            result[f'traj_pred_{idx}'] = traj_pred.flatten().tolist()
            # traj_score = mllm_result_1['traj_score']
            # result[f'traj_score_{idx}'] = traj_score.flatten().tolist()
            result[f'diff_{idx}'] = diff.tolist()
            result[f'diff_sum_{idx}'] = diff.sum()
        end = time.perf_counter()
        print(f"Elapsed time: {end - start:.6f} seconds")

        return result
