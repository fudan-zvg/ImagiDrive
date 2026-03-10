import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
from multiprocessing import Process
import multiprocessing as mp
import torch
import torch.nn as nn
import random
from tqdm import tqdm
from dataclasses import dataclass
from base_vla_diffusion import Vla_Diffusion_model


@dataclass
class BaseConfig:
    model_path: str = ""
    is_train: bool = False
    rank: int = 0
    base_data_path: str = "" # nuscenes data path 
    data_path: str = "" # prompt data path
    gen_data_path: str = "" # generated data path
    height: int = 576
    width: int = 1024
    n_conds: int = 6
    num_frames: int = 25
    low_vram: bool = True

def main(rank, config, data, common_tokens_splited, search_map_0):
    config.rank = rank
    model = Vla_Diffusion_model(cfg = config)
    results = dict()
    temp_save_path = f"./gpu_1.json"
    with open(temp_save_path, "w") as f:
        json.dump([], f) 

    for idx, token in enumerate(tqdm(common_tokens_splited)):
        diff = model(data[search_map_0[token]])
        with open(temp_save_path, "r+") as f:
            current_results = json.load(f)
            diff_sum = diff.sum()
            # convert tensor to list so it can be serialized as JSON
            results = {'idx': idx, 'diff': diff.tolist(), 'diff_sum':diff_sum}
            current_results.append(results)
            f.seek(0)
            json.dump(current_results, f, indent=4)
        print(f"Processed token {token}, idx {idx}")
    
    print("Finished processing all tokens.")
    print(model.result())
    


    

if __name__== "__main__":
    mp.set_start_method('spawn', force=True)
    config = BaseConfig()
    save_path = "./base_results.json"
    data = []
    with open(config.data_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    search_map_0 = {str(item["token"]): i for i, item in enumerate(data)}
    with open(config.gen_data_path, "r") as f1:
        gen_data = json.load(f1) 
    search_map_1 = {str(item["tokens"]): i for i, item in enumerate(gen_data)}
    common_tokens = list(set(search_map_0.keys()) & set(search_map_1.keys()))
    # random.shuffle(common_tokens)
    num_gpus = torch.cuda.device_count()
    tokens_per_gpu = len(common_tokens) // num_gpus
    # split tokens across GPUs; the last chunk may have a few extra tokens
    sliced_tokens = [common_tokens[i * tokens_per_gpu:(i + 1) * tokens_per_gpu] for i in range(num_gpus)]

    if len(common_tokens) % num_gpus != 0:
        sliced_tokens[-1].extend(common_tokens[num_gpus * tokens_per_gpu:])
     # launch multi-process inference
    # processes = []
    # for rank in range(num_gpus):
    #     p = Process(target=main, args=(rank, config, data, sliced_tokens[rank], search_map_0))
    #     p.start()
    #     processes.append(p)

    # # wait for all processes to finish
    # for p in processes:
    #     p.join()
    main(0, config, data, sliced_tokens[0], search_map_0)
    
    # # merge results from all GPUs
    # data_dict = []
    # for rank in range(num_gpus):
    #     temp_save_path = f"./gpu_{rank}.json"
    #     with open(temp_save_path, 'r') as f:
    #         data_dict.extend(json.load(f))
    #     os.remove(temp_save_path)
    
    # # write merged results
    # with open(save_path, "w") as f:
    #     json.dump(data_dict, f, indent=4)

    # print(f"Inference finished. Results saved to {save_path}")

    # main()


