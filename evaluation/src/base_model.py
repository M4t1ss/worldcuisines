from transformers import set_seed
from PIL import Image
from datasets import load_dataset
import torch
import numpy as np
import json
import math
from io import BytesIO
from tqdm import tqdm
import os

MODEL_HANDLE = {}


def load_model_processor(model_path, fp32=False, multi_gpu=False):
    raise NotImplementedError


def eval_instance(model, processor, image_file, query):
    raise NotImplementedError


def set_all_seed(seed=42):
    set_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_kb_from_hf():
    ds = load_dataset("world-cuisines/food-kb")
    data = ds["main"].to_pandas()
    return data


def get_url_jpg_map(data):
    url_jpg_map = {}
    for _, row in data.iterrows():
        for i in range(1, 9):
            if row[f"image{i}_url"] is not None:
                url_jpg_map[row[f"image{i}_url"]] = Image.open(
                    BytesIO(row[f"image{i}"]["bytes"])
                )
    return url_jpg_map


def get_vqa_from_hf(task, subset="large"):
    if task == 1:
        vqa = load_dataset("world-cuisines/vqa", name="task1")
    elif task == 2:
        vqa = load_dataset("world-cuisines/vqa", name="task2")
    else:
        raise ValueError(
            "Invalid task number. Please choose from 1 (dish name) or 2 (region)"
        )

    # Load test_small or test_large based on the subset
    split = f"test_{subset}"
    vqa_data = vqa[split].to_pandas()
    return vqa_data


def get_unique_filename(path: str):
    """Generate a unique file name by adding (1), (2), etc. if the file already exists."""
    if not os.path.exists(path):
        return path

    base, ext = os.path.splitext(path)
    counter = 1
    new_path = f"{base} ({counter}){ext}"

    while os.path.exists(new_path):
        counter += 1
        new_path = f"{base} ({counter}){ext}"

    return new_path


def export_result(result: list, path: str, replace=False):
    """Export results to a file, ensuring unique file names."""
    path = path if replace else get_unique_filename(path)
    with open(path, "w") as outfile:
        for entry in result:
            json.dump(entry, outfile)
            outfile.write("\n")


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def log_error(error_message, log_file="error.txt"):
    """Log error message to a text file."""
    with open(log_file, "a") as f:
        f.write(error_message + "\n")


def main(task, qa_type, model_path, fp32, multi_gpu, limit=np.inf,
         st_idx=None, ed_idx=None, chunk_num=1, chunk_id=0, subset="large"):
    set_all_seed()

    kb_data = get_kb_from_hf()
    url_jpg_map = get_url_jpg_map(kb_data)
    vqa_data = get_vqa_from_hf(task, subset=subset)

    suffix_slice = ""
    if st_idx is not None or ed_idx is not None:
        _ = len(vqa_data)
        vqa_data = vqa_data.iloc[st_idx:ed_idx]
        print(f"  > Total Data to Process: {len(vqa_data):8,}.  (of {_:,})")
        print(f"          Start-End Index:  {st_idx}  to  {ed_idx}")
        suffix_slice += ("" if st_idx is None else f"_s{st_idx}") + ("" if ed_idx is None else f"_e{ed_idx}")

    if not ((chunk_num == 1) and (chunk_id == 0)):
        chunk_index = split_list(vqa_data.index, chunk_num)[chunk_id]
        print(f"  > Total Data to Process: {len(chunk_index):8,}.  (of {len(vqa_data):,})")
        print(f"    {chunk_num:>5} chunks with ID#:  {chunk_id}  (start idx: {chunk_index[0]})")
        vqa_data = vqa_data.loc[chunk_index]
        suffix_slice += f".chunk{chunk_id}_of_{chunk_num}"

    model, processor, tokenizer = load_model_processor(model_path, fp32, multi_gpu)

    list_res = []
    count = 0
    error_counter = 0

    def _log_error(msg, suf=""):
        pref = f"./result/error/{subset}_task{task}_{qa_type}_{MODEL_HANDLE[model_path]}"
        cur_suf = suffix_slice + (f".{os.getenv('SLURM_JOB_ID')}" if 'SLURM_JOB_ID' in os.environ else "")
        log_error(msg, f"{pref}_error{cur_suf}.txt")
        export_result(list_res, f"{pref}_pred_{suf}{cur_suf}.jsonl", replace=True)

    try:
        for _, row in tqdm(vqa_data.iterrows(), total=len(vqa_data)):
            res = {}
            try:
                image_file = url_jpg_map[row["image_url"]]
                query = row["multi_choice_prompt"] if qa_type == "mc" else row["open_ended_prompt"]
                response = eval_instance(model, processor, image_file, query, tokenizer=tokenizer)
    
                res["qa_id"] = row["qa_id"]
                res["prediction"] = response
                res["lang"] = row["lang"]
                qa_type_txt = "answer" if qa_type == "oe" else "multi_choice_answer"
                res[qa_type_txt] = row[qa_type_txt]
                list_res.append(res)

            except Exception as e:
                _log_error(f"Error at row {row['qa_id']} ({row['lang']}): {str(e)}", f"latest")
                error_counter += 1

            count += 1
            if count == limit:
                break

    except KeyboardInterrupt:
        print("KeyboardInterrupt. Exporting latest results...")
        _log_error(f"KeyboardInterrupt at row {row['qa_id']} ({row['lang']})", "interrupt")

    return list_res
