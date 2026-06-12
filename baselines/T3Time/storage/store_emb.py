import sys
import os
import time
import h5py
import argparse
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader
from data_provider.data_loader_save import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom
from gen_prompt_emb import GenPromptEmb

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda", help="")
    parser.add_argument("--data_path", type=str, default="ETTh1")
    parser.add_argument("--num_nodes", type=int, default=7)
    parser.add_argument("--input_len", type=int, default=96)
    parser.add_argument("--output_len", type=int, default=96)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--l_layers", type=int, default=12)
    parser.add_argument("--model_name", type=str, default="gpt2")
    parser.add_argument("--divide", type=str, default="train")
    parser.add_argument("--num_workers", type=int, default=min(10, os.cpu_count()))
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument(
        "--prompt_batch_size",
        type=int,
        default=32,
        help="Number of GPT-2 prompts to embed per forward pass.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate requested embeddings even if matching cache files already exist.",
    )
    return parser.parse_args()

def get_dataset(data_path, flag, input_len, output_len):
    datasets = {
        'ETTh1': Dataset_ETT_hour,
        'ETTh2': Dataset_ETT_hour,
        'ETTm1': Dataset_ETT_minute,
        'ETTm2': Dataset_ETT_minute
    }
    dataset_class = datasets.get(data_path, Dataset_Custom)
    return dataset_class(flag=flag, size=[input_len, 0, output_len], data_path=data_path)


def count_cached_prefix(save_path, limit):
    count = 0
    for index in range(limit):
        if not os.path.exists(os.path.join(save_path, f"{index}.h5")):
            break
        count += 1
    return count


def write_meta(save_path, args, dataset_size, required_samples, generated_samples):
    cached_prefix_samples = count_cached_prefix(save_path, max(dataset_size, required_samples))
    meta = {
        "data_path": args.data_path,
        "divide": args.divide,
        "input_len": args.input_len,
        "output_len": args.output_len,
        "current_dataset_size": dataset_size,
        "required_samples": required_samples,
        "cached_prefix_samples": cached_prefix_samples,
        "written_samples": cached_prefix_samples,
        "generated_samples": generated_samples,
        "complete_for_current_output_len": cached_prefix_samples >= dataset_size,
        "max_samples": args.max_samples,
        "prompt_batch_size": args.prompt_batch_size,
        "embedding_batch_size": args.batch_size,
        "cache_layout": "shared_by_input_length",
    }
    with open(os.path.join(save_path, "_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def save_embeddings(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    train_set = get_dataset(args.data_path, 'train', args.input_len, args.output_len)
    test_set = get_dataset(args.data_path, 'test', args.input_len, args.output_len)
    val_set = get_dataset(args.data_path, 'val', args.input_len, args.output_len)

    data_loader = {
        'train': DataLoader(train_set, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers),
        'test': DataLoader(test_set, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers),
        'val': DataLoader(val_set, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers)
    }[args.divide]

    save_path = os.path.join(
        os.environ.get("T3TIME_EMBED_ROOT", "./Embeddings"),
        args.data_path,
        f"seq{args.input_len}",
        args.divide,
    )
    os.makedirs(save_path, exist_ok=True)

    emb_time_path = f"./Results/emb_logs/"
    os.makedirs(emb_time_path, exist_ok=True)

    dataset_size = len(data_loader.dataset)
    required_samples = min(args.max_samples if args.max_samples else dataset_size, dataset_size)
    if not args.force:
        cached_prefix_samples = count_cached_prefix(save_path, required_samples)
        if cached_prefix_samples >= required_samples:
            write_meta(save_path, args, dataset_size, required_samples, generated_samples=0)
            print(
                f"Embedding cache ready for {args.data_path}/{args.divide}: "
                f"{required_samples}/{required_samples} samples"
            )
            return

    gen_prompt_emb = GenPromptEmb(
        device=device, # type: ignore
        input_len=args.input_len,
        data_path=args.data_path,
        model_name=args.model_name,
        d_model=args.d_model,
        layer=args.l_layers,
        divide=args.divide,
        prompt_batch_size=args.prompt_batch_size,
    ).to(device)

    generated = 0
    written = 0
    for i, (x, y, x_mark, y_mark) in enumerate(data_loader):
        batch_start = i * args.batch_size
        if batch_start >= required_samples:
            break
        missing_offsets = []
        missing_indices = []
        for offset in range(len(x)):
            sample_index = batch_start + offset
            if sample_index >= required_samples:
                break
            file_path = os.path.join(save_path, f"{sample_index}.h5")
            if args.force or not os.path.exists(file_path):
                missing_offsets.append(offset)
                missing_indices.append(sample_index)
        if not missing_offsets:
            continue

        embeddings = gen_prompt_emb.generate_embeddings(
            x[missing_offsets].to(device),
            x_mark[missing_offsets].to(device),
        )

        if embeddings.dim() == 2:
            embeddings = embeddings.unsqueeze(0)
        embeddings_np = embeddings.detach().cpu().numpy()
        for offset, sample_index in enumerate(missing_indices):
            file_path = os.path.join(save_path, f"{sample_index}.h5")
            with h5py.File(file_path, 'w') as hf:
                hf.create_dataset('embeddings', data=embeddings_np[offset:offset + 1])
        generated += len(missing_indices)
        written = max(written, max(missing_indices) + 1)

        # # Save and visualize the first sample
        # if i >= 0:
        #     break
    write_meta(save_path, args, dataset_size, required_samples, generated_samples=generated)
    cached_prefix_samples = count_cached_prefix(save_path, required_samples)
    print(
        f"Generated {generated} embedding samples for {args.data_path}/{args.divide}; "
        f"cache ready {cached_prefix_samples}/{required_samples}"
    )
    
if __name__ == "__main__":
    args = parse_args()
    t1 = time.time()
    save_embeddings(args)
    t2 = time.time()
    print(f"Total time spent: {(t2 - t1)/60:.4f} minutes")
