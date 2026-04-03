#VoxCeleb2 json lists, split dev into train+valid (Not used in the paper, for future work)
import os
import json
import random
import argparse

def collect_clips(vox2_root, subset):
    sub_dir = os.path.join(os.path.abspath(vox2_root), subset)
    if not os.path.isdir(sub_dir):
        return []
    out = []
    for dirpath, _, filenames in os.walk(sub_dir):
        for f in filenames:
            if f.lower().endswith('.mp4'):
                out.append({'video': os.path.abspath(os.path.join(dirpath, f))})
    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vox2_root', required=True)
    parser.add_argument('--output_dir', default='data')
    parser.add_argument('--val_ratio', type=float, default=0.03)
    args = parser.parse_args()

    root = os.path.abspath(args.vox2_root)
    if not os.path.isdir(root):
        return
    dev_list = collect_clips(root, 'dev')
    test_list = collect_clips(root, 'test')
    random.seed(1234)
    random.shuffle(dev_list)
    n_val = max(1, int(len(dev_list) * args.val_ratio))
    valid = dev_list[:n_val]
    train = dev_list[n_val:]
    out_dir = os.path.abspath(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)
    for name, data in [('vox_train', train), ('vox_valid', valid), ('vox_test', test_list)]:
        with open(os.path.join(out_dir, f'{name}.json'), 'w') as f:
            json.dump(data, f, indent=2)

if __name__ == '__main__':
    main()
