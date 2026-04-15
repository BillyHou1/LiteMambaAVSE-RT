import os
import json
import random
import argparse

def scan_dirs(noise_dirs):
    out =[]
    for d in noise_dirs:
        d = os.path.abspath(d)
        if not os.path.isdir(d):
            continue
        for root, _, files in os.walk(d):
            for f in files:
                if f.lower().endswith(('.wav', '.flac')):
                    out.append(os.path.abspath(os.path.join(root,f)))
    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--noise_dirs', nargs='+', required=True)
    parser.add_argument('--output_dir', default='data')
    parser.add_argument('--val_ratio', type=float, default=0.05)
    args = parser.parse_args()

    files = scan_dirs(args.noise_dirs)
    random.seed(1234)
    random.shuffle(files)
    n_val = max(1, int(len(files) * args.val_ratio))
    valid = files[:n_val]
    train = files[n_val:]
    out_dir = os.path.abspath(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)
    for name, data in [('noise_train', train), ('noise_valid', valid)]:
        with open(os.path.join(out_dir, f'{name}.json'), 'w') as f:
            json.dump(data, f, indent=2)

if __name__ == '__main__':
    main()
