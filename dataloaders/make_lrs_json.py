#read LRS2 split txt files and dump json lists
import os
import json
import argparse

def read_split(txt_path, root):
    root = os.path.abspath(root)
    if not os.path.isfile(txt_path):
        return []
    out = []
    with open(txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            full = os.path.normpath(os.path.join(root, line))
            out.append({'video': os.path.abspath(full)})
    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lrs2_root', required=True)
    parser.add_argument('--output_dir', default='data')
    args = parser.parse_args()

    root = os.path.abspath(args.lrs2_root)
    if not os.path.isdir(root):
        return
    os.makedirs(args.output_dir, exist_ok=True)
    out_dir = os.path.abspath(args.output_dir)
    for txt_name, json_name in [('train.txt', 'lrs2_train.json'),
                                 ('val.txt', 'lrs2_valid.json'),
                                 ('test.txt', 'lrs2_test.json')]:
        txt_path = os.path.join(root, txt_name)
        lst = read_split(txt_path, root)
        with open(os.path.join(out_dir, json_name), 'w') as f:
            json.dump(lst, f, indent=2)

if __name__ == '__main__':
    main()
