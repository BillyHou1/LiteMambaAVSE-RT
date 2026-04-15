import os
import json
import argparse

def collect_pairs(grid_root):
    grid_root = os.path.abspath(grid_root)
    if not os.path.isdir(grid_root):
        return {}
    audio_paths = {}
    video_paths = {}
    for dirpath, _, filenames in os.walk(grid_root):
        rel = os.path.relpath(dirpath, grid_root)
        parts = rel.split(os.sep)
        if len(parts) < 2 or parts[0] not in ('audio', 'video'):
            continue
        spk = parts[1]
        if spk in ('audio', 'video'):
            continue
        for f in filenames:
            stem, ext = os.path.splitext(f)
            ext = ext.lower()
            full = os.path.abspath(os.path.join(dirpath, f))
            key = (spk, stem)
            if ext == '.wav':
                audio_paths[key] = full
            elif ext == '.mpg':
                video_paths[key] = full
    common = set(audio_paths) & set(video_paths)
    by_speaker = {}
    for key in common:
        spk = key[0]
        if spk not in by_speaker:
            by_speaker[spk] = []
        by_speaker[spk].append({'audio': audio_paths[key], 'video': video_paths[key]})
    for spk in by_speaker:
        by_speaker[spk].sort(key=lambda x: x['audio'])
    return by_speaker

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--grid_root', required=True)
    parser.add_argument('--output_dir', default='data')
    args = parser.parse_args()

    train_spk = [f's{i}' for i in range(1, 29)]
    valid_spk = [f's{i}' for i in range(29, 32)]
    test_spk = [f's{i}' for i in range(32, 35)]

    by_speaker = collect_pairs(args.grid_root)
    if not by_speaker:
        return
    train, valid, test = [], [], []
    for spk, pairs in by_speaker.items():
        if spk in train_spk:
            train.extend(pairs)
        elif spk in valid_spk:
            valid.extend(pairs)
        elif spk in test_spk:
            test.extend(pairs)
    out_dir = os.path.abspath(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)
    for name, data in [('grid_train', train), ('grid_valid', valid), ('grid_test', test)]:
        with open(os.path.join(out_dir, f'{name}.json'), 'w') as f:
            json.dump(data, f, indent=2)

if __name__ == '__main__':
    main()
