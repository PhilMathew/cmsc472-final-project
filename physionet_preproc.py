from argparse import ArgumentParser

from pathlib import Path
import pandas as pd
import wfdb
from tqdm import tqdm


def extract_data(dir_path, dx_map):
    sig_paths, dx_labels = [], []
    for p in dir_path.iterdir():
        if p.suffix == '.hea':
            p_no_suffix = str(p)[:-(len('.hea'))]
            try:
                sig, ann = wfdb.rdsamp(p_no_suffix)
            except Exception as e:
                continue
            dx_list = ann['comments'][2][len('Dx: '):].split(',')
            dx_label_map = [1 if str(v) in dx_list else 0 for v in dx_map.values()]
            
            sig_paths.append(Path(f'{p_no_suffix}.mat'))
            dx_labels.append(dx_label_map)
            
    return sig_paths, dx_labels


def build_data_csv(root_dir, dx_map):
    dxs = list(dx_map.keys())
    df_dict = {k:[] for k in ('signal_file', *dxs)}
    for p in root_dir.iterdir():
        if p.is_dir():
            dir_list = [d for d in p.iterdir() if d.suffix != '.html']
            for q in tqdm(dir_list, desc=f'Extracting from {p.name}'):
                sig_paths, dx_labels = extract_data(q, dx_map)
                df_dict['signal_file'].extend(sig_paths)
                for label_list in dx_labels:
                    for dx, label in zip(dxs, label_list):
                        df_dict[dx].append(label)
                        
    df = pd.DataFrame(df_dict)
    
    return df


def main():
    parser = ArgumentParser(description='ECG Database Preprocessing Script')
    parser.add_argument('--data_root_dir', dest='data_root_dir', help='Path to directory containing WFDBRecords folder and ConditionNames_SNOMED-CT.csv')
    parser.add_argument('-o', '--output_dir', dest='output_dir', default='data_csvs', help='Path to directory to output CSV to')
    args = parser.parse_args()
    
    data_root_dir = Path(args.data_root_dir)
    
    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        output_dir.mkdir()
    
    wfdb_dir, dx_map_path = data_root_dir / 'WFDBRecords', data_root_dir / 'ConditionNames_SNOMED-CT.csv'
    dx_map_df = pd.read_csv(dx_map_path)
    dx_names, dx_ids = list(dx_map_df['Acronym Name']), list(dx_map_df['Snomed_CT'])
    dx_map = {k: v for k, v in zip(dx_names, dx_ids)}
    df = build_data_csv(wfdb_dir, dx_map)
    
    for dx in dx_names:
        print(f'Num. with {dx}: {len(df[df[dx] == 1])}')
    
    df.to_csv(str(output_dir / 'data.csv'), columns=df.columns, index=False)
    
    
    
if __name__ == '__main__':
    main()