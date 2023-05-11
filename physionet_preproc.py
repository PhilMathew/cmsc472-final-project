from pathlib import Path
import pandas as pd
import wfdb
from tqdm import tqdm


def extract_data(dir_path, dx_of_interest):
    sig_paths, dx_lists, dx_of_interest_labels = [], [], []
    for p in dir_path.iterdir():
        if p.suffix == '.hea':
            p_no_suffix = str(p)[:-(len('.hea'))]
            try:
                sig, ann = wfdb.rdsamp(p_no_suffix)
            except Exception as e:
                continue
            dx_list = ann['comments'][2][len('Dx: '):].split(',')
            has_dx_of_interest = str(dx_of_interest) in dx_list
            
            sig_paths.append(Path(f'{p_no_suffix}.mat'))
            dx_lists.append(dx_list)
            dx_of_interest_labels.append(1 if has_dx_of_interest else 0)
            
    return sig_paths, dx_lists, dx_of_interest_labels


def build_data_csv(root_dir, dx_of_interest, dx_of_interest_name):
    df_dict = {'signal_file': [], 'dx': [], dx_of_interest_name: []}
    for p in root_dir.iterdir():
        if p.is_dir():
            dir_list = [d for d in p.iterdir() if d.suffix != '.html']
            for q in tqdm(dir_list, desc=f'Extracting from {p.name}'):
                sig_paths, dx_lists, dx_of_interest_labels = extract_data(q, dx_of_interest)
                df_dict['signal_file'].extend(sig_paths)
                df_dict['dx'].extend(dx_lists)
                df_dict[dx_of_interest_name].extend(dx_of_interest_labels)

    df = pd.DataFrame(df_dict)
    
    return df


def main():
    data_root_dir = Path('/home/phil/Documents/vscode-projects/UMD/cmsc472-final-project/physionet/files/ecg-arrhythmia/1.0.0/')
    wfdb_dir, dx_map_path = data_root_dir / 'WFDBRecords', data_root_dir / 'ConditionNames_SNOMED-CT.csv'
    dx_map = pd.read_csv(dx_map_path)
    dx_id = list(dx_map[dx_map['Acronym Name'] == 'AFIB']['Snomed_CT'])[0]
    df = build_data_csv(wfdb_dir, dx_id, 'AFIB')
    print(f"Num. with Dx: {len(df[df['AFIB'] == 1])}")
    df.to_csv('data_csvs/data.csv', columns=df.columns, index=False)
    
    
if __name__ == '__main__':
    main()