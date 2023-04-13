# -*- coding: utf-8 -*-

import Encodings
import numpy as np
import math

def apply_corrections(data, update_df):
    # Update Valid entries
    valid_entries = update_df[~update_df['protein_sequence'].isnull()]
    data = data.set_index('seq_id')
    valid_entries = valid_entries.set_index('seq_id')
    data.update(valid_entries)
    data.reset_index()
    print(f"Replaced {len(valid_entries)} entries with interchanged pH and tm" )

    # Remove invalid entries
    invalid_entries = update_df[update_df['protein_sequence'].isnull()]
    data.loc[invalid_entries['seq_id'],'protein_sequence'] = None
    data = data[~data['protein_sequence'].isnull()]

    print(f"Removed {len(invalid_entries)} entries with invalid data" )

    print(f"Training data has {len(data)} entries" )
    
    return data

def split_test_dev(data, test_set_fraction):
    # Shuffle data
    data = data.sample(frac=1)

    # Extract test set
    selected_rows = math.floor((data.shape[0] * test_set_fraction))

    train_data = data[0:(data.shape[0] - selected_rows)]
    test_data = data[-selected_rows:]
    
    return train_data , test_data


def pad_list(l, content, width):
    l.extend([content] * (width - len(l)))
    return l

def normalize(x, xmin, xmax):
    return (x - xmin)/(xmax - xmin)

def apply_encoding(data, pad_x_data=True, one_hot_features=True, **kwargs):
    
    x = data['protein_sequence'].apply(list).to_numpy()
    
    for itr in range(0,len(x)):
        try: 
            x[itr] = [Encodings.letter2num[out] for out in x[itr]]
            if pad_x_data is True : 
                x[itr] = pad_list(x[itr], 0, 9000)
            
        except Exception:
            print(f"Iterator was {itr}")
            break
    
    # On Normalize data if not using one hot features.
    if one_hot_features is False:
        x_out = normalize(np.stack(x,axis=0), 0.0, 26.0)
    else:
        x_out = np.stack(x,axis=0)
    
    # Always normalize tm and ph features
    # y_out = normalize(data['tm'].to_numpy(), 25.1, 130.0)
    y_out = data['tm'].to_numpy()
    z_out = normalize(data['pH'].to_numpy(), 1.99, 11.0)
    
    
    
    return x_out, y_out, z_out