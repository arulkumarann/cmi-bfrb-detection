import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class CMISequenceDataset(Dataset):

    def __init__(self, df, is_train=True):

        self.is_train = is_train

        self.feature_cols = [c for c in df.columns if c not in ['sequence_id', 'gesture']]
        
        label_col = 'gesture'
        
        self.seqs = []
        self.labels = []
        for seq_id, group in df.groupby('sequence_id'):            
            seq_features = group[self.feature_cols].values.astype(np.float32)
            self.seqs.append(seq_features)
            
            if self.is_train:
                label = group[label_col].iloc[0]
                self.labels.append(label)
        
    def __len__(self):
        return len(self.seqs)
    
    def __getitem__(self, idx):
        seq = self.seqs[idx]  
        
        if np.isnan(seq).any():
            print(f"NaN found in sequence index {idx}")

        if self.is_train:
            label = self.labels[idx]
            return torch.from_numpy(seq), torch.tensor(label, dtype=torch.long)
        else:
            return torch.from_numpy(seq), None


def collate_fn(batch):

    sequences, labels = zip(*batch)
    lengths = [seq.shape[0] for seq in sequences]
    max_len = max(lengths)
    num_features = sequences[0].shape[1]
    batch_size = len(sequences)
    padded = torch.zeros(batch_size, max_len, num_features, dtype=torch.float32)
    mask = torch.ones(batch_size, max_len, dtype=torch.bool)  
    for i, seq in enumerate(sequences):
        length = seq.shape[0]
        padded[i, :length, :] = seq
        mask[i, :length] = False  
    if labels[0] is not None:
        labels = torch.tensor(labels, dtype=torch.long)
    else:
        labels = None
    return padded, labels, mask


def pre_processing(df, is_train=True, encoders=None, scaler=None):

    print(f"Starting preprocessing. Shape: {df.shape}")
    
    cols_to_drop = ['row_id', 'sequence_type', 'behavior', 'subject', 'orientation']
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

    print("Handling missing values")
    
    tof_cols = [c for c in df.columns if c.startswith("tof_")]
    if tof_cols:
        df[tof_cols] = df[tof_cols].fillna(-1)
        print(f"Filled {len(tof_cols)} ToF columns with -1")
    
    thm_cols = [c for c in df.columns if c.startswith("thm_")]
    if thm_cols:
        df[thm_cols] = df[thm_cols].fillna(df[thm_cols].mean())
        print(f"Filled {len(thm_cols)} thermopile columns with mean")
    
    rot_cols = [c for c in df.columns if c.startswith("rot_")]
    if rot_cols:
        rot_fill = {'rot_w': 1.0, 'rot_x': 0.0, 'rot_y': 0.0, 'rot_z': 0.0}
        df[rot_cols] = df[rot_cols].fillna(rot_fill)
        print(f"Filled {len(rot_cols)} rotation columns with null quaternion")
    
    
    if thm_cols:
        threshold = 10
        mask = (df.filter(like='thm_') < threshold).any(axis=1)
        filtered_df = df[mask]
        seqs_with_low_values = list(filtered_df['sequence_id'].unique())
        
        print(f"{len(seqs_with_low_values)} sequences with low thermopile values")

        for seq_id in seqs_with_low_values:
            seq_mask = df['sequence_id'] == seq_id
            if (df.loc[seq_mask, 'thm_3'] < threshold).all():
                df.loc[seq_mask, 'thm_3'] = df.loc[seq_mask, 'thm_2']
            if (df.loc[seq_mask, 'thm_1'] < threshold).all():
                df.loc[seq_mask, 'thm_1'] = df.loc[seq_mask, 'thm_2']
        
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    
    if is_train:
        encoders = {}
        for col in categorical_cols:
            if col == 'sequence_id':
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
            else:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])        
                encoders[col] = le
        print(f"Fitted encoders for: {list(encoders.keys())}")
    elif encoders is not None and not is_train:
        for col in categorical_cols:
            if col == 'sequence_id':
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
            else:
                if col in encoders:
                    le = encoders[col]
                    df[col] = le.transform(df[col])
        print("Applied existing encoders")
    
    print("Engineering features...")
    
    sensor_prefixes = ['acc_', 'thm_', 'tof_', 'rot_']
    cols_to_scale = [c for c in df.columns if any(c.startswith(p) for p in sensor_prefixes)]
    
    acc_cols = ['acc_x', 'acc_y', 'acc_z']
    if all(col in df.columns for col in acc_cols):
        acc_mag = np.sqrt(df['acc_x']**2 + df['acc_y']**2 + df['acc_z']**2)
        df['acc_mag'] = acc_mag
        cols_to_scale.append('acc_mag')
        
    rot_xyz_cols = ['rot_x', 'rot_y', 'rot_z']
    if all(col in df.columns for col in rot_xyz_cols):
        rot_mag = np.sqrt(df['rot_x']**2 + df['rot_y']**2 + df['rot_z']**2)
        df['rot_mag'] = rot_mag
        cols_to_scale.append('rot_mag')
    
    quat_cols = ['rot_w', 'rot_x', 'rot_y', 'rot_z']
    if all(col in df.columns for col in quat_cols):
        w, x, y, z = df['rot_w'], df['rot_x'], df['rot_y'], df['rot_z']
        euler_roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))
        euler_pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
        euler_yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
        
        df['euler_roll'] = euler_roll
        df['euler_pitch'] = euler_pitch
        df['euler_yaw'] = euler_yaw
        cols_to_scale.extend(['euler_roll', 'euler_pitch', 'euler_yaw'])
    
    if all(col in df.columns for col in rot_xyz_cols):
        angular_velocity_x = df.groupby('sequence_id')['rot_x'].diff().fillna(0)
        angular_velocity_y = df.groupby('sequence_id')['rot_y'].diff().fillna(0)
        angular_velocity_z = df.groupby('sequence_id')['rot_z'].diff().fillna(0)
        
        df['angular_velocity_x'] = angular_velocity_x
        df['angular_velocity_y'] = angular_velocity_y
        df['angular_velocity_z'] = angular_velocity_z
        cols_to_scale.extend(['angular_velocity_x', 'angular_velocity_y', 'angular_velocity_z'])
    
    print(f"Created {len([c for c in df.columns if c.endswith(('_mag', 'euler_', 'angular_velocity_'))])} new features")
    
    print("Scaling features...")
    
    if is_train:
        scaler = StandardScaler()
        scaler.fit(df[cols_to_scale])
        print(f"Fitted scaler on {len(cols_to_scale)} features")
    
    if scaler is not None:
        df[cols_to_scale] = scaler.transform(df[cols_to_scale])
        print("Applied scaling")
    
    print(f"Final preprocessing shape: {df.shape}")
    print(f"Remaining missing values: {df.isnull().sum().sum()}")
            
    return df, encoders, scaler


def load_and_prepare_data(data_dir='dataset'):

    print("LOADING AND PREPARING BFRB DATASET")
    
    train_df = pd.read_csv(f'{data_dir}/train.csv')
    test_df = pd.read_csv(f'{data_dir}/test.csv')
    
    print(f"Raw train shape: {train_df.shape}")
    print(f"Raw test shape: {test_df.shape}")
    print(f"Unique train gestures: {train_df['gesture'].nunique()}")
    print(f"Unique train sequences: {train_df['sequence_id'].nunique()}")
    
    print("\nProcessing training data...")
    train_df, encoders, scaler = pre_processing(train_df, is_train=True)
    
    print("\nProcessing test data...")
    test_df, _, _ = pre_processing(test_df, is_train=False, encoders=encoders, scaler=scaler)
    
    print("\nSplitting data...")
    train_sequences = train_df['sequence_id'].unique()
    train_seqs, val_seqs = train_test_split(train_sequences, test_size=0.2, random_state=42, 
                                           stratify=train_df.groupby('sequence_id')['gesture'].first())
    
    train_df_part = train_df[train_df['sequence_id'].isin(train_seqs)].copy()
    val_df = train_df[train_df['sequence_id'].isin(val_seqs)].copy()
    
    print(f"Train sequences: {len(train_seqs)}")
    print(f"Val sequences: {len(val_seqs)}")
    print(f"Test sequences: {test_df['sequence_id'].nunique()}")
    
    print("\nCreating PyTorch datasets...")
    train_dataset = CMISequenceDataset(train_df_part, is_train=True)
    val_dataset = CMISequenceDataset(val_df, is_train=True)
    test_dataset = CMISequenceDataset(test_df, is_train=False)
    
    print(f"Train dataset: {len(train_dataset)} sequences")
    print(f"Val dataset: {len(val_dataset)} sequences")
    print(f"Test dataset: {len(test_dataset)} sequences")
    print(f"Feature dimensions: {len(train_dataset.feature_cols)}")
    print(f"Number of classes: {len(set(train_dataset.labels))}")
    
    return {
        'train_dataset': train_dataset,
        'val_dataset': val_dataset,
        'test_dataset': test_dataset,
        'encoders': encoders,
        'scaler': scaler,
        'test_df': test_df,
        'num_features': len(train_dataset.feature_cols),
        'num_classes': len(set(train_dataset.labels))
    }


if __name__ == "__main__":
    data = load_and_prepare_data()
    
    train_loader = DataLoader(data['train_dataset'], batch_size=8, shuffle=True, collate_fn=collate_fn)
    
    print("\nTesting dataloader...")
    for batch in train_loader:
        seqs, labels, mask = batch
        print(f"Batch shape: {seqs.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Mask shape: {mask.shape}")
        print(f"Sample sequence length: {(~mask[0]).sum().item()}")
        break
    
    print("Preprocessing test completedd")