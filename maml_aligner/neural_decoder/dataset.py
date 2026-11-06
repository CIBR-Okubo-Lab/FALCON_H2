# dataset and dataloader for MAML implementation
# 1. Torch dataset
# 2. Lightning datasetmodule

import copy
import pytorch_lightning as pl
from torch.utils.data import IterableDataset, DataLoader
import random
import torch
from torch.nn.utils.rnn import pad_sequence
from typing import List, Dict

# Here we use a Iterable dataset, because we can generate infinite episodes
class MAMLTrainDataset(IterableDataset):
    def __init__(self, data_list: List[Dict], n_task: int = 3, k_tr: int = 5, k_ts: int = 5):
        """
        Infinite stream of MAML episodes.
        
        Args:
            data_list: List of days, each containing:
                - 'sentenceDat': List  # Variable-length sequences
                - 'chars': List        # Character sequences
                - 'charLens': List     # Original sequence lengths
            n_task: Number of tasks per episode
            k_tr: Support samples per task
            k_ts: Query samples per task
        """
        self.data = copy.deepcopy(data_list)
        self.n_task = n_task
        self.k_tr = k_tr
        self.k_ts = k_ts
        
        # Add day_id if not present
        for day_idx, day_data in enumerate(self.data):
            day_data['day_id'] = day_data.get('day_id', day_idx)

    def __iter__(self):
        """
        Infinite iterator that yields MAML episodes.
        Each episode contains n_task randomly selected tasks.
        """
        while True:
            yield self._generate_episode()

    def _pad_samples(self, features: List[torch.Tensor]) -> torch.Tensor:
        """Pad variable-length sequences within a task"""
        return pad_sequence(features, batch_first=True, padding_value=0)

    def _generate_episode(self) -> Dict:
        """Generate one MAML episode with n_task tasks"""
        # Randomly select n_task days (with replacement)
        selected_days = random.choices(self.data, k=self.n_task)
        
        support_sets = []
        query_sets = []

        for day_data in selected_days:
            assert 'day_id' in day_data, f"Missing day_id in day_data: {day_data.keys()}"
            
            # Verify sufficient samples
            total_samples = len(day_data["sentenceDat"])
            if total_samples < (self.k_tr + self.k_ts):
                raise ValueError(
                    f"Day {day_data['day_id']} has only {total_samples} samples, "
                    f"but requires {self.k_tr + self.k_ts}"
                )

            # Randomly sample indices without replacement
            indices = random.sample(range(total_samples), self.k_tr + self.k_ts)
            support_idx = indices[:self.k_tr]
            query_idx = indices[self.k_tr:]

            # Process support set
            support = (
                self._pad_samples([torch.as_tensor(day_data["sentenceDat"][i]) for i in support_idx]),
                self._pad_samples([torch.as_tensor(day_data["chars"][i]) for i in support_idx]),
                torch.tensor([day_data["sentenceDat"][i].shape[0] for i in support_idx]),
                torch.tensor([day_data["charLens"][i] for i in support_idx]),
                torch.tensor(day_data['day_id'], dtype=torch.int64),
            )

            # Process query set
            query = (
                self._pad_samples([torch.as_tensor(day_data["sentenceDat"][i]) for i in query_idx]),
                self._pad_samples([torch.as_tensor(day_data["chars"][i]) for i in query_idx]),
                torch.tensor([day_data["sentenceDat"][i].shape[0] for i in query_idx]),
                torch.tensor([day_data["charLens"][i] for i in query_idx]),
                torch.tensor(day_data['day_id'], dtype=torch.int64),
            )

            support_sets.append(support)
            query_sets.append(query)

        return {
            'support': support_sets,
            'query': query_sets
        }
    

class MAMLValDataset(IterableDataset):
    def __init__(self, train_data_list: List[Dict], val_data_list: List[Dict], k_shot: int = 3):
        """
        MAML validation dataset that yields:
        - Support: k_shot samples from each day in train_data
        - Query: ALL samples from val_data (as one batch)
        
        Args:
            train_data_list: List of training days
            val_data_list: List of validation days
            k_shot: Number of support samples per task
        """
        self.train_data = copy.deepcopy(train_data_list)
        self.val_data = copy.deepcopy(val_data_list)
        self.k_shot = k_shot
        
        # Add day_id if not present
        for day_idx, day_data in enumerate(self.train_data):
            day_data['day_id'] = day_data.get('day_id', day_idx)
        for day_idx, day_data in enumerate(self.val_data):
            day_data['day_id'] = day_data.get('day_id', day_idx)

    def __iter__(self):
        """
        Yields one validation episode containing:
        - Support sets: k_shot samples from each training day
        - Query set: ALL validation samples
        """
        while True:
            yield self._generate_validation_episode()

    def _pad_samples(self, features: List[torch.Tensor]) -> torch.Tensor:
        """Pad variable-length sequences"""
        return pad_sequence(features, batch_first=True, padding_value=0)

    def _generate_validation_episode(self) -> Dict:
        """Generate one validation episode with all val data"""
        # Prepare support sets for each training day
        support_sets = []
        for day_data in self.train_data:
            total_samples = len(day_data["sentenceDat"])
            if total_samples < self.k_shot:
                raise ValueError(
                    f"Day {day_data['day_id']} has only {total_samples} samples, "
                    f"but requires {self.k_shot} for support set"
                )
                
            # Randomly select k_shot samples
            support_idx = random.sample(range(total_samples), self.k_shot)
            
            support = (
                self._pad_samples([torch.as_tensor(day_data["sentenceDat"][i]) for i in support_idx]),
                self._pad_samples([torch.as_tensor(day_data["chars"][i]) for i in support_idx]),
                torch.tensor([day_data["sentenceDat"][i].shape[0] for i in support_idx]),
                torch.tensor([day_data["charLens"][i] for i in support_idx]),
                torch.tensor(day_data['day_id'], dtype=torch.int64),
            )
            support_sets.append(support)

        # Prepare query sets (all validation data)
        query_sets = []
        for day_data in self.val_data:
            # Convert to tensors first
            neural_feats = [torch.as_tensor(x) for x in day_data["sentenceDat"]]
            chars = [torch.as_tensor(x) for x in day_data["chars"]]
            
            query = (
                self._pad_samples(neural_feats),
                self._pad_samples(chars),
                torch.tensor([x.shape[0] for x in neural_feats]),
                torch.tensor(day_data["charLens"]),
                torch.tensor(day_data['day_id'], dtype=torch.int64),
            )
            query_sets.append(query)

        return {
            'support': support_sets,
            'query': query_sets
        }


class MAMLDataModule(pl.LightningDataModule):
    def __init__(self, 
                 loaded_data: Dict[str, List[Dict]],
                 n_task: int = 3,
                 k_tr: int = 5,
                 k_ts: int = 5,
                 k_val_fs: int = 3,
                 num_workers: int = 4):
        """
        PyTorch Lightning DataModule for MAML training/validation
        
        Args:
            loaded_data: Dictionary containing:
                - "train": List of training days (each day contains 'sentenceDat', 'chars', etc.)
                - "val": List of validation days
            n_task: Number of tasks per training episode
            k_tr: Number of support samples per training task
            k_ts: Number of query samples per training task  
            k_val_fs: Number of support samples per validation day
            num_workers: Number of workers for data loading
        """
        super().__init__()     
        self.loaded_data = loaded_data
        self.n_task = n_task
        self.k_tr = k_tr
        self.k_ts = k_ts
        self.k_val_fs = k_val_fs
        self.num_workers = num_workers

    def setup(self, stage):
        """
        Initialize datasets for current stage
        Called on every GPU in distributed training
        """
        self.train_ds = MAMLTrainDataset(
                data_list=self.loaded_data["train"],
                n_task=self.n_task,
                k_tr=self.k_tr,
                k_ts=self.k_ts
            )
            
        self.val_ds = MAMLValDataset(
                train_data_list=self.loaded_data["train"],
                val_data_list=self.loaded_data["val"],
                k_shot=self.k_val_fs
            )

    def train_dataloader(self):
        """
        Returns:
            DataLoader: Generates training episodes where each batch contains:
                - 'support': List of n_task support sets
                - 'query': Corresponding query sets
        """
        return DataLoader(
            dataset=self.train_ds,
            batch_size=1,  # Each episode is one batch
            num_workers=self.num_workers,
            collate_fn=lambda x: x[0],  # Directly return the episode
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0
        )



    def val_dataloader(self):
        """
        Returns:
            DataLoader: Generates validation episodes where each batch contains:
                - 'support': Support sets from all training days
                - 'query': All validation data grouped by day
        """
        return DataLoader(
            dataset=self.val_ds,
            batch_size=1,  # Full validation set as one batch
            num_workers=self.num_workers,
            collate_fn=lambda x: x[0],  # Directly return the episode
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0
        )

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        def _to_device(data):
            if isinstance(data, dict):
                return {k: _to_device(v) for k, v in data.items()}
            elif isinstance(data, (list, tuple)):
                return type(data)(_to_device(v) for v in data)
            elif torch.is_tensor(data):
                return data.to(device)
            else:
                return data

        return _to_device(batch)


    @staticmethod
    def inspect_batch(batch: Dict):
        """
        Debug utility to inspect batch structure
        
        Args:
            batch: Dictionary containing:
                - 'support': List of (neural, chars, lengths) tuples
                - 'query': List of (neural, chars, lengths) tuples
        """
        print("\n=== Batch Structure ===")
        print(f"Batch dataset has {len(batch['support'])} tasks:")
        print("-"*50)
        
        # Support set info
        print(f"Support Sets:")
        for i, (neural, chars, neural_lens, char_lens, day_id) in enumerate(batch['support']):
            print(f"  Task {i}: (samples: {len(neural_lens)})")
            print(f"    Day id: {day_id.item()}")
            print(f"    Neural: {neural.shape} (lengths: {neural_lens.tolist()})")
            print(f"    Chars: {chars.shape} (lengths: {char_lens.tolist()})")
        
        # Query set info
        print(f"\nQuery Sets:")
        for i, (neural, chars, neural_lens, char_lens, day_id) in enumerate(batch['query']):
            print(f"  Task {i}: (samples: {len(neural_lens)})")
            print(f"    Day id: {day_id.item()}")
            print(f"    Neural: {neural.shape} (lengths: {neural_lens.tolist()})")
            print(f"    Chars: {chars.shape} (lengths: {char_lens.tolist()})")