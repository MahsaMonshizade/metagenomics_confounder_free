import torch
import math
from torch.utils.data import DataLoader, TensorDataset

def create_stratified_dataloader(x_train, y_train, batch_size):
    """
    Create a stratified DataLoader so that each batch reflects the overall 
    class distribution.

    Parameters:
      x_train (torch.Tensor): Input features.
      y_train (torch.Tensor): Target labels.
      batch_size (int): Desired batch size.

    Returns:
      DataLoader: A DataLoader that yields stratified batches.
    
    Note:
      If the number of distinct classes exceeds the batch size, not all classes 
      can be present in every batch. In that case, consider increasing the batch size.
    """
    # Remove extra dimensions if necessary.
    labels = y_train.squeeze()
    unique_labels = labels.unique()

    # Count samples per class.
    class_counts = {label.item(): (labels == label).sum().item() for label in unique_labels}
    total_samples = len(labels)
    class_proportions = {label: count / total_samples for label, count in class_counts.items()}

    # Determine the number of samples per class per batch.
    samples_per_class = {}
    remainders = {}
    total_allocated = 0
    for label, proportion in class_proportions.items():
        exact_count = proportion * batch_size
        count = int(math.floor(exact_count))
        samples_per_class[label] = count
        remainders[label] = exact_count - count
        total_allocated += count

    # Allocate remaining slots based on highest decimal remainder.
    remaining_slots = batch_size - total_allocated
    for label, _ in sorted(remainders.items(), key=lambda item: item[1], reverse=True):
        if remaining_slots <= 0:
            break
        samples_per_class[label] += 1
        remaining_slots -= 1

    # Get indices for each class and shuffle them.
    class_indices = {label.item(): (labels == label).nonzero(as_tuple=True)[0] for label in unique_labels}
    for label in class_indices:
        indices = class_indices[label]
        class_indices[label] = indices[torch.randperm(len(indices))]

    def stratified_batches(class_indices, samples_per_class):
        """
        Generate a list of stratified batches using the allocated samples_per_class.
        """
        batches = []
        class_cursors = {label: 0 for label in class_indices}
        num_samples = sum([len(indices) for indices in class_indices.values()])
        num_batches = math.ceil(num_samples / batch_size)
        for _ in range(num_batches):
            batch = []
            # For each class, take the appropriate number of indices.
            for label, indices in class_indices.items():
                start = class_cursors[label]
                count = samples_per_class[label]
                end = start + count
                if start < len(indices):
                    # Append indices from the current class.
                    batch_indices = indices[start: min(end, len(indices))]
                    batch.extend(batch_indices.tolist())
                    class_cursors[label] += count
            if batch:
                # Shuffle the batch indices.
                batch = torch.tensor(batch)[torch.randperm(len(batch))].tolist()
                batches.append(batch)
        return batches

    batches = stratified_batches(class_indices, samples_per_class)

    # Custom batch sampler.
    class StratifiedBatchSampler(torch.utils.data.Sampler):
        def __init__(self, batches):
            self.batches = batches

        def __iter__(self):
            for batch in self.batches:
                yield batch

        def __len__(self):
            return len(self.batches)

    dataset = TensorDataset(x_train, y_train)
    sampler = StratifiedBatchSampler(batches)
    g = torch.Generator()
    g.manual_seed(42)
    loader = DataLoader(dataset, batch_sampler=sampler, shuffle=False, num_workers=4,worker_init_fn=lambda wid: torch.manual_seed(42 + wid), generator=g)
    return loader
