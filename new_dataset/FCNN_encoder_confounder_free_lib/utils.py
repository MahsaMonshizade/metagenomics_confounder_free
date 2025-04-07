import torch
import math
from torch.utils.data import DataLoader, TensorDataset

def create_stratified_dataloader(x_train, y_train, batch_size):
    """
    Create a stratified DataLoader that ensures class proportions are maintained in each batch.
    """
    # Compute class counts and proportions
    labels = y_train.squeeze()
    unique_labels = labels.unique()
    class_counts = {label.item(): (labels == label).sum().item() for label in unique_labels}
    total_samples = len(labels)
    class_proportions = {label: count / total_samples for label, count in class_counts.items()}

    # Compute samples per class per batch
    samples_per_class = {}
    remainders = {}
    total_samples_in_batch = 0

    for label, proportion in class_proportions.items():
        exact_samples = proportion * batch_size
        samples = int(math.floor(exact_samples))
        remainder = exact_samples - samples
        samples_per_class[label] = samples
        remainders[label] = remainder
        total_samples_in_batch += samples

    # Distribute remaining slots based on the largest remainders
    remaining_slots = batch_size - total_samples_in_batch
    sorted_labels = sorted(remainders.items(), key=lambda x: x[1], reverse=True)
    for i in range(remaining_slots):
        label = sorted_labels[i % len(sorted_labels)][0]
        samples_per_class[label] += 1

    # Get indices for each class and shuffle them
    class_indices = {label.item(): (labels == label).nonzero(as_tuple=True)[0] for label in unique_labels}
    for label in class_indices:
        indices = class_indices[label]
        class_indices[label] = indices[torch.randperm(len(indices))]

    # Generate stratified batches
    def stratified_batches(class_indices, samples_per_class, batch_size):
        batches = []
        class_cursors = {label: 0 for label in class_indices}
        num_samples = sum([len(indices) for indices in class_indices.values()])
        num_batches = math.ceil(num_samples / batch_size)

        for _ in range(num_batches):
            batch = []
            for label, indices in class_indices.items():
                cursor = class_cursors[label]
                samples = samples_per_class[label]
                # If we've run out of samples for this class, skip
                if cursor >= len(indices):
                    continue
                # Adjust samples if not enough samples left
                if cursor + samples > len(indices):
                    samples = len(indices) - cursor
                batch_indices = indices[cursor:cursor+samples]
                batch.extend(batch_indices.tolist())
                class_cursors[label] += samples
            # Shuffle batch indices
            if batch:
                batch = torch.tensor(batch)[torch.randperm(len(batch))].tolist()
                batches.append(batch)
        return batches

    batches = stratified_batches(class_indices, samples_per_class, batch_size)

    # Create a custom BatchSampler
    class StratifiedBatchSampler(torch.utils.data.BatchSampler):
        def __init__(self, batches):
            self.batches = batches

        def __iter__(self):
            for batch in self.batches:
                yield batch

        def __len__(self):
            return len(self.batches)

    # Create a dataset and a DataLoader with the custom BatchSampler
    dataset = TensorDataset(x_train, y_train)
    batch_sampler = StratifiedBatchSampler(batches)
    data_loader = DataLoader(dataset, batch_sampler=batch_sampler)

    return data_loader
