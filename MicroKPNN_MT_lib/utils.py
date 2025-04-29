import torch
import math
from torch.utils.data import DataLoader, TensorDataset, BatchSampler

def create_stratified_dataloader(
    x_train: torch.Tensor,
    y_train_disease: torch.Tensor,
    y_train_confounder: torch.Tensor,
    batch_size: int,
    pad_incomplete: bool = True
) -> DataLoader:
    """
    Create a stratified DataLoader for a two‐task model (disease + confounder).

    Stratification is done on the disease labels, but the loader will return
    both targets so you can compute both losses downstream.

    Parameters:
      x_train (Tensor):      feature matrix, shape (N, D)
      y_train_disease (Tensor): binary (or categorical) disease labels, shape (N,1)
      y_train_confounder (Tensor): binary (or categorical) confounder labels, shape (N,1)
      batch_size (int):      number of samples per batch
      pad_incomplete (bool): if True, pad the last batch up to batch_size

    Returns:
      DataLoader: yields tuples (x_batch, y_disease_batch, y_confounder_batch)
    """
    # 1) Prepare stratification on disease labels
    labels = y_train_disease.squeeze()              # shape (N,)
    unique_labels = labels.unique()
    total = len(labels)
    # compute class proportions
    class_counts = {int(lbl): int((labels == lbl).sum()) for lbl in unique_labels}
    class_props  = {lbl: cnt/total for lbl, cnt in class_counts.items()}

    # determine how many of each label per batch
    samples_per_class = {}
    remainders = {}
    allocated = 0
    for lbl, prop in class_props.items():
        exact = prop * batch_size
        base  = math.floor(exact)
        samples_per_class[lbl] = base
        remainders[lbl] = exact - base
        allocated += base

    # distribute leftover slots by largest fractional remainder
    leftovers = batch_size - allocated
    for lbl, _ in sorted(remainders.items(), key=lambda x: -x[1])[:leftovers]:
        samples_per_class[lbl] += 1

    # 2) Shuffle and index each class
    indices_per_class = {
        int(lbl): (labels == lbl).nonzero(as_tuple=True)[0][torch.randperm(class_counts[int(lbl)])]
        for lbl in unique_labels
    }

    # 3) Build list of batch‐indices
    def make_batches():
        cursors = {lbl: 0 for lbl in indices_per_class}
        num_batches = math.ceil(total / batch_size)
        for _ in range(num_batches):
            batch_idxs = []
            for lbl, idxs in indices_per_class.items():
                start = cursors[lbl]
                take  = samples_per_class[lbl]
                end   = min(start + take, len(idxs))
                batch_piece = idxs[start:end].tolist()
                batch_idxs.extend(batch_piece)
                cursors[lbl] = end
            # pad if needed
            if pad_incomplete and len(batch_idxs) < batch_size:
                pad_n = batch_size - len(batch_idxs)
                batch_idxs.extend(
                    torch.tensor(batch_idxs)[torch.randperm(len(batch_idxs))][:pad_n].tolist()
                )
            # shuffle inside batch
            yield torch.tensor(batch_idxs)[torch.randperm(len(batch_idxs))].tolist()

    batches = list(make_batches())

    # 4) Create the sampler and DataLoader
    class StratifiedBatchSampler(BatchSampler):
        def __init__(self, batches): 
            super().__init__(None, batch_size, False)
            self.batches = batches
        def __iter__(self):
            for b in self.batches: yield b
        def __len__(self):
            return len(self.batches)

    dataset = TensorDataset(x_train, y_train_disease, y_train_confounder)
    sampler = StratifiedBatchSampler(batches)
    return DataLoader(dataset, batch_sampler=sampler)
