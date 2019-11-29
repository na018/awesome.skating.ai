from skatingAI.Data.skating_dataset import get_dataset, get_dataset_flat, get_batch
if __name__ == "__main__":
    batch_size = 5

    ds = get_dataset()
    batches = get_batch(ds, batch_size)
    ds_x, ds_y = get_dataset_flat()
    print(ds_x.shape, ds_y.shape)
    print(f"You've got {len(batches)} random batches")
