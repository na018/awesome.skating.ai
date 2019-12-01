from skatingAI.Data.skating_dataset import get_dataset, get_dataset_flat, get_batch, check_empty_frames
if __name__ == "__main__":
    batch_size = 5

    #ds = get_dataset()
    ds = get_dataset_flat()
    #batches = get_batch(ds, batch_size)
    #ds_x, ds_y = get_dataset_flat()
    # check_empty_frames()
    # save_frames_to_video()
    #print(ds_x.shape, ds_y.shape)
    #print(f"You've got {len(batches)} random batches")
