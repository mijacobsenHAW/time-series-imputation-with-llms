from torch.utils.data import DataLoader
from data_provider.datasets import DatasetEttHour, DatasetEttMinute, DatasetCustom

data_dict = {
    'ETTh1': DatasetEttHour,
    'ETTh2': DatasetEttHour,
    'ETTm1': DatasetEttMinute,
    'ETTm2': DatasetEttMinute,
    'custom': DatasetCustom
}


def get_dataloader_and_dataset(args, flag: str):
    """
    Creates a DataLoader instance for a specified dataset. The function dynamically selects the dataset
    based on the provided arguments, configures it for training, testing, or validation, and creates
    a DataLoader for batch processing of data.

    Args:
        args (Namespace): A Namespace object containing configuration options for the dataset and DataLoader.
                          Expected attributes include `data` (dataset identifier), `root` (root path for dataset files),
                          `data_path` (relative path to the dataset file), `sequence_len`, `label_len`, `prediction_len`,
                          `embed` (embedding type), `percent` (data split percentage), and `batch_size`.
        flag (str): Indicates the purpose for which the DataLoader is created. Expected values are 'train', 'test', or 'valid'.

    Returns:
        tuple: A tuple containing the instantiated dataset and DataLoader for the specified configuration. The dataset object
               provides access to raw data samples, while the DataLoader allows for batch-wise iteration over the dataset,
               with options for shuffling and batching configured based on the `flag` parameter.

    Note:
        - The `timeenc` attribute in `args` determines whether time features are encoded. It is set based on the value of `args.embed`.
        - The `percent` attribute specifies the percentage of data to be used, which is particularly relevant for test splits.
        - Batch size, shuffle flag, and frequency parameters are adjusted based on whether the DataLoader is for training, testing, or validation.
        - This function supports extending with additional datasets by adding them to the `data_dict` and ensuring they comply with the expected interface.
    """
    data_set = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1
    percent = args.percent

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = 1  # for testing
        freq = args.frequency
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size  # bsz for train and valid
        freq = args.frequency

    data_set = data_set(
        root_path=args.root,
        data_path=args.data_path,
        flag=flag,
        size=[args.sequence_len, args.label_len, args.prediction_len],
        timeenc=timeenc,
        percent=percent,
        freq=freq,
    )
    print(f"{flag.capitalize()} dataset size:", len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=0,  # Consider adjusting based on system capabilities
        drop_last=drop_last)

    return data_set, data_loader
