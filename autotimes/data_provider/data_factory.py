"""Fixed Monthly source data factory."""

from torch.utils.data import DataLoader

from autotimes.data_provider.data_loader import Dataset_TSF


def data_provider(args, flag):
    if flag == "test":
        shuffle = False
        drop_last = False
    elif flag == "val":
        shuffle = args.val_set_shuffle
        drop_last = False
    else:
        shuffle = True
        drop_last = args.drop_last
    dataset = Dataset_TSF(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.token_len],
        seasonal_patterns=args.seasonal_patterns,
        drop_short=args.drop_short,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        drop_last=drop_last,
    )
    print(flag, len(dataset))
    return dataset, loader

