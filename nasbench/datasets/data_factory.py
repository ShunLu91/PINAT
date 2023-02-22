from datasets.nb101_dataset import Nb101DatasetPINAT
from datasets.nb201_dataset import Nb201DatasetPINAT
from torch.utils.data import DataLoader


def create_dataloader(args):
    # load dataset
    if args.bench == '101':
        train_set = Nb101DatasetPINAT(split=args.train_split, data_type='train')
        test_set = Nb101DatasetPINAT(split=args.eval_split, data_type='test')
    elif args.bench == '201':
        train_set = Nb201DatasetPINAT(split=int(args.train_split), data_type='train', data_set=args.dataset)
        test_set = Nb201DatasetPINAT(split='all', data_type='test', data_set=args.dataset)
    else:
        raise ValueError('No defined NAS bench!')

    # initialize dataloader
    train_loader = DataLoader(train_set, batch_size=args.train_batch_size,
                              num_workers=0, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=args.eval_batch_size, shuffle=False,
                             num_workers=2 if args.eval_split == '100' else 16)
    return train_loader, test_loader, train_set, test_set
