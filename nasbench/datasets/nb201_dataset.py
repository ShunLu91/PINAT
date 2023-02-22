import random
import numpy as np

import torch
from torch.utils.data import Dataset


class Nb201DatasetPINAT(Dataset):
    def __init__(self, split, candidate_ops=5, data_type='train', data_set='cifar10'):
        self.nasbench201_dict = np.load('data/nasbench201/nasbench201_dict.npy', allow_pickle=True).item()
        self.sample_range = list()
        self.candidate_ops = candidate_ops
        if data_type == 'train':
            self.sample_range = random.sample(range(0, len(self.nasbench201_dict)), int(split))
        elif data_type == 'valid':
            self.sample_range = random.sample(range(0, len(self.nasbench201_dict)), int(split))
        elif data_type == 'test':
            self.sample_range = range(0, len(self.nasbench201_dict))
        else:
            raise ValueError('Wrong data_type!')

        self.data_type = data_type
        self.data_set = data_set
        if self.data_set == 'cifar10':
            self.val_mean, self.val_std = 0.836735, 0.128051
            self.test_mean, self.test_std = 0.870563, 0.129361
        elif self.data_set == 'cifar100':
            self.val_mean, self.val_std = 0.612818, 0.121428
            self.test_mean, self.test_std = 0.613878, 0.121719
        elif self.data_set == 'imagenet16':
            self.val_mean, self.val_std = 0.337928, 0.092423
            self.test_mean, self.test_std = 0.335682, 0.095140
        else:
            raise ValueError('Wrong data_set!')
        self.max_edge_num = 6

        # # Compute mean and std of acc
        # dataset = 'cifar100'
        # val_acc_list = []
        # test_acc_list = []
        # for index in self.nasbench201_dict.keys():
        #     val_acc = self.nasbench201_dict[index]['%s_valid' % dataset]
        #     test_acc = self.nasbench201_dict[index]['%s_test' % dataset]
        #     val_acc_list.append(val_acc/100)
        #     test_acc_list.append(test_acc/100)
        #     if int(index) % 1000 == 0:
        #         print(index)
        # print('Dataset: %s' % dataset)
        # print('self.val_mean, self.val_std = %f, %f' % (np.mean(val_acc_list), np.std(val_acc_list)))
        # print('self.test_mean, self.test_std = %f, %f' % (np.mean(test_acc_list), np.std(test_acc_list)))
        # exit()

    def __len__(self):
        return len(self.sample_range)

    def _check(self, item):
        n = item["num_vertices"]
        ops = item["operations"]
        adjacency = item["adjacency"]
        mask = item["mask"]
        assert np.sum(adjacency) - np.sum(adjacency[:n, :n]) == 0
        assert np.sum(ops) == n
        assert np.sum(ops) - np.sum(ops[:n]) == 0
        assert np.sum(mask) == n and np.sum(mask) - np.sum(mask[:n]) == 0

    def normalize(self, num):
        if self.data_type == 'train':
            return (num - self.val_mean) / self.val_std
        elif self.data_type == 'test':
            return (num - self.test_mean) / self.test_std
        else:
            raise ValueError('Wrong data_type!')

    def denormalize(self, num):
        if self.data_type == 'train':
            return num * self.val_std + self.val_mean
        elif self.data_type == 'test':
            return num * self.test_std + self.test_mean
        else:
            raise ValueError('Wrong data_type!')

    def _rand_flip(self, batch_pos):
        batch_lap_pos_enc = torch.from_numpy(batch_pos)
        sign_flip = torch.rand(batch_lap_pos_enc.size(1))
        sign_flip[sign_flip >= 0.5] = 1.0
        sign_flip[sign_flip < 0.5] = -1.0
        batch_lap_pos_enc = batch_lap_pos_enc * sign_flip.unsqueeze(0)
        return batch_lap_pos_enc

    def _generate_lapla_matrix(self, adj_matrix):
        degree = np.diag(np.sum(adj_matrix, axis=1))
        unnormalized_lapla = degree - adj_matrix
        return unnormalized_lapla

    def _convert_arch_to_seq(self, matrix, ops):
        # 'none', 'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3'
        seq = []
        n = len(matrix)
        max_n = 4
        for col in range(1, max_n):
            if col >= n:
                seq += [0 for i in range(col)]
                seq.append(0)
            else:
                for row in range(col):
                    seq.append(matrix[row][col]+1)
                    if ops[col+row] == 0:  # none
                        seq.append(3)
                    elif ops[col+row] == 1:  # skip_connect
                        seq.append(4)
                    elif ops[col+row] == 2:  # nor_conv_1x1
                        seq.append(5)
                    elif ops[col+row] == 3:  # nor_conv_3x3
                        seq.append(6)
                    elif ops[col+row] == 4:  # avg_pool_3x3
                        seq.append(7)
        return seq

    def __getitem__(self, index):
        index = self.sample_range[index]
        # val_acc, test_acc = self.metrics[index, -1, self.seed, -1, 2:]
        val_acc = self.nasbench201_dict[str(index)]['%s_valid' % self.data_set]
        test_acc = self.nasbench201_dict[str(index)]['%s_test' % self.data_set]
        adjacency = self.nasbench201_dict[str(index)]['adj_matrix']
        lapla = self._generate_lapla_matrix(adj_matrix=adjacency)
        operation = np.array(self.nasbench201_dict[str(index)]['operation'], dtype=np.float32)
        ops_onehot = np.array([[i == k for i in range(self.candidate_ops)]
                               for k in operation], dtype=np.float32)
        n = np.linalg.matrix_rank(adjacency) + 1

        seq = self._convert_arch_to_seq(adjacency, operation)
        encoder_input = seq
        decoder_input = [0] + encoder_input[:-1]

        # links
        edge_index = []
        for i in range(adjacency.shape[0]):
            idx_list = np.where(adjacency[i])[0].tolist()
            for j in idx_list:
                edge_index.append([i, j])
        if np.sum(edge_index) == 0:
            edge_index = []
            for i in range(adjacency.shape[0]):
                for j in range(adjacency.shape[0] - 1, i, -1):
                    edge_index.append([i, j])

        edge_num = len(edge_index)
        pad_num = self.max_edge_num - edge_num
        if pad_num > 0:
            edge_index = np.pad(np.array(edge_index), ((0, pad_num), (0, 0)), 'constant', constant_values=(0, 0))
        edge_index = torch.tensor(edge_index, dtype=torch.int64)
        edge_index = edge_index.transpose(1, 0)

        result = {
            # "num_vertices": n,
            "num_vertices": 4,
            "edge_num": edge_num,
            "adjacency": np.array(adjacency, dtype=np.float32),
            "lapla": lapla,
            # "lapla_nor": self.lapla_nor[index],
            "operations": ops_onehot,
            "features": torch.from_numpy(operation).long(),
            # "lap_pos_enc": self._rand_flip(self.lap_pos_enc[index]),
            # "lap_pos_enc": self.lap_pos_enc[index],
            # "mask": np.array([i < n for i in range(7)], dtype=np.float32),
            "val_acc": torch.tensor(self.normalize(val_acc/100), dtype=torch.float32),
            "test_acc": torch.tensor(self.normalize(test_acc/100), dtype=torch.float32),
            "val_acc_ori": val_acc,
            "test_acc_ori": test_acc,
            'encoder_input': torch.LongTensor(encoder_input),
            'decoder_input': torch.LongTensor(decoder_input),
            'decoder_target': torch.LongTensor(encoder_input),
            "edge_index_list": edge_index,
        }
        return result
