import numpy as np

NAS_BENCH_201 = ['none', 'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3']


def get_adj_matrix(arch):
    adj_matrix = np.zeros(shape=(4, 4))
    for i in range(len(adj_matrix)):
        for j in range(i + 1, 4):
            adj_matrix[i][j] = 1

    op_list = arch.split('+')
    edge_op_list = list()
    for node, edge_op in enumerate(op_list):
        edge_op = edge_op.split('|')
        for _op in edge_op:
            if len(_op) <= 1:
                continue
            _idx = int(_op[-1])
            _op = _op[:-2]
            edge_op_list.append(NAS_BENCH_201.index(_op))
            if _op == 'none':
                assert adj_matrix[_idx][node + 1] == 1
                adj_matrix[_idx][node + 1] = 0
    return adj_matrix, edge_op_list


def distill(result):
    result = result.split('\n')
    cifar10_v = result[3].replace(' ', '').split(':')
    cifar10 = result[5].replace(' ', '').split(':')
    cifar100 = result[7].replace(' ', '').split(':')
    imagenet16 = result[9].replace(' ', '').split(':')

    cifar10_train = float(cifar10[1].strip(',test')[-7:-2].strip('='))
    cifar10_test = float(cifar10[2][-7:-2].strip('='))
    cifar10_valid = float(cifar10_v[2][-7:-2].strip('='))
    cifar100_train = float(cifar100[1].strip(',valid')[-7:-2].strip('='))
    cifar100_valid = float(cifar100[2].strip(',test')[-7:-2].strip('='))
    cifar100_test = float(cifar100[3][-7:-2].strip('='))
    imagenet16_train = float(imagenet16[1].strip(',valid')[-7:-2].strip('='))
    imagenet16_valid = float(imagenet16[2].strip(',test')[-7:-2].strip('='))
    imagenet16_test = float(imagenet16[3][-7:-2].strip('='))

    return cifar10_train, cifar10_valid, cifar10_test, cifar100_train, cifar100_valid, \
           cifar100_test, imagenet16_train, imagenet16_valid, imagenet16_test


if __name__ == '__main__':
    nasbench201_dict = {}
    arch_pointer = 0
    arch_counter = 0
    line_counter = 0
    arch_result = list()
    # 201_info.txt is simply extracted from the official file: NAS-Bench-201-v1_1-096897.7z
    with open('./nasbench201/201_info.txt', 'r') as f:
        for line in f:
            if arch_pointer == 0:
                arch = line
                adj_matrix, operation = get_adj_matrix(arch)
            arch_result.append(line)
            arch_pointer += 1
            if arch_pointer == 10:
                cifar10_train, cifar10_valid, cifar10_test, cifar100_train, cifar100_valid, \
                cifar100_test, imagenet16_train, imagenet16_valid, imagenet16_test = distill("".join(arch_result))
                arch_result = list()
                arch_pointer = 0
                model_dict = {
                    'arch': arch,
                    'adj_matrix': adj_matrix,
                    'operation': operation,
                    'cifar10_train': cifar10_train,
                    'cifar10_valid': cifar10_valid,
                    'cifar10_test': cifar10_test,
                    'cifar100_train': cifar100_train,
                    'cifar100_valid': cifar100_valid,
                    'cifar100_test': cifar100_test,
                    'imagenet16_train': imagenet16_train,
                    'imagenet16_valid': imagenet16_valid,
                    'imagenet16_test': imagenet16_test,
                }
                nasbench201_dict.update({str(arch_counter):model_dict})
                arch_counter += 1
                if arch_counter % 1000 == 0:
                    print(arch_counter)
            line_counter += 1
            # if line_counter == 100:
            #     np.save('nasbench201_dict.npy', nasbench201_dict)
            #     import sys
            #     sys.exit()
        np.save('nasbench201_dict_with_arch.npy', nasbench201_dict)

    nasbench201_dict = np.load('nasbench201_dict_with_arch.npy', allow_pickle=True).item()
    for key in nasbench201_dict.keys():
        print(nasbench201_dict[key])
        break
