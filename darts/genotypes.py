from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

PRIMITIVES = [
    # 'none',
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5'
]

# pinat_c0=97.39
pinat_c0 = Genotype(
    normal=[('dil_conv_3x3', 0), ('sep_conv_5x5', 1), ('dil_conv_5x5', 0), ('sep_conv_3x3', 1), ('avg_pool_3x3', 0),
            ('sep_conv_3x3', 3), ('skip_connect', 0), ('sep_conv_3x3', 3)], normal_concat=[2, 3, 4, 5],
    reduce=[('sep_conv_5x5', 0), ('sep_conv_3x3', 1), ('dil_conv_3x3', 0), ('dil_conv_3x3', 2), ('skip_connect', 0),
            ('avg_pool_3x3', 3), ('skip_connect', 0), ('avg_pool_3x3', 2)], reduce_concat=[2, 3, 4, 5])
# pinat_c1=97.42
pinat_c1 = Genotype(
    normal=[('dil_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('sep_conv_5x5', 0),
            ('skip_connect', 3), ('dil_conv_3x3', 2), ('dil_conv_5x5', 3)], normal_concat=[2, 3, 4, 5],
    reduce=[('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('sep_conv_5x5', 2), ('avg_pool_3x3', 1),
            ('sep_conv_5x5', 3), ('dil_conv_5x5', 0), ('dil_conv_3x3', 3)], reduce_concat=[2, 3, 4, 5])
# pinat_c2=97.58
pinat_c2 = Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('dil_conv_5x5', 2), ('skip_connect', 0),
            ('dil_conv_5x5', 1), ('sep_conv_3x3', 0), ('dil_conv_3x3', 2)], normal_concat=[2, 3, 4, 5],
    reduce=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('dil_conv_3x3', 1), ('avg_pool_3x3', 2),
            ('skip_connect', 3), ('max_pool_3x3', 0), ('dil_conv_5x5', 3)], reduce_concat=[2, 3, 4, 5])



