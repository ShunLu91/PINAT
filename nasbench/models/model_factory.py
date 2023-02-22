from models.PINAT.pinat import PINATModel


def create_model(args):
    pos_enc_dim_dict = {'101': 7, '201': 4}
    net = PINATModel(
        bench=args.bench, pos_enc_dim=pos_enc_dim_dict[args.bench],
        adj_type='adj_lapla', n_layers=3, n_head=4, pine_hidden=16, linear_hidden=96,
        n_src_vocab=5, d_word_vec=80, d_k=64, d_v=64, d_model=80, d_inner=512,
    )

    return net
