import argparse

def str2bool(v):
    return v.lower() in ('true')

def get_parameters():

    parser = argparse.ArgumentParser()

    # Model hyper-parameters
    parser.add_argument('--model', type=str, default='sagan', choices=['baseline', 'sn_on_g_d', 'sn_on_g_d_ttur', 'sagan16', 'sagan32'])
    parser.add_argument('--im_size', type=int, default=64)
    parser.add_argument('--z_dim', type=int, default=128)
    parser.add_argument('--g_ch', type=int, default=64)
    parser.add_argument('--d_ch', type=int, default=64)
    parser.add_argument('--version', type=str, default='sagan')

    # Training setting
    parser.add_argument('--num_epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--optimizer_type', type=str, default='Adam')
    parser.add_argument('--baseline_lr', type=float, default=0.0002)
    parser.add_argument('--ttur_gen_lr', type=float, default=0.0001)
    parser.add_argument('--ttur_dis_lr', type=float, default=0.0004)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--beta1', type=float, default=0.0)
    parser.add_argument('--beta2', type=float, default=0.9)

    # Misc
    parser.add_argument('--dataset', type=str, default='cifar', choices=['lsun', 'cifar'])
    parser.add_argument('--use_tensorboard', type=str2bool, default=False)

    # Path
    parser.add_argument('--image_path', type=str, default='./data')
    #parser.add_argument('--log_path', type=str, default='./logs')
    parser.add_argument('--model_save_path', type=str, default='./checkpoints')
    parser.add_argument('--sample_path', type=str, default='./samples')
    #parser.add_argument('--attn_path', type=str, default='./attn')

    # epoch size
    parser.add_argument('--model_calc_score', type=int, default=1)
    parser.add_argument('--sample_save_epoch', type=int, default=10)
    parser.add_argument('--model_save_epoch', type=float, default=10)


    return parser.parse_args()