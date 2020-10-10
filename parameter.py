import argparse

def str2bool(v):
    return v.lower() in ('true')

def get_parameters():

    parser = argparse.ArgumentParser()

    # Model hyper-parameters
    parser.add_argument('--model', type=str, default='sagan', choices=['baseline', 'sn_on_g_d', 'sn_on_g_d_ttur', 'sagan'])
    parser.add_argument('--im_size', type=int, default=64) #32
    parser.add_argument('--im_center_corp', type=int, default=0) #0 means - no center corp for images
    parser.add_argument('--z_dim', type=int, default=128)
    parser.add_argument('--g_ch', type=int, default=64) #256
    parser.add_argument('--d_ch', type=int, default=64) #128
    parser.add_argument('--feat_k', type=int, default=32)

    # Training setting
    parser.add_argument('--version', type=str, default='sagan')
    parser.add_argument('--load_checkpoint', type=str2bool, default=False)
    parser.add_argument('--final_checkpoint_name', type=str, default='sagan.pt')
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
    parser.add_argument('--dataset', type=str, default='cifar', choices=['lsun', 'cifar', 'gwb', 'celeba'])

    # Path
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--model_save_path', type=str, default='./checkpoints')
    parser.add_argument('--sample_path', type=str, default='./samples')

    # epoch size
    parser.add_argument('--calc_score_step', type=int, default=500)
    parser.add_argument('--model_save_epoch', type=float, default=5)
    parser.add_argument('--sample_save_step', type=float, default=1000)


    return parser.parse_args()