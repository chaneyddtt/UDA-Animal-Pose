import os.path as osp
def print_options(args):
    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in sorted(vars(args).items()):
        comment = ''
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '----------------- End -------------------'
    print(message)

    # save to the disk
    file_name = osp.join(args.checkpoint, 'opt.txt')
    with open(file_name, 'wt') as args_file:
        args_file.write(message)
        args_file.write('\n')