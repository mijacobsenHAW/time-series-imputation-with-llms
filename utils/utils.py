import matplotlib.pyplot as plt


def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    lr_adjust = {"epoch": 0.001}  # fallback

    if args.adjust_lr == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    if args.adjust_lr == 'type7':
        lr_adjust = {epoch: args.learning_rate * (0.7 ** ((epoch - 1) // 1))}
    if args.adjust_lr == 'type6':
        lr_adjust = {epoch: args.learning_rate * (0.6 ** ((epoch - 1) // 1))}
    elif args.adjust_lr == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }

    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')


def count_trainable_parameters(model):
    """
    Counts the total and trainable number of parameters in a given PyTorch model.

    Args:
        model (torch.nn.Module): The model for which to count parameters.

    Returns:
        tuple: A tuple containing the total number of parameters and the number of trainable parameters.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params