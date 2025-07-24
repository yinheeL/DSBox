# active_selection_tool/strategies/loss_pred_loss.py

import torch

def LossPredLoss(input, target, margin=1.0, reduction='mean'):
    """
    Ranking Loss, used to train the loss prediction head.
    Input:
    input (Tensor): shape = (batch_size,) predicted loss [ĥℓ_1, ĥℓ_2, ...].
    target (Tensor): shape = (batch_size,) true loss [ℓ_1, ℓ_2, ...].
    First, pair the batches one by one (such as i ↔ batch_size-1-i), and then perform hinge loss on the difference between the two rankings.
    """
    assert input.shape == target.shape
    batch_size = input.size(0)
    assert batch_size % 2 == 0,

    # 翻转，做两两配对
    input_diff = input - input.flip(0)
    target_diff = target - target.flip(0)


    half = batch_size // 2
    input_diff = input_diff[:half]
    target_diff = target_diff[:half]


    one = 2 * torch.sign(torch.clamp(target_diff, min=0)) - 1


    loss = torch.clamp(margin - one * input_diff, min=0)
    if reduction == 'mean':
        return loss.mean()
    return loss.sum()
