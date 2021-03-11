import torch
from torch.nn.functional import cross_entropy
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

def masked_loss(scores,targets,tgt_pad_index = 0 ):
    """
    Calculate cross entropy loss for a given scores and targets, ignoring loss for padding_index
    @params scores(Tensor): output scores - shape(batch, tgt_seq_len-1, tgt_vocab_size)
    @params target(Tensor): gold target - shape(batch,tgt_seq_len) - Discard 1st time step
    @params tgt_pad_index(int): Target pad index for which loss is ignored
    
    @returns loss(Tensor): loss size zero tensor
    """
    targets = targets[:,1:]
    tgt_vocab_size = scores.shape[2]
    scores_reshaped = scores.view(-1,tgt_vocab_size)
    targets_reshape = targets.reshape(-1)
    loss = cross_entropy(scores_reshaped,targets_reshape,ignore_index = tgt_pad_index)
    return loss

def masked_accuracy(scores,targets,tgt_pad_index = 0):
    """
    Calculate accuracy for a given scores and targets, ignoring it for padding_index
    @params scores(Tensor): output scores - shape(batch, tgt_seq_len-1, tgt_vocab_size)
    @params target(Tensor): gold target - shape(batch,tgt_seq_len) - Discard 1st time step
    @params tgt_pad_index(int): Target pad index for which loss is ignored
    
    @returns accuracy(Float): Accuracy
    """
    with torch.no_grad():
        targets = targets[:,1:]
        mask = (targets!=tgt_pad_index)
        num_words = mask.sum().float()
        preds = torch.argmax(scores, dim=-1)
        truths = (preds == targets)*mask
        accuracy = truths.sum()/num_words
    return accuracy


