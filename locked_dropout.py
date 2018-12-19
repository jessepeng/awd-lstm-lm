import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence, pack_padded_sequence


class LockedDropout(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, dropout=0.5):
        if not self.training or not dropout:
            return x
        sequence_lengths = None
        if isinstance(x, PackedSequence):
            x, sequence_lengths = pad_packed_sequence(x)
        m = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1 - dropout)
        mask = Variable(m, requires_grad=False) / (1 - dropout)
        mask = mask.expand_as(x)
        result = mask * x
        if sequence_lengths is not None:
            result = pack_padded_sequence(result, sequence_lengths)
        return result
