import torch
from collections import OrderedDict


class dataPrefetcher():

    def __init__(self, loader):
        self.loader = loader
        self.stream = torch.cuda.Stream()
        self.preload()
    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True).float()
            self.next_target = self.next_target.cuda(non_blocking=True).float()
    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        inputs = self.next_input
        targets = self.next_target
        self.preload()
        return inputs, targets
    def __iter__(self):
        return self
    

class dataPrefetcherFrozen():

    def __init__(self, loader):
        self.loader = loader
        self.stream = torch.cuda.Stream()
        self.preload()
    def preload(self):
        try:
            (self.next_input, self.next_target), self.next_best_way = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            self.next_best_way = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True).float()
            self.next_target = self.next_target.cuda(non_blocking=True).float()
            self.next_best_way = self.next_best_way.cuda(non_blocking=True).float()
    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        inputs = self.next_input
        targets = self.next_target
        next_best_way = self.next_best_way
        self.preload()
        return inputs, targets, next_best_way
    def __iter__(self):
        return self


def loadWeights(net, pth_file, device):
    state_dict = torch.load(pth_file, map_location=device)
    # create new OrderedDict that does not contain `module.`
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        head = k[:7]
        if head == 'module.':
            name = k[7:]  # remove `module.`
        else:
            name = k
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict)
    return net

