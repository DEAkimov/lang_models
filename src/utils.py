import torch
from torch import optim
from torch.nn import NLLLoss


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def weight_norm(parameters):
    total_norm = 0
    for p in parameters:
        param_norm = p.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm


def init_optimizer(net_parameters, optim_params):
    if optim_params.optimizer == 'SGD':
        optimizer = optim.SGD(
            net_parameters,
            optim_params.learning_rate,
            momentum=optim_params.momentum,
            weight_decay=optim_params.weight_decay
        )
    elif optim_params.optimizer == 'Adam':  # WARNING: Adam with weight decay!
        optimizer = optim.Adam(
            net_parameters,
            optim_params.learning_rate,
            (optim_params.beta0, optim_params.beta1),
            weight_decay=optim_params.weight_decay,
            amsgrad=optim_params.ams
        )
    else:
        raise NotImplementedError
    return optimizer


def init_scheduler(optimizer, scheduler_params):
    if scheduler_params.type == 'inv_sqrt':
        def lr_lambda(step):
            if step == 0 and wm == 0:
                return 1.
            else:
                return 1. / (step ** 0.5) if step > wm \
                    else step / (wm ** 1.5)
        wm = scheduler_params.warmup_step
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    elif scheduler_params.type == 'const':
        scheduler = None
    else:
        raise NameError('wrong scheduler')
    return scheduler


def init_criterion(smoothing, dict_size, ignore_index):
    if smoothing == 0:
        criterion = NLLLoss(ignore_index=ignore_index)
    else:
        criterion = SmoothLoss(dict_size + 1, smoothing, ignore_index)
    return criterion


class SmoothLoss:
    def __init__(self, n_class, smoothing, ignore_index):
        self.n_class = n_class
        self.smoothing = smoothing  # smoothed probability
        self.ignore_index = ignore_index
        # (n_class - 2) because one of the token corresponds
        # to smoothed prob and one corresponds
        # to <pad_idx>, which probability is zero
        self.alpha = (1.0 - smoothing) / (n_class - 2)

    def smooth_label(self, input_t):
        mask = input_t.ne(self.ignore_index).to(torch.float32)
        batch, time = input_t.size()
        smoothed_t = torch.full(
            (batch, time, self.n_class),
            self.alpha,
            dtype=torch.float32,
            device=input_t.device
        )
        smoothed_t.scatter_(-1, input_t.unsqueeze(-1), self.smoothing)
        smoothed_t[:, :, self.ignore_index] = 0.  # p for <pad_idx> is always zero
        smoothed_t = smoothed_t * mask.unsqueeze(-1)
        return smoothed_t

    def __call__(self, log_p, input_t):
        p = self.smooth_label(input_t)
        return -(p * log_p).mean()


if __name__ == '__main__':
    f = 10
    loss = SmoothLoss(f + 1, 0.95, f)

    t = torch.tensor([[1, 2, 3, f]], dtype=torch.long)
    logit = torch.randn([1, 4, f + 1])
    log = torch.nn.functional.log_softmax(logit, dim=-1)
    print(loss(log, t))
