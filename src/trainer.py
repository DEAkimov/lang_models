from time import time
from tqdm import tqdm

import torch
from torch.nn import LogSoftmax
from torch.nn.utils import clip_grad_norm_

from src.utils import count_parameters, weight_norm, init_optimizer, init_scheduler, init_criterion


class Trainer:
    def __init__(self,
                 model, device,
                 optimizer_params, scheduler_params,
                 grad_norm, smoothing,
                 segment_len, ignore_index, dict_size,
                 data_loader, bpe, writer):
        self.model = model
        self.device = device
        self.optimizer = init_optimizer(self.model.parameters(), optimizer_params)
        self.scheduler = init_scheduler(self.optimizer, scheduler_params)
        self.segment_len = segment_len
        self.dict_size = dict_size
        self.log_softmax = LogSoftmax(dim=-1)
        self.grad_norm = grad_norm
        self.criterion = init_criterion(smoothing, dict_size, ignore_index)

        self.data_loader = data_loader
        self.bpe = bpe
        self.writer = writer

    def init_print(self):
        # TODO: fill up this function
        num_parameters = count_parameters(self.model.parameters())
        print('number of trainable parameters: {}'.format(num_parameters))

    def save(self):
        pass

    def load(self):
        pass

    def optimizer_step(self, loss):
        if self.scheduler is not None:
            self.scheduler.step()
        loss.backward()
        grad_norm = clip_grad_norm_(
            self.model.parameters(),
            self.grad_norm
        )
        self.optimizer.step()
        return grad_norm

    def loss_on_batch(self, batch):
        data, topic = batch
        memory = None
        data_len = data.size(1)
        segment_len = self.segment_len
        num_steps = segment_len // data_len
        mean_step_loss, mean_grad_norm = 0.0, 0.0
        for step in range(num_steps):
            model_input = data[:, step * segment_len:(step + 1) * segment_len]
            logits, memory = self.model(model_input, topic, memory)
            log_sm = self.log_softmax(logits[:, :-1])
            loss = self.criterion(
                log_sm.contiguous().view(-1, self.dict_size),
                model_input[:, 1:].contiguous().view(-1)
            )
            mean_step_loss += loss.item()
            mean_grad_norm += self.optimizer_step(loss)
        return mean_step_loss / num_steps, mean_grad_norm / num_steps

    def train_step(self, batch):
        start = time()
        loss, grad_norm = self.loss_on_batch(batch)
        elapsed_time = time() - start
        wn = weight_norm(self.model.parameters())
        lr = self.optimizer.param_groups[0]['lr']
        self.writer.update_metrics(wn, grad_norm, elapsed_time, lr, loss)

    def train(self, n_steps, steps_per_iter):
        self.model.train()
        for step in range(n_steps):
            batch = self.data_loader.read_batch()
            self.train_step(batch)
            if step % steps_per_iter == 0:
                bpe_code = self.model.generate(250)
                generated_string = self.bpe.decode_ids(bpe_code)
                self.writer.write_metrics(generated_string)
