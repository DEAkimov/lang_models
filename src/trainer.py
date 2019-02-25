from torch.nn import LogSoftmax, NLLLoss


class Trainer:
    def __init__(self,
                 model, optimizer,
                 segment_len, ignore_index, dict_size,
                 data_loader):
        self.model = model
        self.optimizer = optimizer
        self.segment_len = segment_len
        self.dict_size = dict_size
        self.log_softmax = LogSoftmax(dim=-1)
        self.criterion = NLLLoss(ignore_index=ignore_index)

        self.data_loader = data_loader

    def init_print(self):
        pass

    def save(self):
        pass

    def load(self):
        pass

    def optimize_step(self, loss):
        pass

    def train_step(self, data, topic):
        memory = None
        data_len = data.size(1)
        segment_len = self.segment_len
        num_steps = segment_len // data_len
        mean_step_loss = 0.0
        for step in range(num_steps):
            model_input = data[:, step * segment_len:(step + 1) * segment_len]
            logits, memory = self.model(model_input, topic, memory)
            log_sm = self.log_softmax(logits[:, :-1])
            loss = self.criterion(
                log_sm.contiguous().view(-1, self.dict_size),
                model_input[:, 1:].contiguous().view(-1)
            )
            self.optimize_step(loss)
            mean_step_loss += loss.item()
        return mean_step_loss / num_steps
