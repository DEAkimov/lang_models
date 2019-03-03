from tensorboardX import SummaryWriter


class Writer:
    def __init__(self, logdir, batches_per_iter):
        self.writer = SummaryWriter(logdir)
        self.batches_per_iter = batches_per_iter
        self.iterations_done = 0

        self.weight_norm, self.grad_norm = 0.0, 0.0
        self.sec_per_batch, self.lr, self.loss = 0.0, 0.0, 0.0

    def update_metrics(
            self,
            weight_norm, grad_norm,
            sec_per_batch, lr, loss
    ):
        self.weight_norm += weight_norm
        self.grad_norm += grad_norm
        self.sec_per_batch += sec_per_batch
        self.lr += lr
        self.loss += loss

    def _write(self, tag, scalar):
        self.writer.add_scalar(
            tag,
            scalar / self.batches_per_iter,
            self.iterations_done
        )

    def _write_logs(self):
        self._write('train/weight_norm', self.weight_norm)
        self._write('train/grad_norm', self.grad_norm)
        self._write('train/sec_per_batch', self.sec_per_batch)
        self._write('train/lr', self.lr)
        self._write('train/loss', self.loss)

    def write_metrics(self, string):
        self._write_logs()
        self.writer.add_text('generated_by_lm', string, self.iterations_done)
        self.iterations_done += 1
        self._zero_statistics()

    def _zero_statistics(self):
        self.weight_norm, self.grad_norm = 0.0, 0.0
        self.sec_per_batch, self.lr, self.loss = 0.0, 0.0, 0.0
