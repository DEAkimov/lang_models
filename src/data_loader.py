import os
import re
import numpy as np
import torch


class DataLoader:
    def __init__(self,
                 data_path, bpe_dict, batch_size,
                 pad_idx=0, bos_idx=1, eos_idx=2):
        self.data_path = data_path
        self.folders = os.listdir(data_path)
        self.topics = [int(s[:2]) for s in self.folders]
        self.num_topics = [len(os.listdir(data_path + f)) for f in self.folders]
        print([(f, l) for f, l in zip(self.folders, self.num_topics)])
        self.bpe_dict = bpe_dict
        self.batch_size = batch_size

        self.pad_idx, self.bos_idx, self.eos_idx = pad_idx, bos_idx, eos_idx

    def merge(self, lines):
        lengths = [len(line) for line in lines]
        max_len = max(lengths)
        batch_t = torch.full([len(lines), max_len], self.pad_idx)
        for i, line in enumerate(lines):
            batch_t[i, :len(line)] = torch.tensor(line, dtype=torch.long)
        return batch_t

    def read_batch(self):
        topic_indices = np.random.randint(0, len(self.topics), self.batch_size)
        file_indices = [np.random.randint(0, self.num_topics[i]) for i in topic_indices]
        encoded_lines = []
        for i in topic_indices:
            for j in file_indices:
                f_name = self.data_path + self.folders[i] + '/' + str(j) + '.txt'
                with open(f_name, 'r') as f:
                    line = f.readline()
                line = re.sub('\s+', ' ', line).strip()
                bpe = [self.bos_idx] + self.bpe_dict.encode_as_idx(line) + [self.eos_idx]
                encoded_lines.append(bpe)
        return topic_indices, self.merge(encoded_lines)


if __name__ == '__main__':
    dl = DataLoader('../resources/ero/', None, 10)
    batch = dl.read_batch()
    print(batch)

