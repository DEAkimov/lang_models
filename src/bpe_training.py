import os
import numpy as np
from tqdm import tqdm
import sentencepiece as spm


if __name__ == '__main__':
    # ============= concatenate all jokes into single file =============
    # path = '../resources/jokes/'
    # files = os.listdir(path)
    # full_lines = []
    # for file in files:
    #     with open(path + file) as f:
    #         full_lines += f.readlines()
    # np.random.shuffle(full_lines)
    # with open('../resources/all_jokes.txt', 'w') as f:
    #     f.writelines(full_lines)

    # ============= train bpe vocab for jokes =============
    # spm.SentencePieceTrainer.Train(
    #     '--input=../resources/all_jokes.txt --model_prefix=bpe_jokes --vocab_size=1000 --character_coverage=1.0'
    # )

    # ============= encode training data =============
    # bpe = spm.SentencePieceProcessor()
    # bpe.load('bpe.model')
    # # 0 = unk, 0 = pad, 1 = bos, 2 = eos
    # with open('../resources/ero/03_group/0.txt', 'r') as f:
    #     lines = f.readlines()
    # code = bpe.encode_as_ids(lines[0])
    # print(len(lines), len(lines[0]), len(code))

    # max_len = 0
    #
    # encoded_lines = []
    # for line in tqdm(lines):
    #     num_chars = len(line)
    #     num_words = len(line.split())
    #     encoded = bpe.encode_as_ids(line)
    #     if len(encoded) > max_len:
    #         max_len = len(encoded)
    #
    #     encoded_lines.append(
    #         str(encoded[:198])[1:-1].replace(',', '') + '\t' + str(num_chars) + '\t' + str(num_words) + '\n'
    #     )
    # print(max_len)
    # with open('../resources/dialogs/valid_bpe.txt', 'w') as f:
    #     f.writelines(encoded_lines)

    # ============= concatenate all ero into single file =============
    bpe = spm.SentencePieceProcessor()
    bpe.load('bpe.model')

    path = '../resources/ero/'
    folders = os.listdir(path)
    lines = []
    for folder in folders:
        files = os.listdir(path + folder)
        max_len, min_len, mean_len = 0, 10_000_000, 0
        for file in files:
            with open(path + folder + '/' + file, 'r') as f:
                line = f.readline()
            num_chars = len(line)
            num_words = len(line.split())
            code = bpe.encode_as_ids(line)
            if len(code) > max_len:
                max_len = len(code)
            if len(code) < min_len:
                min_len = len(code)
            mean_len += len(code)

            if len(code) < 512:
                continue

            target_line = str(code).replace(',', '')[1:-1] + '\t' + str(num_chars) + '\t' + str(num_words) + '\n'
            lines.append(target_line)
        mean_len /= len(files)
        print(len(lines), min_len, max_len, mean_len)
    with open('train_bpe_ero.txt', 'w') as f:
        f.writelines(lines)

    # full_lines = []
    # for file in files:
    #     with open(path + file) as f:
    #         full_lines += f.readlines()
    # np.random.shuffle(full_lines)
    # with open('../resources/all_jokes.txt', 'w') as f:
    #     f.writelines(full_lines)

    # ============= train bpe vocab for ero =============
    # spm.SentencePieceTrainer.Train(
    #     '--input=../resources/to_bpe.txt --model_prefix=bpe --vocab_size=10000 --character_coverage=1.0'
    # )
