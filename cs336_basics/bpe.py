import multiprocessing
import regex as re

from tqdm import tqdm
from pathlib import Path
from collections import defaultdict

from .pretokenization_example import find_chunk_boundaries
from .path import TEST_TEXT, TS_Valid, TS_TRAIN, OWT_VALID


class BPE:

    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    def __init__(
        self, 
        file_path: Path = TEST_TEXT, 
        vocab_size: int = 1000, 
        special_tokens: list[str] = ["<|endoftext|>"], 
        num_processes: int = 1
):
        self.file_path = file_path
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens
        self.num_processes = num_processes

        self.vocab = [s.encode("utf-8") for s in special_tokens] + [bytes([i]) for i in range(256)]
        self.merges = list()

    
    @staticmethod
    def get_boundaries(
        file_path: Path, 
        num_processes: int, 
        special_tokens: str
    ) -> list[int]:
        """
        计算二进制文件的分块边界
        """
        with open(file_path, "rb") as f:
            boundaries = find_chunk_boundaries(
                file=f,
                desired_num_chunks=num_processes,
                split_special_token=special_tokens.encode("utf-8")
            )
        return boundaries


    def fit(self):
        boundaries = self.get_boundaries(self.file_path, self.num_processes, self.special_tokens[0])
                        
        multiprocessing.set_start_method("spawn")
        with multiprocessing.Pool(self.num_processes) as pool:
            token_dicts = pool.starmap(self.pre_tokenize, zip(boundaries[:-1], boundaries[1:]))

        all_token_dict = defaultdict(int)
        pair_dict = defaultdict(int)
        for token_dict in token_dicts:
            for token, count in token_dict.items():
                all_token_dict[token] += count
                for i in range(len(token) - 1):
                    pair_dict[(token[i], token[i + 1])] += count

        while len(pair_dict) > 0 and len(self.vocab) < self.vocab_size:
            merge_pair = max(pair_dict.items(), key=lambda x: (x[1], x[0]))
            if merge_pair[1] == 0:
                break
            # print(merge_pair)

            merge_pair = merge_pair[0]
            merge_bytes = merge_pair[0] + merge_pair[1]
            self.vocab.append(merge_bytes)
            self.merges.append(merge_pair)

            key_changes = list()
            for token, count in all_token_dict.items():
                new_token = list(token)
                updated = False
                i = 0
                while i < len(new_token) - 1:
                    if (new_token[i], new_token[i + 1]) == merge_pair:
                        updated = True
                        pair_dict[merge_pair] -= count
    
                        if i > 0:
                            pair_dict[(new_token[i - 1], new_token[i])] -= count
                            pair_dict[(new_token[i - 1], merge_bytes)] += count
                        if i < len(new_token) - 2:
                            pair_dict[(new_token[i + 1], new_token[i + 2])] -= count
                            pair_dict[(merge_bytes, new_token[i + 2])] += count
    
                        new_token[i] = merge_bytes
                        new_token.pop(i + 1)
                    i += 1

                if updated:
                    key_changes.append((token, tuple(new_token), count))

            for old_key, new_key, count in key_changes:
                del all_token_dict[old_key]
                all_token_dict[new_key] += count

            if pair_dict[merge_pair] <= 0:
                del pair_dict[merge_pair]
        
        vocab_dict = {idx: self.vocab[idx] for idx in range(len(self.vocab))}
        return vocab_dict, self.merges


    def pre_tokenize(
        self,
        start: int, 
        end: int
    ) -> dict[tuple, int]:
        token_dict = defaultdict(int)

        with open(self.file_path, "rb") as f:
            f.seek(start)
            chunk = f.read(end - start)

        text = chunk.decode("utf-8")
        pattern = '|'.join([re.escape(s) for s in self.special_tokens])
        split_text = re.split(pattern, text)

        for sub_text in split_text:
            for token in re.finditer(self.PAT, sub_text):
                key = tuple(bytes([x]) for x in token.group().encode("utf-8"))
                token_dict[key] += 1

        return token_dict


if __name__ == "__main__":
    bpe = BPE(TS_Valid, 15000, num_processes=8)
    vocab, merges = bpe.fit()
    print(len(vocab))
    print(len(merges))
    print("finish")
   