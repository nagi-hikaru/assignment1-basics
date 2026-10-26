import multiprocessing
import pickle
from collections import defaultdict
from pathlib import Path

import regex as re
from tqdm import tqdm

from .pretokenization_example import find_chunk_boundaries


# 数据路径
ROOT_DIR = Path(__file__).parent.parent

DATA_DIR = ROOT_DIR / "data"

TS_TRAIN = DATA_DIR / "TinyStoriesV2-GPT4-train.txt"
TS_Valid = DATA_DIR / "TinyStoriesV2-GPT4-valid.txt"

OWT_TRAIN = DATA_DIR / "owt_train.txt"
OWT_VALID = DATA_DIR / "owt_valid.txt"


class BPE:

    # 预分词正则规则
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    def __init__(
        self, 
        file_path: Path,              # 包含训练数据的文本文件路径
        vocab_size: int,              # 最终词表的正整数上限
        special_tokens: list[str],    # 特殊 token 列表 
        desired_num_chunks: int = 1,  # 期望分块个数
        num_processes: int = 1        # 分块并行进程数，默认为 1
    ) -> None:
        self.file_path = file_path
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens

        # 预分词参数
        self.desired_num_chunks = desired_num_chunks
        self.num_processes = num_processes

        # 初始化词表和合并列表
        self.vocab = [s.encode("utf-8") for s in special_tokens] + [bytes([i]) for i in range(256)]
        self.idx2token = None
        self.merges = []

        # 初始化 all_token_dict 和 pair_dict
        self.all_token_dict = defaultdict(int)
        self.all_pair_dict = defaultdict(int)


    @staticmethod
    def load(path: Path) -> dict:
        """从 pickle 文件加载原始状态字典。

        Returns:
            ``{"vocab": dict[int, bytes], "merges": list[tuple[bytes, bytes]]}``
        """
        with open(path, "rb") as f:
            return pickle.load(f)


    def fit(self) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        """训练 BPE 词表。

        流程：并行预分词 → 累计计数 → 循环合并最高频字节对 → 翻转词表。

        Returns:
            ``(vocab_dict, merges)``：token 索引 → 字节串 的字典，
            以及按合并顺序记录的字节对列表。
        """
        res_dicts = self._pre_tokenize()
        self._accumulate_counts(res_dicts)
        self._train_merges()

        # 翻转词表字典
        self.idx2token = {idx: token for idx, token in enumerate(self.vocab)}
        return self.idx2token, self.merges


    def save(self, path: Path) -> None:
        """将词表和合并列表序列化到 ``path``（pickle 格式）。"""
        with open(path, "wb") as f:
            pickle.dump(
                {"vocab": self.idx2token, "merges": self.merges},
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )


    def _pre_tokenize(self) -> list[tuple[dict[tuple, int], dict[tuple, int]]]:
        """按分块边界并行预分词，返回各分块的 token 频次字典。"""
        boundaries = self._get_boundaries()
        with multiprocessing.Pool(self.num_processes) as pool:
            return pool.starmap(
                self._pre_tokenize_block, 
                zip(boundaries[:-1], boundaries[1:])
            )


    def _get_boundaries(self) -> list[int]:
    
            """计算二进制文件的分块边界。
    
            Returns:
                升序排列的字节偏移列表，形如 ``[0, b1, b2, ..., file_size]``。
                相邻两个元素构成一个半开区间 ``[boundaries[i], boundaries[i+1])``，
                即第 ``i`` 个分块的字节范围；分块数为 ``len(boundaries) - 1``。
            """
             
            with open(self.file_path, "rb") as f:
                boundaries = find_chunk_boundaries(
                    file=f,
                    desired_num_chunks=self.desired_num_chunks,
                    split_special_token=self.special_tokens[0].encode("utf-8")
                )
            # print(f"boundaries(total: {len(boundaries) - 1}): \n{boundaries}")
            return boundaries
    
    
    def _pre_tokenize_block(
        self,
        start: int, 
        end: int
    ) -> tuple[dict[tuple, int], dict[tuple, int]]:
        
        """对文件中指定字节区间做预分词，并统计每种 token 的频次。

        处理流程：
            1. 从 ``self.file_path`` 中读取 ``[start, end)`` 区间的字节；
            2. 以 UTF-8 解码为文本；
            3. 按 ``self.special_tokens`` 切分成若干文本片段；
            4. 对每个片段用正则 ``self.PAT`` 提取 token；
            5. 以 token 的 UTF-8 字节元组为键，累计出现次数。

        Args:
            start: 读取区间的起始字节偏移（包含）。
            end: 读取区间的结束字节偏移（不包含）。

        Returns:
            以 ``tuple[bytes, ...]``（token 的 UTF-8 字节序列）为键、
            出现次数为值的计数字典。例如 ``{ (b'a', b'b'): 3, ... }``。
        """

        token_dict = defaultdict(int)
        pair_dict = defaultdict(int)

        with open(self.file_path, "rb") as f:
            f.seek(start)
            chunk = f.read(end - start)

        text = chunk.decode("utf-8")
        pattern = '|'.join([re.escape(s) for s in self.special_tokens])
        split_text = re.split(pattern, text)

        for sub_text in split_text:
            for token in re.finditer(self.PAT, sub_text):

                # 加速2 -> 切片避免每次重新转换
                raw = token.group().encode("utf-8")
                key = tuple(raw[i:i+1] for i in range(len(raw)))
                # key = tuple(bytes([i]) for i in token.group().encode("utf-8"))
                token_dict[key] += 1

                for i in range(len(key) - 1):
                    pair_dict[(key[i], key[i + 1])] += 1

        return token_dict, pair_dict


    def _accumulate_counts(self, res_dicts: list[tuple[dict[tuple, int], dict[tuple, int]]]) -> None:
        """将各分块的频次合并进 ``all_token_dict``，并统计相邻字节对频次。"""
        for token_dict, pair_dict in res_dicts:
            for token, count in token_dict.items():
                self.all_token_dict[token] += count
            for pair, count in pair_dict.items():
                self.all_pair_dict[pair] += count

    def _train_merges(self) -> None:
        """循环合并最高频字节对，直到达到 ``vocab_size`` 或无 pair 可合并。"""
        pbar = tqdm(
            total=self.vocab_size, 
            initial= 256 + len(self.special_tokens),
            desc="BPE training:", 
            unit=" merge"
        )
        while len(self.all_pair_dict) > 0 and len(self.vocab) < self.vocab_size:
            merge_pair = max(
                self.all_pair_dict.items(), key=lambda x: (x[1], x[0])
            )[0]
            merge_bytes = merge_pair[0] + merge_pair[1]
            self.vocab.append(merge_bytes)
            self.merges.append(merge_pair)
            self._apply_merge(merge_pair, merge_bytes)
            pbar.update(1)
        pbar.close()


    def _apply_merge(self, merge_pair: tuple[bytes, bytes], merge_bytes: bytes) -> None:
        """把所有 token 中的 ``merge_pair`` 替换为 ``merge_bytes``，同步维护计数。"""
        updated_tokens = []
        for token, count in self.all_token_dict.items():

            # 加速1 -> 快速跳过
            if merge_pair[0] not in token or merge_pair[1] not in token:
                continue

            new_token = list(token)
            updated = False
            i = 0
            while i < len(new_token) - 1:
                if (new_token[i], new_token[i + 1]) == merge_pair:
                    updated = True

                    self._dec_pair(merge_pair, count)
                    if i > 0:
                        self.all_pair_dict[(new_token[i - 1], merge_bytes)] += count
                        self._dec_pair((new_token[i - 1], new_token[i]), count)
                    if i < len(new_token) - 2:
                        self.all_pair_dict[(merge_bytes, new_token[i + 2])] += count
                        self._dec_pair((new_token[i + 1], new_token[i + 2]), count)

                    new_token[i] = merge_bytes
                    new_token.pop(i + 1)
                i += 1

            if updated:
                updated_tokens.append((token, tuple(new_token), count))

        for old_token, new_token, count in updated_tokens:
            del self.all_token_dict[old_token]
            self.all_token_dict[new_token] = (
                self.all_token_dict.get(new_token, 0) + count
            )


    def _dec_pair(self, pair: tuple[bytes, bytes], count: int) -> None:
        """减少 pair 计数，归零则从 ``pair_dict`` 中删除。"""
        self.all_pair_dict[pair] -= count
        if self.all_pair_dict[pair] == 0:
            del self.all_pair_dict[pair]


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")

    file_path = TS_Valid
    
    bpe = BPE(
        file_path=file_path, 
        vocab_size=10000, 
        special_tokens=["<|endoftext|>"],
        desired_num_chunks=16,
        num_processes=8
    )
    vocab, merges = bpe.fit()

    print(f"vocab length: {len(vocab)}")
    print(f"merges length: {len(merges)}")

    bpe.save(file_path.with_suffix(".pkl"))

    print("-" * 10)
    print("finish")
   