from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent

DATA_DIR = ROOT_DIR / "data"

TEST_TEXT = DATA_DIR / "test.txt"

TS_TRAIN = DATA_DIR / "TinyStoriesV2-GPT4-train.txt"
TS_Valid = DATA_DIR / "TinyStoriesV2-GPT4-valid.txt"

OWT_TRAIN = DATA_DIR / "owt_train.txt"
OWT_VALID = DATA_DIR / "owt_valid.txt"