from typing import Optional

from transformers import AutoTokenizer

_tokenizer = None


def get_tokenizer(name_or_path: Optional[str] = None):
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(
            name_or_path,  # type: ignore
            truncation_side='left',
            padding_side="left",
        )
    return _tokenizer
