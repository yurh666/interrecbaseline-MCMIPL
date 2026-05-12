from __future__ import annotations

from src.embedding.item_encoder import ItemEncoder


class BGEEncoder(ItemEncoder):
    def __init__(self, model_name: str = "BAAI/bge-m3", dim: int = 128, normalize: bool = True) -> None:
        super().__init__(mode="bge", dim=dim, normalize=normalize, model_name=model_name)
