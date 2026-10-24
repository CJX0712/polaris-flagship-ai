"""L3 分页 KV Cache (PagedAttention)

零重写: 复用 Kwon et al. 2023 "vLLM: Efficient Memory Management for
Large Language Model Serving with PagedAttention" 的分块显存管理思想。
将 KV 按固定大小 block 存储, 用 block table 映射逻辑位置→物理 block。

可验证:
- INV-15 分页 KV 与稠密 KV 在数值上完全一致 (无量化/无损失, 仅重排)
- R-L3-01 长序列下内存约为稠密的 50% (按需分配 block, 无预留)
"""
from __future__ import annotations

import numpy as np


class PagedKVCache:
    """按序列分块存储 K/V。K 与 V 各用独立 block 缓冲。"""

    def __init__(self, block_size: int = 16, kv_dim: int = 64, max_blocks: int = 4096):
        self.block_size = block_size
        self.kv_dim = kv_dim
        self.max_blocks = max_blocks
        self.k_blocks = np.zeros((max_blocks, block_size, kv_dim), dtype=np.float64)
        self.v_blocks = np.zeros((max_blocks, block_size, kv_dim), dtype=np.float64)
        self.block_table: dict[int, list] = {}
        self.seq_len: dict[int, int] = {}
        self._free = list(range(max_blocks))
        self._next_seq = 0

    def new_sequence(self) -> int:
        sid = self._next_seq
        self._next_seq += 1
        self.block_table[sid] = []
        self.seq_len[sid] = 0
        return sid

    def _ensure_block(self, sid: int):
        if self.seq_len[sid] % self.block_size == 0:
            if not self._free:
                raise MemoryError("KV cache out of blocks")
            self.block_table[sid].append(self._free.pop())

    def append(self, sid: int, k: np.ndarray, v: np.ndarray):
        """追加单个 token 的 K,V (各 [kv_dim])。"""
        k = np.asarray(k, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64)
        self._ensure_block(sid)
        bi = self.block_table[sid][-1]
        p = self.seq_len[sid] % self.block_size
        self.k_blocks[bi, p] = k
        self.v_blocks[bi, p] = v
        self.seq_len[sid] += 1

    def get_kv(self, sid: int) -> tuple[np.ndarray, np.ndarray]:
        """返回该序列截至当前的稠密 K,V [L, kv_dim] (INV-15 等价性来源)。"""
        L = self.seq_len[sid]
        K = np.zeros((L, self.kv_dim), dtype=np.float64)
        V = np.zeros((L, self.kv_dim), dtype=np.float64)
        for l in range(L):
            bi = self.block_table[sid][l // self.block_size]
            K[l] = self.k_blocks[bi, l % self.block_size]
            V[l] = self.v_blocks[bi, l % self.block_size]
        return K, V

    @property
    def memory_blocks_used(self) -> int:
        return self.max_blocks - len(self._free)
