from transformers.cache_utils import Cache


class TTTCache(Cache):
    """Per-request fast weights, pending chunk updates, and local attention KV.

    Writers own fast weights; readers store only their local KV window. State
    belongs to the returned cache so independent requests can be interleaved.
    This cache supports prefill followed by single-token greedy/sampling decode.
    """

    def __init__(self):
        super().__init__()
        self.states = {}
        self._seen_tokens = 0

    def get_seq_length(self, layer_idx=0):
        return self._seen_tokens

    def get_max_length(self):
        return None

    def update(self, *args, **kwargs):
        raise NotImplementedError('TTTCache is updated by the TTT decoder.')

    def reorder_cache(self, beam_idx):
        raise NotImplementedError('TTTCache supports greedy/sampling decode, not beam search.')

    def to_legacy_cache(self):
        raise NotImplementedError('A KV-only legacy cache cannot represent TTT fast weights.')
