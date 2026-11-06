import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from deepvoice.xlsr import LAYERS, embed_batch


def _tiny_model():
    # same architecture family as XLS-R (stable layer norm, attention mask), random tiny weights
    cfg = transformers.Wav2Vec2Config(
        hidden_size=16, num_hidden_layers=24, num_attention_heads=2, intermediate_size=32,
        conv_dim=(8,) * 7, feat_extract_norm="layer", do_stable_layer_norm=True,
        num_conv_pos_embeddings=16, num_conv_pos_embedding_groups=2,
    )
    torch.manual_seed(0)
    return transformers.Wav2Vec2Model(cfg).eval()


def test_padding_does_not_change_embedding():
    model = _tiny_model()
    rng = np.random.default_rng(0)
    short, long = rng.normal(0, 0.1, 8000).astype(np.float32), rng.normal(0, 0.1, 16000).astype(np.float32)
    alone = embed_batch(model, [short])
    batched = embed_batch(model, [short, long])
    assert alone.shape == (1, len(LAYERS), 16)
    np.testing.assert_allclose(alone[0].astype(float), batched[0].astype(float), atol=2e-2)
