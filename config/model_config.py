from dataclasses import dataclass

@dataclass
class ModelConfig:
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    hidden_act: str = 'gelu'
    layer_norm_eps: int = 1e-12
    image_size: int = 224
    patch_size: int = 16
    n_channels: int = 3
    qkv_bias = True

    hidden_dim: int = 1536

    att_dropout: int = 0.2 #TODO: CHANGE THIS TO MATCH THE ORIGINAL
    batch_size: int = 16

    out_dim: int = 37

    