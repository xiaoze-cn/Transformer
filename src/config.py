from dataclasses import dataclass, field


@dataclass
class DataConfig:
    dataset_name: str = "IWSLT/iwslt2017"
    dataset_config: str = "iwslt2017-zh-en"
    source_lang: str = "zh"
    target_lang: str = "en"
    raw_dir: str = "data/raw"
    data_dir: str = "data"
    tokenizer_prefix: str = "data/spm"
    vocab_size: int = 16000
    max_length: int = 96


@dataclass
class TrainConfig:
    epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 0.0005
    label_smoothing: float = 0.1
    grad_clip: float = 1.0
    device: str = "auto"


@dataclass
class ModelConfig:
    d_model: int = 256
    nhead: int = 4
    num_layers: int = 3
    dim_feedforward: int = 1024
    dropout: float = 0.1


@dataclass
class DecodeConfig:
    max_length: int = 96
    beam_size: int = 4


@dataclass
class Config:
    seed: int = 42
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    decode: DecodeConfig = field(default_factory=DecodeConfig)


cfg = Config()
