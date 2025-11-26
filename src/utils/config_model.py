from pydantic import BaseModel, Field
from typing import Dict, List, Optional


class AugmentationConfig(BaseModel):
    """
    Конфигурация для модуля аугментации (back-translation).
    Поля должны совпадать с секцией 'augmentation' в config.yaml.
    """
    input_path: str
    output_path: str
    min_examples: int
    bt_rounds: int
    bt_beam_size: int

    class Config:
        extra = "ignore"  # игнорировать лишние ключи


class AppConfig(BaseModel):
    """
    Основная модель конфигурации приложения.
    """
    model_name: str
    output_dir: str
    subtitles_dir: str
    train_data_path: str
    val_data_path: str
    save_dir: str
    use_lora: bool
    lora_r: int = Field(..., ge=1)
    lora_alpha: int
    lora_dropout: float
    batch_size: int
    num_epochs: int
    learning_rate: float
    weight_decay: float
    logging_steps: int
    save_total_limit: int

    log_level: str
    log_format: str

    categories: Dict[str, List[str]]

    random_state: int = 42
    val_split_ratio: float = 0.2
    use_class_weights: bool = True

    # позволяем тестам не иметь этой секции
    augmentation: Optional[AugmentationConfig] = None

    class Config:
        extra = "ignore"
