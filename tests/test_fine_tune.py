import os
import shutil
import tempfile
import pytest
import torch
from types import SimpleNamespace

# Импортируем модуль, в котором лежит fine_tune_model
import src.fine_tune as ft
from src.utils.config_model import AppConfig

# --- 1. ПОДГОТОВКА DUMMY-КОМПОНЕНТОВ ---

class DummyDataset:
    """
    Заглушка вместо TextDataset:
    возвращает два примера с маленькими тензорами.
    """
    def __init__(self, path, cfg_dict, model_name):
        # ничего не читаем из path
        pass

    def get_label_mapping(self):
        return {"any": 0}

    def __len__(self):
        return 2

    def __getitem__(self, idx):
        # Вернём самый простой «батч»-словарь
        return {
            "input_ids": torch.tensor([1,2,3], dtype=torch.long),
            "attention_mask": torch.tensor([1,1,1], dtype=torch.long),
            "labels": torch.tensor(0, dtype=torch.long)
        }

    @staticmethod
    def collate_fn(batch):
        # просто возвращаем batch, Trainer.train() у DummyTrainer его не использует
        return batch

class DummyTrainer:
    """
    Заглушка вместо transformers.Trainer:
    просто помечает, что train() был вызван.
    """
    def __init__(self, model, args, train_dataset, eval_dataset, data_collator, tokenizer=None):
        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.data_collator = data_collator
        self.tokenizer = tokenizer
        self.trained = False

    def train(self):
        # Симулируем один шаг обучения
        self.trained = True
        return SimpleNamespace()

@pytest.fixture(autouse=True)
def patch_everything(monkeypatch, tmp_path):
    """
    Подменяем внутри src.fine_tune:
      - TextDataset → DummyDataset
      - Trainer      → DummyTrainer
      - AutoTokenizer.from_pretrained → dummy возвращающий SimpleNamespace
      - AutoModelForSequenceClassification.from_pretrained → DummyModel
    """
    # 1. Подмена датасета
    monkeypatch.setattr(ft, "TextDataset", DummyDataset)

    # 2. Подмена Trainer
    monkeypatch.setattr(ft, "Trainer", DummyTrainer)

    # 3. Подмена токенизатора — нам он не нужен, просто placeholder
    class DummyTokenizer:
        pass
    monkeypatch.setattr(ft.AutoTokenizer, "from_pretrained",
                        lambda name: DummyTokenizer())

    # 4. Подмена модели
    class DummyModel:
        def to(self, device): pass
        def save_pretrained(self, path):
            os.makedirs(path, exist_ok=True)
            # Создадим фейковый файл, чтобы можно было проверить
            with open(os.path.join(path, "pytorch_model.bin"), "w") as f:
                f.write("dummy")

    monkeypatch.setattr(ft.AutoModelForSequenceClassification, "from_pretrained",
                        lambda name, num_labels: DummyModel())

    yield

@pytest.fixture
def cfg(tmp_path):
    """
    Минимальный AppConfig для теста.
    Все поля обязательны, поэтому даём минимальные валидные значения.
    """
    out_dir = tmp_path / "out"
    cfg_dict = {
        "model_name": "dummy-model",
        "output_dir": str(out_dir),
        "subtitles_dir": str(tmp_path / "subs"),
        "train_data_path": "ignored.json",
        "val_data_path": "ignored.json",
        "save_dir": str(out_dir),
        "use_lora": False,
        "lora_r": 1,
        "lora_alpha": 1,
        "lora_dropout": 0.0,
        "batch_size": 2,
        "num_epochs": 1,
        "learning_rate": 1e-5,
        "weight_decay": 0.0,
        "logging_steps": 1,
        "save_total_limit": 1,
        "log_level": "INFO",
        "log_format": "%(asctime)s %(levelname)s %(message)s",
        "categories": {"any": ["url1", "url2"]},
    }
    return AppConfig(**cfg_dict)

@pytest.mark.unit
def test_fine_tune_smoke(cfg):
    """
    Smoke-test: проверяем, что fine_tune_model:
      1) не падает,
      2) создаёт папку final_model с dummy-файлом.
    """
    # Запуск точечного fine-tuning
    ft.fine_tune_model(cfg)

    # Проверяем, что модель сохранилась
    final_dir = os.path.join(cfg.save_dir, "final_model")
    assert os.path.isdir(final_dir), "Папка final_model должна быть создана"
    file_path = os.path.join(final_dir, "pytorch_model.bin")
    assert os.path.isfile(file_path), "Файл pytorch_model.bin должен быть сохранён"
