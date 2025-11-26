import json
import os
from typing import Dict, Mapping, Sequence

import torch  # type: ignore[import]
from torch.utils.data import DataLoader, Dataset  # type: ignore[import]
from transformers import AutoTokenizer, DataCollatorWithPadding  # type: ignore[import]


os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


BatchItem = Dict[str, torch.Tensor]


class TextDataset(Dataset):
    def __init__(self, json_path: str, config: dict, model_name: str, max_length: int = 512):
        """
        Загружает данные из JSON, токенизирует текст и преобразует категории в числовые индексы.

        :param json_path: путь к JSON-файлу с данными.
        :param config: конфиг с категориями.
        :param model_name: имя предобученной модели (используется для загрузки соответствующего токенизатора).
        :param max_length: максимальная длина токенизированного текста (по умолчанию 512).
        """
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        # Загружаем список категорий из конфига и создаем маппинг категория -> индекс
        self.config: Mapping[str, Sequence[str]] = config
        self.label_to_idx: Dict[str, int] = {
            cat: idx for idx, cat in enumerate(self.config["categories"])
        }
        self.idx_to_label = {idx: cat for cat, idx in self.label_to_idx.items()}

        # Загружаем токенизатор от модели
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length

        # Проверяем, что JSON содержит нужные ключи
        for item in self.data:
            if "text" not in item or "category" not in item:
                raise ValueError("JSON-файл должен содержать ключи 'text' и 'category'.")

        # Извлекаем тексты и метки, преобразуем категории в индексы
        self.texts = [item["text"] for item in self.data]
        self.labels = [self.label_to_idx[item["category"]] for item in self.data]

    def __len__(self):
        """ Возвращает количество примеров в датасете. """
        return len(self.texts)

    def __getitem__(self, idx: int) -> BatchItem:
        """
        Возвращает токенизированный текст и числовой индекс категории.

        :param idx: индекс примера.
        :return: словарь с токенизированным текстом (input_ids, attention_mask и т. д.) и метка категории (labels).
        """
        text = self.texts[idx]
        label = self.labels[idx]

        # Токенизируем текст с padding и truncation
        encoding = self.tokenizer(
            text,
            padding=False,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        # Убираем лишнюю размерность (по умолчанию tokenizer возвращает тензор с размерностью [1, seq_length])
        encoding = {key: val.squeeze(0) for key, val in encoding.items()}

        # Вставляем метку в словарь
        encoding['labels'] = torch.tensor(label, dtype=torch.long)

        return encoding

    def get_label_mapping(self):
        """ Возвращает словарь {категория: индекс}. """
        return self.label_to_idx

def get_data_collator(tokenizer: AutoTokenizer) -> DataCollatorWithPadding:
    """
    Создаёт collator с динамическим padding'ом.
    """
    return DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding=True
    )


def get_dataloader(
    json_path: str,
    config: Mapping[str, Sequence[str]],
    model_name: str,
    batch_size: int = 8,
    shuffle: bool = True,
) -> DataLoader:
    """
    Создает DataLoader для работы с батчами.

    :param json_path: путь к JSON-файлу с данными.
    :param config: конфиг.
    :param model_name: название предобученной модели для токенизации.
    :param batch_size: размер батча (по умолчанию 8).
    :param shuffle: перемешивать данные или нет (по умолчанию True).
    :return: DataLoader
    """
    dataset = TextDataset(json_path, config, model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    collator = get_data_collator(tokenizer)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collator,
    )
