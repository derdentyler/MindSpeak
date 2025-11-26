import os
import re
from collections import Counter
from typing import Dict, List, Tuple, TypedDict

from sklearn.model_selection import train_test_split

from transformers import AutoTokenizer

from src.youtube_scraper import YouTubeScraper
from src.subtitle_preprocessor import SubtitlePreprocessor
from src.dataset_saver import DatasetSaver
from src.utils.logger_loader import LoggerLoader
from src.utils.config_model import AppConfig


class DatasetEntry(TypedDict):
    category: str
    text: str


class YouTubeDatasetBuilder:
    """
    Сбор датасета: скачивание, очистка, разбивка на чанки и сохранение в JSON.
    """

    CHUNK_SIZE: int = 500  # число токенов в одном чанке

    def __init__(self, cfg: AppConfig):
        self.cfg: AppConfig = cfg
        self.logger = LoggerLoader().get_logger()

        # директории
        self.subtitle_dir: str = cfg.subtitles_dir
        os.makedirs(self.subtitle_dir, exist_ok=True)
        self.scraper = YouTubeScraper(self.subtitle_dir)

        self.output_dir: str = cfg.output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.train_saver = DatasetSaver(os.path.join(self.output_dir, "train.json"))
        self.val_saver = DatasetSaver(os.path.join(self.output_dir, "val.json"))
        self.val_split_ratio: float = cfg.val_split_ratio
        self.random_state: int = cfg.random_state

        # для токенизации при чанкинге
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

        # статистика
        self.total_videos: int = 0
        self.downloaded_subtitles: int = 0
        self.skipped_videos: List[Dict[str, str]] = []

    def _chunk_text(self, text: str) -> List[str]:
        """
        Разбивает один длинный текст на список чанков по CHUNK_SIZE токенов,
        декодирует их и нормализует двойные дефисы.
        """
        all_ids: List[int] = self.tokenizer.encode(text, add_special_tokens=False)
        chunks: List[str] = []
        for i in range(0, len(all_ids), self.CHUNK_SIZE):
            chunk_ids = all_ids[i : i + self.CHUNK_SIZE]
            # декодируем обратно в текст
            chunk_text = self.tokenizer.decode(chunk_ids, clean_up_tokenization_spaces=True)
            # Нормализуем двойные дефисы: " - - " → "--"
            chunk_text = re.sub(r"\s*-\s*-\s*", "--", chunk_text.strip())
            chunks.append(chunk_text)
        return chunks

    def _split_dataset(
        self,
        dataset: List[DatasetEntry]
    ) -> Tuple[List[DatasetEntry], List[DatasetEntry]]:
        """
        Стратифицированный split полной выборки на train и val.
        """
        if not dataset:
            return [], []

        categories = [item["category"] for item in dataset]
        counts = Counter(categories)
        min_samples = min(counts.values())

        if min_samples < 2:
            self.logger.warning(
                "Некоторые классы имеют < 2 примеров, стратификация может не сработать."
            )
            split_idx = int(len(dataset) * (1 - self.val_split_ratio))
            return dataset[:split_idx], dataset[split_idx:]

        try:
            train_data, val_data = train_test_split(
                dataset,
                test_size=self.val_split_ratio,
                random_state=self.random_state,
                stratify=categories
            )
            self.logger.info(
                f"Split completed: train={len(train_data)}, val={len(val_data)}"
            )
            return train_data, val_data
        except ValueError as exc:
            self.logger.error(f"Ошибка стратификации: {exc}")
            split_idx = int(len(dataset) * (1 - self.val_split_ratio))
            return dataset[:split_idx], dataset[split_idx:]

    def build_dataset(self) -> None:
        dataset: List[DatasetEntry] = []

        for category, urls in self.cfg.categories.items():
            for url in urls:
                self.total_videos += 1
                self.logger.info(f"🔍 Обрабатываю {url} (категория: {category})")
                try:
                    # 1) Скачать субтитры
                    path_vtt = self.scraper.download_subtitles(url)
                    if not path_vtt:
                        self.logger.warning(f"⚠️ Нет сабов: {url}")
                        self.skipped_videos.append({"url": url, "reason": "Нет сабов"})
                        continue
                    self.downloaded_subtitles += 1

                    # 2) Очистка
                    cleaned_txt = path_vtt.replace(".vtt", "_cleaned.txt")
                    SubtitlePreprocessor(path_vtt, cleaned_txt).process()

                    # 3) Чтение текста
                    try:
                        with open(cleaned_txt, "r", encoding="utf-8") as f:
                            text: str = f.read()
                    except Exception as e:
                        self.logger.error(f"Ошибка чтения {cleaned_txt}: {e}")
                        self.skipped_videos.append({"url": url, "reason": "Чтение файла"})
                        continue

                    if not text.strip():
                        self.logger.warning(f"⚠️ Пустой текст: {url}")
                        self.skipped_videos.append({"url": url, "reason": "Пустой текст"})
                        continue

                    # 4) Разбивка на чанки и добавление в датасет
                    for chunk in self._chunk_text(text):
                        dataset.append(DatasetEntry(category=category, text=chunk))

                except Exception as e:
                    self.logger.exception(f"Ошибка обработки {url}: {e}")
                    self.skipped_videos.append({"url": url, "reason": "Общая ошибка"})
                    continue

        # 5) Сохранение
        if dataset:
            try:
                train_data, val_data = self._split_dataset(dataset)
                self.train_saver.save(train_data)
                self.logger.info(f"✅ Train датасет сохранён в {self.output_dir}/train.json")
                self.val_saver.save(val_data)
                self.logger.info(f"✅ Val датасет сохранён в {self.output_dir}/val.json")
            except Exception as e:
                self.logger.error(f"Ошибка сохранения: {e}")
        else:
            self.logger.warning("⚠️ Датасет пуст")

        # финальная статистика
        self.logger.info(f"📊 Загружено {self.downloaded_subtitles}/{self.total_videos} видео")
        if self.skipped_videos:
            self.logger.warning("⚠️ Пропущенные видео:")
            for it in self.skipped_videos:
                self.logger.warning(f"   ❌ {it['url']} — {it['reason']}")

