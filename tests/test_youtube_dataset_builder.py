import os
import json
import shutil
import pytest

from src.utils.config_loader import ConfigLoader
from src.youtube_dataset_builder import YouTubeDatasetBuilder
from src.youtube_scraper import YouTubeScraper
from src.subtitle_preprocessor import SubtitlePreprocessor

TEST_CONFIG_PATH = "tests/configs/test_config.yaml"
TEST_OUTPUT_DIR  = "tests/data/processed/"


@pytest.fixture(autouse=True)
def clean_output():
    # Перед каждым запуском чистим папку
    if os.path.exists(TEST_OUTPUT_DIR):
        shutil.rmtree(TEST_OUTPUT_DIR)
    os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)
    yield
    # После — тоже чистим
    shutil.rmtree(TEST_OUTPUT_DIR)

@pytest.fixture
def cfg():
    cfg = ConfigLoader(config_path=TEST_CONFIG_PATH).get_config()
    # Подставляем тестовые пути
    cfg.output_dir    = TEST_OUTPUT_DIR
    cfg.subtitles_dir = TEST_OUTPUT_DIR  # туда же
    return cfg

@pytest.fixture
def fake_files(tmp_path, monkeypatch, cfg):
    """
    Готовим:
    - фиктивный .vtt файл для каждого URL
    - подменяем download_subtitles, чтобы возвращал этот .vtt
    - подменяем SubtitlePreprocessor.process(), чтобы создавал cleaned.txt
    """
    # По категориям и URL из cfg
    fake_vtts = {}
    for cat, urls in cfg.categories.items():
        for url in urls:
            vid = url.split("v=")[-1]
            vtt = tmp_path / f"{vid}.vtt"
            vtt.write_text("WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nTest")
            fake_vtts[url] = str(vtt)

    # Мокаем download_subtitles
    def fake_download(self, url):
        return fake_vtts[url]
    monkeypatch.setattr(YouTubeScraper, "download_subtitles", fake_download)

    # Мокаем процессинг: из .vtt → .txt с тем же basename+"_cleaned.txt"
    def fake_process(self):
        # self.input_path и self.output_path из SubtitlePreprocessor
        with open(self.input_path, "r", encoding="utf-8") as inp:
            txt = inp.read()
        with open(self.output_path, "w", encoding="utf-8") as outp:
            outp.write(txt + "\n--cleaned--")
    monkeypatch.setattr(SubtitlePreprocessor, "process", fake_process)

    return fake_vtts

@pytest.mark.integration
def test_youtube_dataset_builder(fake_files, cfg):
    """
    Теперь build_dataset соберёт фиктивные субтитры,
    «очистит» их, и сохранит dataset.json.
    """
    builder = YouTubeDatasetBuilder(cfg)
    builder.output_dir = TEST_OUTPUT_DIR
    builder.build_dataset()

    train_path = os.path.join(TEST_OUTPUT_DIR, "train.json")
    val_path = os.path.join(TEST_OUTPUT_DIR, "val.json")
    assert os.path.isfile(train_path), "❌ Train датасет не был сохранён!"
    assert os.path.isfile(val_path), "❌ Val датасет не был сохранён!"

    with open(train_path, "r", encoding="utf-8") as f:
        train_data = json.load(f)
    with open(val_path, "r", encoding="utf-8") as f:
        val_data = json.load(f)

    total_urls = sum(len(urls) for urls in cfg.categories.values())
    assert len(train_data) + len(val_data) == total_urls

    for rec in train_data + val_data:
        assert "category" in rec and "text" in rec
        assert rec["text"].endswith("--cleaned--")
