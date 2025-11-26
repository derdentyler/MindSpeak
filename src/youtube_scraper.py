import os
import yt_dlp
from typing import Optional

from src.utils.logger_loader import LoggerLoader

class YouTubeScraper:
    """
    Класс для скачивания субтитров с YouTube.
    """

    def __init__(self, save_path: str = "data/raw"):
        """
        :param save_path: Папка для сохранения субтитров.
        """
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)
        self.logger = LoggerLoader().get_logger()

    def download_subtitles(self, video_url: str, lang: str = "ru") -> Optional[str]:
        """
        Скачивает субтитры и приводит имя файла к стандартному формату.

        :param video_url: Ссылка на видео.
        :param lang: Язык субтитров (по умолчанию "ru").
        :return: Путь к обработанному файлу с субтитрами или None, если субтитры не найдены.
        """
        video_id = video_url.split("v=")[-1]  # Извлекаем ID видео
        expected_file = os.path.join(self.save_path, f"{video_id}.vtt")

        ydl_opts = {
            'writesubtitles': True,
            'subtitleslangs': [lang],
            'skip_download': True,
            'outtmpl': os.path.join(self.save_path, f"{video_id}.%(ext)s"),
            'quiet': True,
            'writeautomaticsub': True
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=False)
            subtitles = info.get("subtitles") or info.get("automatic_captions")

            if subtitles and lang in subtitles:
                ydl.download([video_url])

                # Ищем скачанный файл (так как yt-dlp добавляет .ru.vtt)
                for file in os.listdir(self.save_path):
                    if file.startswith(video_id) and file.endswith(".vtt"):
                        original_path = os.path.join(self.save_path, file)
                        # После скачивания файла
                        if os.path.exists(expected_file):
                            os.remove(expected_file)  # Удаляем старый файл, если он есть
                        os.rename(original_path, expected_file)  # Переименовываем
                        self.logger.info(f"✅ Субтитры сохранены: {expected_file}")
                        return expected_file

                self.logger.warning(
                    "⚠️ Субтитры скачаны, но не удалось найти файл."
                )
                return None
            else:
                self.logger.warning(f"❌ Субтитры на языке '{lang}' не найдены.")
                return None
