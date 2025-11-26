import argparse
import subprocess
from src.utils.config_loader import ConfigLoader
from src.utils.logger_loader import LoggerLoader
from src.youtube_dataset_builder import YouTubeDatasetBuilder
from src.fine_tune import fine_tune_model
from src.data_augmentation.augmenter_pipeline import DataAugmentationPipeline

logger = LoggerLoader().get_logger()

def run_eda():
    logger.info("Launching EDA notebook...")
    subprocess.run(["jupyter", "notebook", "notebooks/eda.ipynb"], check=True)

def main():
    parser = argparse.ArgumentParser(description="Run pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["scrape", "eda", "train", "augment"],
        required=True,
        help="Choose task to run"
    )
    # Вручную переопределяем вход/выход для аугментации,
    # если вдруг нужно быстро проверить с другим файлом
    parser.add_argument(
        "--override_input",
        type=str,
        help="(Optional) Переопределить путь к входному JSON для аугментации"
    )
    parser.add_argument(
        "--override_output",
        type=str,
        help="(Optional) Переопределить путь к выходному JSON для аугментации"
    )

    args = parser.parse_args()

    # Загружаем конфиг
    cfg = ConfigLoader(args.config).get_config()
    logger.info(f"Loaded config from {args.config}")
    logger.info(f"Using model: {cfg.model_name}")

    if args.task == "scrape":
        logger.info("Running data scraper...")
        builder = YouTubeDatasetBuilder(cfg)
        builder.build_dataset()

    elif args.task == "eda":
        run_eda()

    elif args.task == "train":
        logger.info("Starting fine-tuning...")
        fine_tune_model(cfg)

    elif args.task == "augment":
        logger.info("Starting data augmentation (back-translation)…")
        aug_cfg = cfg.augmentation

        # Если в CLI передали --override_input/--override_output, то они имеют приоритет
        input_path = args.override_input or aug_cfg.input_path
        output_path = args.override_output or aug_cfg.output_path

        # Параметры для пайплайна аугментации
        min_examples = aug_cfg.min_examples
        bt_rounds = aug_cfg.bt_rounds
        bt_beam_size = aug_cfg.bt_beam_size

        pipeline = DataAugmentationPipeline(
            input_json=input_path,
            output_json=output_path,
            min_examples=min_examples,
            bt_rounds=bt_rounds,
            bt_beam_size=bt_beam_size
        )
        pipeline.run()

    else:
        logger.error("Unknown task")

if __name__ == "__main__":
    main()
