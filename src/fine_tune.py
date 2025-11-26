import os
import torch  # type: ignore[import]
from transformers import (  # type: ignore[import]
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)  # type: ignore[import]
from peft import LoraConfig, get_peft_model, TaskType  # type: ignore[import]
from src.dataset import TextDataset
from src.utils.logger_loader import LoggerLoader
from src.utils.config_model import AppConfig  # импорт Pydantic-модели

logger = LoggerLoader().get_logger()

def fine_tune_model(cfg: AppConfig) -> None:
    """
    Запускает fine-tuning классификатора на основе LLM с опциональным LoRA.

    :param cfg: проверенный AppConfig, в котором заданы:
        * model_name, пути train/val, директория сохранения
        * параметры LoRA (use_lora, lora_r, lora_alpha, lora_dropout)
        * гиперпараметры обучения (batch_size, num_epochs, learning_rate, weight_decay, logging_steps, save_total_limit)
    :return: None
    Побочные эффекты: создаёт `Trainer`, обучает модель, сохраняет результаты в `cfg.save_dir`,
    при включённой LoRA также сохраняет адаптеры отдельно.
    """
    logger.info("Starting fine-tuning process...")

    # ========== Настройки из конфига ==========
    # Конфигурация LoRA и пути сохранения берутся из cfg
    use_lora     = cfg.use_lora
    lora_r       = cfg.lora_r
    lora_alpha   = cfg.lora_alpha
    lora_dropout = cfg.lora_dropout
    save_dir     = cfg.save_dir
    os.makedirs(save_dir, exist_ok=True)

    # ========== Устройство ==========
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # ========== Датасеты ==========
    # Датасеты строим на основе путей из cfg и токенизатора той же модели
    train_ds = TextDataset(cfg.train_data_path, cfg.model_dump(), cfg.model_name)
    val_ds   = TextDataset(cfg.val_data_path,   cfg.model_dump(), cfg.model_name)
    num_labels = len(train_ds.get_label_mapping())

    # ========== Токенизатор и модель ==========
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model     = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        num_labels=num_labels
    )
    logger.info(f"Loaded base model: {cfg.model_name}")

    # ========== Применение LoRA (PEFT) ==========
    if use_lora:
        logger.info("Applying LoRA PEFT...")
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            # LoRA таргетирует проекции q/v, наиболее влияющие на attention
            target_modules=["q_proj", "v_proj"],
            bias="none"
        )
        model = get_peft_model(model, peft_config)

    model.to(device)

    # ========== Аргументы тренировки ==========
    # Все гиперпараметры обучения подаются из конфигурации
    training_args = TrainingArguments(
        output_dir=save_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        num_train_epochs=cfg.num_epochs,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        logging_dir="logs",
        logging_steps=cfg.logging_steps,
        save_total_limit=cfg.save_total_limit,
        push_to_hub=False,
    )

    # ========== Trainer ==========
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=TextDataset.collate_fn,
        tokenizer=tokenizer,
    )

    # ========== Запуск обучения ==========
    logger.info("Training started...")
    trainer.train()

    # ========== Сохранение ==========
    final_dir = os.path.join(save_dir, "final_model")
    model.save_pretrained(final_dir)
    logger.info(f"Model saved to {final_dir}")

    if use_lora:
        logger.info("LoRA adapters saved separately.")
