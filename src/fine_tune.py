import os
import random

import numpy as np  # type: ignore[import]
import torch  # type: ignore[import]
from sklearn.metrics import (  # type: ignore[import]
    accuracy_score,
    f1_score,
    precision_recall_fscore_support
)
from sklearn.utils.class_weight import compute_class_weight  # type: ignore[import]
from transformers import (  # type: ignore[import]
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)
from peft import LoraConfig, get_peft_model, TaskType  # type: ignore[import]
from src.dataset import TextDataset, get_data_collator
from src.utils.logger_loader import LoggerLoader
from src.utils.config_model import AppConfig  # импорт Pydantic-модели

logger = LoggerLoader().get_logger()

def set_all_seeds(seed: int = 42) -> None:
    """
    Устанавливает seed для всех генераторов случайных чисел.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    set_seed(seed)
    logger.info(f"Random seed установлен: {seed}")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_score(labels, predictions)
    f1_weighted = f1_score(labels, predictions, average="weighted")
    f1_macro = f1_score(labels, predictions, average="macro")
    precision, recall, _, _ = precision_recall_fscore_support(
        labels, predictions, average="weighted"
    )
    return {
        "accuracy": accuracy,
        "f1_weighted": f1_weighted,
        "f1_macro": f1_macro,
        "precision": precision,
        "recall": recall,
    }


def compute_class_weights(train_dataset: TextDataset) -> torch.Tensor:
    labels = train_dataset.labels
    num_classes = len(train_dataset.get_label_mapping())
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(num_classes),
        y=labels,
    )
    weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
    logger.info(f"Class weights computed: {weights_tensor}")
    return weights_tensor


class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = torch.nn.CrossEntropyLoss()
        if self.class_weights is not None:
            loss_fct = torch.nn.CrossEntropyLoss(
                weight=self.class_weights.to(model.device)
            )
        loss = loss_fct(
            logits.view(-1, self.model.config.num_labels),
            labels.view(-1),
        )
        return (loss, outputs) if return_outputs else loss


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
    set_all_seeds(cfg.random_state)

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
    class_weights = (
        compute_class_weights(train_ds) if cfg.use_class_weights else None
    )

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
        load_best_model_at_end=True,
        metric_for_best_model="f1_weighted",
        greater_is_better=True,
    )

    data_collator = get_data_collator(tokenizer)

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
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
