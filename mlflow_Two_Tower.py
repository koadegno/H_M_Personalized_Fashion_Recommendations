import pandas as pd
import numpy as np
import json
import yaml
from tqdm import tqdm
from pathlib import Path
import plotly.express as px
from datetime import datetime
import plotly.graph_objects as go

import random
import mlflow
import mlflow.pytorch
from mlflow.types.schema import Schema, ColSpec, TensorSpec
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from typing import Callable, Dict, Any, List, Tuple
from sklearn.model_selection import ParameterGrid, train_test_split

from preprocessing import preprocess_articles, preprocess_customers, preprocess_transactions

##################
#### DATASET #####
##################

class CustomDataset(Dataset):
    def __init__(self, df):
        self.data = {col: torch.tensor(df[col].values) if pd.api.types.is_numeric_dtype(df[col]) else df[col].values for col in df.columns}
        self.df = df.copy()

    def __len__(self):
        return len(next(iter(self.data.values())))

    def __getitem__(self, idx):
        return {key: value[idx] for key, value in self.data.items()}


def custom_collate_fn(batch):
    """
    Custom collate function to handle mixed-type datasets

    Args:
        batch (list): List of dictionaries from the dataset

    Returns:
        dict: A dictionary with tensors and arrays properly handled
    """
    # Initialize dictionaries to collect data
    collated_batch = {}

    # Iterate through keys of the first item to get all column names
    for key in batch[0].keys():
        # Collect values for this key
        values = [item[key] for item in batch]

        # Check the type of the first value to determine how to process
        if isinstance(values[0], torch.Tensor):
            # If it's already a tensor, stack them
            collated_batch[key] = torch.stack(values)
        elif isinstance(values[0], np.ndarray):
            # If it's a numpy array, convert to tensor or keep as array
            if values[0].dtype in [np.float32, np.float64, np.int32, np.int64]:
                collated_batch[key] = torch.tensor(values)
            else:
                collated_batch[key] = np.array(values)
        else:
            # For other types (like strings), keep as list or array
            collated_batch[key] = np.array(values)

    return collated_batch


##################
#### MODELS #####
##################

class StringLookup:

    def __init__(self, vocabulary: List[str], mask_token=None):
        self.vocab = {word: idx for idx, word in enumerate(vocabulary)}
        self.vocab["<UNK>"] = len(self.vocab)
        self.mask_token = mask_token

    def __call__(self, inputs: np.ndarray) -> torch.Tensor:
        # Convert string inputs to indices
        indices = [self.vocab.get(x, self.vocab["<UNK>"]) for x in inputs]
        return torch.tensor(indices, dtype=torch.long)

    def __len__(self):
        return len(self.vocab)


class StringEmbedding(nn.Module):
    def __init__(self, user_id_list: List[str], embedding_dimension: int):
        super().__init__()

        # Create the vocabulary space
        self.string_lookup = StringLookup(user_id_list)

        # The real embeddings
        self.embedding = nn.Embedding(num_embeddings=len(self.string_lookup), embedding_dim=embedding_dimension)

    def forward(self, user_ids: torch.Tensor) -> torch.Tensor:
        # Convert user IDs to indices
        return self.embedding(self.string_lookup(user_ids))


class QueryTower(nn.Module):

    def __init__(self, user_id_list: List[str], embedding_dimension: int):
        super().__init__()

        self.user_embedding = StringEmbedding(user_id_list, embedding_dimension)
        self.age_normalization = nn.BatchNorm1d(1)

        self.dense_nn = nn.Sequential(nn.Linear(in_features=embedding_dimension + 3, out_features=embedding_dimension), nn.Dropout(0.2), nn.ReLU(), nn.Linear(embedding_dimension, embedding_dimension))

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        user_embedding = self.user_embedding(inputs["customer_id"])
        normalized_age = self.age_normalization(inputs["age"].float().unsqueeze(1))
        month_sin = inputs["month_sin"].float().unsqueeze(1)
        month_cos = inputs["month_cos"].float().unsqueeze(1)

        concatenated_inputs = torch.cat([user_embedding, normalized_age, month_sin, month_cos], dim=1)

        outputs = self.dense_nn(concatenated_inputs)
        return outputs


class ItemTower(nn.Module):

    def __init__(self, item_id_list: List[str], garment_group_list: List[str], index_group_list: List[str], embedding_dimension: int):
        super().__init__()
        self.item_embedding = StringEmbedding(item_id_list, embedding_dimension)

        # Garment group setup
        self.garment_group_lookup = StringLookup(vocabulary=garment_group_list)
        self.garment_group_size = len(garment_group_list)

        # Index group setup
        self.index_group_lookup = StringLookup(vocabulary=index_group_list)
        self.index_group_size = len(index_group_list)

        input_dim = embedding_dimension + self.garment_group_size + self.index_group_size
        self.dense_nn = nn.Sequential(nn.Linear(input_dim, embedding_dimension), nn.Dropout(0.2), nn.ReLU(), nn.Linear(embedding_dimension, embedding_dimension))

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:

        # Convert article_id strings to embeddings
        item_embedding = self.item_embedding(inputs["article_id"])

        garment_indices = self.garment_group_lookup(inputs["garment_group_name"])
        garment_one_hot = torch.zeros((garment_indices.size(0), self.garment_group_size), dtype=torch.float)
        garment_one_hot.scatter_(1, garment_indices.unsqueeze(1), 1.0)

        # Convert index group strings to one-hot encodings
        index_indices = self.index_group_lookup(inputs["index_group_name"])
        index_one_hot = torch.zeros((index_indices.size(0), self.index_group_size), dtype=torch.float)
        index_one_hot.scatter_(1, index_indices.unsqueeze(1), 1.0)

        # Concatenate all features
        concatenated = torch.cat([item_embedding, garment_one_hot, index_one_hot], dim=1)

        outputs = self.dense_nn(concatenated)

        return outputs


class TwoTowerModel(nn.Module):
    def __init__(self, query_tower: nn.Module, item_tower: nn.Module):
        super().__init__()
        self.query_tower = query_tower
        self.item_tower = item_tower

    def forward(self, batch: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        query_embeddings = self.query_tower(batch)
        item_embeddings = self.item_tower(batch)

        # Normalize embeddings
        query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
        item_embeddings = F.normalize(item_embeddings, p=2, dim=1)

        return query_embeddings, item_embeddings

class RetrievalLoss(nn.Module):

    def __init__(self, temperature: float = None):
        super(RetrievalLoss, self).__init__()
        self.temperature = temperature

    def forward(self, query_embeddings, item_embeddings, labels=None):

        # Calculate dot product (similarity scores) between query and candidates
        similarities = torch.matmul(query_embeddings, item_embeddings.t())  # [batch_size, num_items]

        if self.temperature is not None:
            similarities /= self.temperature

        num_queries = similarities.shape[0]
        # num_candidates = similarities.shape[1]

        if labels is None:
            labels = torch.eye(num_queries)  # [batch_size]

        loss_fn = nn.CrossEntropyLoss(reduce="none")

        loss = loss_fn(similarities, labels)
        return loss

class TopKAccuracyMetric:
    def __init__(self, k_values: List[int]):
        self.k_values = sorted(k_values)
        self.reset()

    def reset(self):
        self.total = 0
        self.correct = {k: 0 for k in self.k_values}

    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        # predictions: [batch_size, num_items] tensor of similarity scores
        # targets: [batch_size] tensor of correct item indices

        # Get the top k predictions for each query
        _, top_indices = torch.topk(predictions, max(self.k_values), dim=1)

        # Convert targets to column vector for broadcasting
        targets = targets.view(-1, 1)

        # Check if correct item is in top k predictions
        for k in self.k_values:
            correct_in_k = (top_indices[:, :k] == targets).any(dim=1)
            self.correct[k] += correct_in_k.sum().item()

        self.total += len(targets)

    def compute(self):
        return {f"top_{k}_accuracy": (self.correct[k] / self.total) * 100 for k in self.k_values}


def evaluate_and_top_k(model: TwoTowerModel, val_loader: DataLoader, unique_items_loader: DataLoader, retrieval_loss_fn: RetrievalLoss, k_values: List[int]) -> Dict[str,float]:
    model.eval()
    metric = TopKAccuracyMetric(k_values)

    # First, compute all item embeddings
    print("[Evaluation] Computing item embeddings...")
    all_item_embeddings = []
    all_item_ids = []

    with torch.no_grad():
        for batch in tqdm(unique_items_loader):
            item_embeddings = model.item_tower(batch)
            all_item_embeddings.append(item_embeddings.cpu())
            all_item_ids.extend(batch["article_id"].cpu())

    all_item_embeddings = torch.cat(all_item_embeddings, dim=0)

    # Create mapping from item_id to index
    item_id_to_idx = {item_id.item(): idx for idx, item_id in enumerate(all_item_ids)}

    print("[Evaluation] Computing top-k accuracy...")
    val_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluation: "):

            # Get query embeddings
            query_embeddings, item_embeddings = model(batch)

            val_loss += retrieval_loss_fn(query_embeddings, item_embeddings)

            # Compute similarity scores with all items
            similarity_scores = torch.matmul(query_embeddings, all_item_embeddings.t())

            item_data = batch["article_id"].astype(int)

            # Get target indices
            target_indices = torch.tensor([item_id_to_idx[item_id] for item_id in item_data])

            # Update metrics
            metric.update(similarity_scores, target_indices)

    metrics = metric.compute()
    val_loss /= len(val_loader)
    metrics['loss'] = val_loss
    for metric in metrics:
        print("Validation Metrics {}: {:4f}".format(metric, metrics[metric]))
    return metrics


class ExperimentTracker:

    def __init__(self, experiment_name: str, tracking_uri: str = "http://localhost:5000"):
        """Initialize MLflow experiment tracker"""
        mlflow.set_tracking_uri(tracking_uri)
        self.experiment = mlflow.set_experiment(experiment_name)
        print("Experiment initialized")

    def log_dataset_info(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, Any]:
        """Log dataset statistics and distributions"""
        dataset_info = {}

        # Basic statistics
        for split_name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
            stats = {
                f"{split_name}_rows": len(df),
                f"{split_name}_unique_users": df["customer_id"].nunique(),
                f"{split_name}_unique_items": df["article_id"].nunique(),
                f"{split_name}_sparsity": 1 - (len(df) / (df["customer_id"].nunique() * df["article_id"].nunique())),
            }
            dataset_info.update(stats)

        # Create distribution plots
        for col in ["age", "price"]:
            fig = px.histogram(train_df, x=col, title=f"{col} Distribution")
            fig.write_html(f"{col}_dist.html")
            mlflow.log_artifact(f"{col}_dist.html")

        # Log temporal distribution
        if "t_dat" in train_df.columns:
            temporal_dist = train_df["t_dat"].value_counts().sort_index()
            fig = px.line(x=temporal_dist.index, y=temporal_dist.values, title="Temporal Distribution of Transactions")
            fig.write_html("temporal_dist.html")
            mlflow.log_artifact("temporal_dist.html")

        return dataset_info

    def log_model_architecture(self, model: nn.Module):
        """Log model architecture details"""
        # Save model summary
        model_info = {"total_params": sum(p.numel() for p in model.parameters()), "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad), "architecture": str(model)}

        with open("model_info.json", "w") as f:
            json.dump(model_info, f, indent=2)
        mlflow.log_artifact("model_info.json")

        return model_info

    def create_confusion_matrix(self, predictions: np.ndarray, labels: np.ndarray, k: int):
        """Create and log confusion matrix for top-k predictions"""
        correct_predictions = (predictions[:, :k] == labels.reshape(-1, 1)).any(axis=1)

        tn = len(correct_predictions) - correct_predictions.sum()
        tp = correct_predictions.sum()

        fig = go.Figure(
            data=go.Heatmap(
                z=[[tp, 0], [0, tn]],
                x=["Predicted Positive", "Predicted Negative"],
                y=["Actual Positive", "Actual Negative"],
                text=[[tp, 0], [0, tn]],
                texttemplate="%{text}",
                textfont={"size": 20},
                colorscale="Viridis",
            )
        )

        fig.update_layout(title=f"Confusion Matrix (Top-{k})")
        fig.write_html(f"confusion_matrix_top_{k}.html")
        mlflow.log_artifact(f"confusion_matrix_top_{k}.html")


class ModelTrainer:
    def __init__(self, model_config: Dict[str, Any], training_config: Dict[str, Any]):
        self.model_config = model_config
        self.training_config = training_config
        self.tracker = ExperimentTracker("two-tower-recommendations")

        self.save_folder_path = Path("models/saved_models")
        self.save_folder_path.mkdir(exist_ok=True, parents=True)

    def train_and_evaluate(self, train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader, unique_item_loader: DataLoader) -> Dict[str, float]:
        """Train model and track experiment with MLflow"""
        with mlflow.start_run():
            # Log configurations
            mlflow.log_params(self.model_config)
            mlflow.log_params(self.training_config)

            # Initialize model and optimizer
            model = self._create_model()
            optimizer = self._create_optimizer(model)
            scheduler = self._create_scheduler(optimizer)

            # Log dataset info
            dataset_info = self.tracker.log_dataset_info(train_loader.dataset.df, val_loader.dataset.df, test_loader.dataset.df)
            mlflow.log_params(dataset_info)

            # Log model architecture
            model_info = self.tracker.log_model_architecture(model)
            mlflow.log_params(model_info)

            # Training loop
            best_val_metrics = {}
            retrieval_loss_fn = RetrievalLoss()

            for epoch in range(self.training_config["num_epochs"]):
                # Train
                train_loss = self._train_epoch(model, train_loader, optimizer, retrieval_loss_fn)
                mlflow.log_metric("train_loss", train_loss, step=epoch)

                # Evaluate
                val_metrics = self._evaluate(model, val_loader, unique_item_loader, retrieval_loss_fn)
                for metric_name, value in val_metrics.items():
                    mlflow.log_metric(f"val_{metric_name}", value, step=epoch)

                # Update scheduler
                if scheduler is not None:
                    scheduler.step(val_metrics["top_100_accuracy"])

                # Save best model
                if not best_val_metrics or val_metrics["top_100_accuracy"] > best_val_metrics["top_100_accuracy"]:
                    best_val_metrics = val_metrics
                    # mlflow.pytorch.log_model(model, "best_model")
                    signature = mlflow.models.infer_signature(train_loader.dataset.df)
                    mlflow.pytorch.log_model(model, "best_model", signature=signature)

                    # Save locally with metadata
                    self._save_model(model, val_metrics, self.training_config, epoch)

                # elif best_val_metrics and val_metrics["top_100_accuracy"] < best_val_metrics["top_100_accuracy"]:
                #     mlflow.pytorch.log_model(model, f"model_epoch_{epoch}")

                # Log learning rate
                current_lr = optimizer.param_groups[0]["lr"]
                mlflow.log_metric("learning_rate", current_lr, step=epoch)

            # Final evaluation on test set and saving
            test_metrics = self._evaluate(model, test_loader, unique_item_loader, retrieval_loss_fn)
            for metric_name, value in test_metrics.items():
                mlflow.log_metric(f"test_{metric_name}", value)

            final_save_path = self._save_model(model, test_metrics, self.training_config)
            print(f"Final model saved to: {final_save_path}")

            return test_metrics

    def _create_model_signature(self, train_loader: DataLoader) -> mlflow.models.ModelSignature:
        """Create model signature from sample batch"""
        # Get a sample batch
        sample_batch = next(iter(train_loader))
        # Define input schema
        input_schema = []
        for key, value in sample_batch.items():

            if isinstance(value, torch.Tensor):
                input_schema.append(ColSpec(name=key, type=mlflow.types.DataType.double))
            elif isinstance(value, np.ndarray):
                if isinstance(value.dtype,np.dtypes.StrDType):
                    input_schema.append(ColSpec(name=key, type=mlflow.types.DataType.string))
                elif isinstance(value.dtype, np.dtypes.IntDType):
                    input_schema.append(ColSpec(name=key, type=str(value.dtype)))

                # input_schema[key] = mlflow.types.DataType.tensor(type=value.dtype, shape=value.shape)

        # Define output schema (embeddings)
        output_schema = Schema([
            TensorSpec(type=np.dtype("float64"), shape=(-1,), name="query_embedding"),
            TensorSpec(type=np.dtype("float64"), shape=(-1,), name="item_embedding"),
            ]
        )

        return mlflow.models.ModelSignature(
            inputs=Schema(input_schema),
            outputs=output_schema
        )

    def _save_model(self, model: nn.Module, metrics: Dict[str, float], params: Dict[str, Any], epoch: int = None):
        """Save model with metadata"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create model name with parameters
        param_str = "_".join([f"{k}-{v}" for k, v in params.items()])
        model_name = f"model_{timestamp}_{param_str}"
        if epoch is not None:
            model_name += f"_epoch_{epoch}"
        model_name = model_name.replace("'","").replace(":","_").replace("{","").replace("}","").replace("-","_").replace(" ","")

        save_path = self.save_folder_path / model_name
        save_path.mkdir(exist_ok=True, parents=True)

        for key, value in metrics.items():
            if isinstance(value,torch.Tensor):
                metrics[key] = value.item()

        # Save model state
        torch.save({"model_state_dict": model.state_dict(), "model_config": self.model_config, "training_config": self.training_config, "metrics": metrics, "params": params}, save_path / "model.pt")

        # Save configs and metrics as JSON for easy reading
        with open(save_path / "metadata.json", "w") as f:
            f.write(json.dumps({"model_config": self.model_config, "training_config": self.training_config, "metrics": metrics, "params": params}, indent=2))

        return save_path

    def _train_epoch(self, model: TwoTowerModel, train_loader: DataLoader, optimizer:torch.optim.Optimizer, retrieval_loss_fn:RetrievalLoss) -> float:
        """Train model for one epoch"""

        train_losses = []
        model.train()
        print_every = 1000

        bformat = "{l_bar}{bar}| {n_fmt}/{total_fmt} {rate_fmt}{postfix}"
        with tqdm(total=len(train_loader), bar_format=bformat) as pbar:

            for batch_idx, train_batch in enumerate(train_loader):

                optimizer.zero_grad()
                query_embeddings, item_embeddings = model(train_batch)
                loss = retrieval_loss_fn(query_embeddings, item_embeddings)
                loss.backward()
                optimizer.step()

                train_losses.append(loss.item())
                mean_train_loss = np.mean(train_losses)

                pbar.set_description(f"Batch: {batch_idx} ")
                pbar.set_postfix_str(" LOSS: {:.4f} ".format(mean_train_loss))
                pbar.update()

                if batch_idx % print_every == 0:
                    print("Training: Batch {0}/{1}. Avg Loss of {2:.4f}".format(batch_idx + 1, len(train_loader), mean_train_loss))

        return mean_train_loss

    def _evaluate(self, model: TwoTowerModel, val_loader: DataLoader, unique_items_loader: DataLoader, retrieval_loss_fn: RetrievalLoss) -> Dict[str,float]:

        return evaluate_and_top_k(model,val_loader,unique_items_loader,retrieval_loss_fn,[100])

    def _create_model(self) -> nn.Module:
        """Create model based on configuration"""
        query_tower = QueryTower(user_id_list=self.model_config["user_ids"], embedding_dimension=self.model_config["embedding_dim"])

        item_tower = ItemTower(
            item_id_list=self.model_config["item_ids"],
            garment_group_list=self.model_config["garment_groups"],
            index_group_list=self.model_config["index_groups"],
            embedding_dimension=self.model_config["embedding_dim"],
        )

        return TwoTowerModel(query_tower, item_tower)

    def _create_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        """Create optimizer based on configuration"""
        optimizer_name = self.training_config["optimizer"].lower()
        lr = self.training_config["learning_rate"]
        weight_decay = self.training_config.get("weight_decay", 0.01)

        if optimizer_name == "adam":
            return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == "adamw":
            return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == "sgd":
            return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    def _create_scheduler(self, optimizer: torch.optim.Optimizer):
        """Create learning rate scheduler based on configuration"""
        scheduler_config = self.training_config.get("scheduler", {})
        if not scheduler_config:
            return None

        scheduler_name = scheduler_config["name"].lower()
        if scheduler_name == "reducelronplateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=scheduler_config.get("factor", 0.5), patience=scheduler_config.get("patience", 2))
        else:
            raise ValueError(f"Unsupported scheduler: {scheduler_name}")

    def _log_example_predictions(self, model: nn.Module, test_loader: DataLoader, item_dataset: Dataset, num_examples: int = 5):
        """Log example predictions for inspection"""
        model.eval()
        examples = []

        with torch.no_grad():
            for batch_data in test_loader:
                # Get predictions
                query_emb = model.get_query_embeddings(batch_data)
                all_item_emb = model.get_item_embeddings_for_dataset(item_dataset)

                similarity = torch.matmul(query_emb, all_item_emb.t())
                _, top_indices = torch.topk(similarity, k=10, dim=1)

                # Store examples
                for i in range(min(num_examples, len(batch_data["customer_id"]))):
                    example = {
                        "customer_id": batch_data["customer_id"][i],
                        "actual_item": batch_data["article_id"][i],
                        "top_predictions": [item_dataset.df.iloc[idx]["article_id"] for idx in top_indices[i].tolist()],
                    }
                    examples.append(example)

                if len(examples) >= num_examples:
                    break

        # Log examples
        with open("example_predictions.json", "w") as f:
            json.dump(examples, f, indent=2)
        mlflow.log_artifact("example_predictions.json")


def run_grid_search(
    param_grid: Dict[str, List[Any]],
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    unique_item_dataloader: DataLoader,
    base_model_config: Dict[str, Any],
    base_training_config: Dict[str, Any],
):
    """Run grid search over hyperparameters"""
    # Create parameter combinations
    param_combinations = list(ParameterGrid(param_grid))

    results = []
    print('Running grid search')
    for params in param_combinations:
        # Update configurations with current parameters
        model_config = base_model_config.copy()
        training_config = base_training_config.copy()

        for param_name, param_value in params.items():
            if param_name in base_model_config:
                model_config[param_name] = param_value
            else:
                training_config[param_name] = param_value

        # Train and evaluate model
        trainer = ModelTrainer(model_config, training_config)
        metrics = trainer.train_and_evaluate(train_loader, val_loader, test_loader, unique_item_dataloader)

        # Store results
        result = {"params": params, "metrics": metrics}
        results.append(result)

    # Create summary report
    summary_df = pd.DataFrame([{**params, **metrics} for result in results for params, metrics in [(result["params"], result["metrics"])]])

    summary_df.to_csv("grid_search_results.csv", index=False)
    mlflow.log_artifact("grid_search_results.csv")

    return summary_df


def load_datasets(preprocessing_function:Dict[str,Callable]=None) -> pd.DataFrame:

    print("Loading dataset")
    articles_df = pd.read_csv("data/articles.csv", encoding="utf-8")
    print("Article shape before preprocessing: ", articles_df.shape)
    articles_df = preprocessing_function['articles'](articles_df)
    print("Article shape after preprocessing: ", articles_df.shape)

    customers_df = pd.read_csv("data/customers.csv", encoding="utf-8")
    print("Customer shape before preprocessing: ", customers_df.shape)
    customers_df = preprocessing_function["customers"](customers_df)
    print("Customer Shape after preprocessing: ", customers_df.shape)

    transaction_df = pd.read_csv("data/transactions_train.csv", encoding="utf-8")
    transaction_df = transaction_df.iloc[:5000]

    print("Transaction shape before preprocessing: ", transaction_df.shape)
    transaction_df = preprocessing_function['transactions'](transaction_df)
    print("Transaction shape after preprocessing: ", transaction_df.shape)

    df = pd.merge(
        pd.merge(
            transaction_df[["article_id", "customer_id", "t_dat", "price", "month_sin", "month_cos"]],
            articles_df[["article_id", "garment_group_name", "index_group_name"]],
            on="article_id",
            how="inner",
        ),
        customers_df[["customer_id", "age", "club_member_status", "age_group"]],
        on="customer_id",
        how="inner",
    )
    return df


def set_seed(seed:int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def main():

    # Load configurations
    with open("models/config/model_config.yaml", "r") as f:
        base_model_config = yaml.safe_load(f)

    with open("models/config/training_config.yaml", "r") as f:
        base_training_config = yaml.safe_load(f)

    # set seed
    SEED = base_training_config['seed']
    BATCH_SIZE = base_training_config['batch_size']
    set_seed(SEED)

    # Load datasets
    preprocessing_function = {"articles": preprocess_articles, "customers": preprocess_customers, "transactions": preprocess_transactions}
    df = load_datasets(preprocessing_function)
    user_id_list = df["customer_id"].unique().tolist()
    item_id_list = df["article_id"].unique().tolist()
    garment_group_list = df["garment_group_name"].unique().tolist()
    index_group_list = df["index_group_name"].unique().tolist()

    base_model_config["user_ids"] = user_id_list
    base_model_config["item_ids"] = item_id_list
    base_model_config["garment_groups"] = garment_group_list
    base_model_config["index_groups"] = index_group_list

    train_df, val_df = train_test_split(df, test_size=0.1, random_state=SEED)
    train_df, test_df = train_test_split(train_df, test_size=0.1, random_state=SEED)

    unique_items_df = df.drop_duplicates("article_id")[["article_id", "garment_group_name", "index_group_name"]]
    unique_items_df["article_id"] = unique_items_df["article_id"].astype(np.int64)

    # Create datasets
    train_dataset = CustomDataset(train_df)
    val_dataset = CustomDataset(val_df)
    test_dataset = CustomDataset(test_df)
    unique_items_dataset = CustomDataset(unique_items_df)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=custom_collate_fn)
    unique_items_loader = DataLoader(unique_items_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print("DataLoader loaded")

    # Define parameter grid
    param_grid = {"embedding_dim": [16, 32, 64, 128], "learning_rate": [0.001, 0.01], "optimizer": ["adam", "adamw", "sgd"], "weight_decay": [0.01, 0.001]}

    # Run grid search
    results_df = run_grid_search(
        param_grid=param_grid,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        unique_item_dataloader=unique_items_loader,
        base_model_config=base_model_config,
        base_training_config=base_training_config,
    )

    print("Summary: ", results_df)

if __name__ == "__main__":

    main()
