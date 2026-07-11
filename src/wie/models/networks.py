import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import torchvision.models as tv_models
except Exception:
    tv_models = None  # Lazy optional import; only needed for ViT variants
import numpy as np
import logging
import os

NETWORK_REGISTRY = {}


def register_network(key):
    def decorator(cls):
        NETWORK_REGISTRY[key] = cls
        return cls

    return decorator


def get_network(key, input_dim, logger=None):
    if logger is None:
        logger = logging.getLogger(__name__)

    if key not in NETWORK_REGISTRY:
        logger.error(f"Network {key} not found in registry.")
        raise ValueError(f"Network {key} not found in registry.")

    # Add input validation for different network types
    if key in ["cnn", "resnet18", "resnet56"]:
        _validate_input_dimensions(
            input_dim,
            key,
            " requires input_dim to be a tuple of (channels, height, width)",
        )
    elif key in ["cnn_cifar", "resnet18_cifar", "resnet56_cifar", "vit"]:
        _validate_input_dimensions(
            input_dim,
            key,
            " requires input_dim to be a tuple of (height, width, channels)",
        )
        # For CIFAR-specific models, automatically convert (C, H, W) format to (H, W, C)
        if (
            isinstance(input_dim, (tuple, list))
            and len(input_dim) == 3
            and (input_dim[0] == 3 and input_dim[1] == 32 and input_dim[2] == 32)
        ):
            input_dim = (input_dim[1], input_dim[2], input_dim[0])
            logger.debug(f"Converted input_dim to (H, W, C) format: {input_dim}")

    logger.debug(f"Creating network: {key} with input dimension: {input_dim}")
    return NETWORK_REGISTRY[key](input_dim, logger)


def _validate_input_dimensions(input_dim, key, arg2):
    if not isinstance(input_dim, (tuple, list)):
        raise ValueError(f"Network {key}{arg2}")
    if len(input_dim) != 3:
        raise ValueError(f"Network {key}{arg2}")
    if not all(isinstance(x, int) for x in input_dim):
        raise ValueError(f"Network {key} requires all input dimensions to be integers")


class BaseModel(nn.Module):
    def __init__(self, logger=None):
        super().__init__()
        self.logger = logger or logging.getLogger(__name__)

    def preprocess_input(self, x):
        raise NotImplementedError

    def forward(self, x):
        # self.logger.debug(f"BaseModel forward pass with input shape: {x.shape}")
        x = self.preprocess_input(x)
        return self.model(x)


class NetList(nn.Module):
    def __init__(self, list_of_models, logger=None):
        super(NetList, self).__init__()
        self.logger = logger or logging.getLogger(__name__)
        self.models = nn.ModuleList(list_of_models)
        self.logger.debug(f"Created NetList with {len(list_of_models)} models")

    def forward(self, x, idx=0):
        self.logger.debug(
            f"NetList forward pass with input shape: {x.shape}, using model at index: {idx}"
        )
        return self.models[idx](x)

    def __iter__(self):
        return iter(self.models)

    def __len__(self):
        return len(self.models)

    def get_model(self, *indices):
        model = self
        for idx in indices:
            if isinstance(model, NetList):
                model = model.models[idx]
            else:
                self.logger.error("Invalid indices for nested NetList access")
                raise ValueError("Invalid indices for nested NetList access")
        return model


@register_network("bert")
class BertClassifier(BaseModel):
    """
    BERT-based binary classifier head.
    Expects input x of shape (B, 2, L): x[:,0,:] -> input_ids, x[:,1,:] -> attention_mask.
    Returns a single logit per sample suitable for BCEWithLogitsLoss.
    """

    def __init__(
        self,
        input_dim=None,
        logger=None,
        model_name: str = "bert-base-uncased",
        freeze_bert_layers: int = 0,
    ):
        super(BertClassifier, self).__init__(logger)
        self.freeze_bert_layers = (
            freeze_bert_layers  # Number of BERT layers to freeze (0 = no freezing)
        )
        # Lazy import to avoid hard dependency if not used
        try:
            from transformers import (
                AutoModel,
                AutoConfig,
                BertConfig,
                BertModel,
                RobertaConfig,
                RobertaModel,
            )
        except Exception as e:
            raise ImportError(
                "transformers is required for BertClassifier. Please install it."
            ) from e

        self.model_name = os.getenv("HF_TEXT_MODEL", model_name)
        self.logger.debug(f"Initializing text model: {self.model_name}")
        self.hidden_size = 768  # sensible default
        # Try local cache first; if unavailable, try online download; then a final offline-safe fallback
        config = None
        model_built = False
        # 1) Local cache
        try:
            config = AutoConfig.from_pretrained(self.model_name, local_files_only=True)
            self.hidden_size = getattr(config, "hidden_size", self.hidden_size)
            try:
                self.bert = AutoModel.from_pretrained(
                    self.model_name, local_files_only=True
                )
            except Exception:
                self.bert = AutoModel.from_config(config)
            model_built = True
        except Exception:
            model_built = False
        # 2) Online download (if network available)
        if not model_built:
            try:
                config = AutoConfig.from_pretrained(self.model_name)
                self.hidden_size = getattr(config, "hidden_size", self.hidden_size)
                self.bert = AutoModel.from_pretrained(self.model_name)
                model_built = True
            except Exception:
                model_built = False
        # 3) Offline-safe fallback minimal model
        if not model_built:
            name_l = self.model_name.lower()
            if "roberta" in name_l:
                config = RobertaConfig(hidden_size=self.hidden_size)
                self.bert = RobertaModel(config)
            else:
                config = BertConfig(hidden_size=self.hidden_size)
                self.bert = BertModel(config)

        # Head dropout (configurable via env var)
        # Priority: HF_TEXT_DROPOUT -> BERT_DROPOUT -> default 0.3
        dropout_env = os.getenv("HF_TEXT_DROPOUT") or os.getenv("BERT_DROPOUT")
        p_dropout = 0.3
        if dropout_env is not None:
            try:
                p_val = float(dropout_env)
                if 0.0 <= p_val < 1.0:
                    p_dropout = p_val
                    if self.logger:
                        self.logger.info(
                            f"Using custom head dropout p={p_dropout} from env"
                        )
                else:
                    if self.logger:
                        self.logger.warning(
                            f"Invalid dropout p={p_val}; must be in [0,1). Using default {p_dropout}."
                        )
            except Exception as e:
                if self.logger:
                    self.logger.warning(
                        f"Failed to parse dropout env var '{dropout_env}': {e}. Using default {p_dropout}."
                    )
        self.dropout = nn.Dropout(p=p_dropout)  # 增加dropout防止过拟合
        self.head = nn.Linear(self.hidden_size, 1)

        # Apply layer freezing if requested
        self._apply_layer_freezing()

    def _apply_layer_freezing(self):
        """Freeze the first N layers of BERT model."""
        if self.freeze_bert_layers <= 0:
            return

        # Freeze embeddings
        if hasattr(self.bert, "embeddings"):
            for param in self.bert.embeddings.parameters():
                param.requires_grad = False

        # Freeze encoder layers
        if hasattr(self.bert, "encoder") and hasattr(self.bert.encoder, "layer"):
            layers_to_freeze = min(
                self.freeze_bert_layers, len(self.bert.encoder.layer)
            )
            for i in range(layers_to_freeze):
                for param in self.bert.encoder.layer[i].parameters():
                    param.requires_grad = False
            if self.logger:
                self.logger.info(f"Frozen {layers_to_freeze} BERT encoder layers")

    def unfreeze_bert_layers(self, num_layers_to_unfreeze: int = -1):
        """Unfreeze BERT layers for fine-tuning.
        Args:
            num_layers_to_unfreeze: Number of layers to unfreeze from the end.
                                   -1 means unfreeze all layers.
        """
        if num_layers_to_unfreeze == 0:
            return

        # Unfreeze all if requested
        if num_layers_to_unfreeze == -1:
            for param in self.bert.parameters():
                param.requires_grad = True
            if self.logger:
                self.logger.info("Unfrozen all BERT layers")
            return

        # Unfreeze specific number of layers from the end
        if hasattr(self.bert, "encoder") and hasattr(self.bert.encoder, "layer"):
            num_layers = len(self.bert.encoder.layer)
            start_unfreeze = max(0, num_layers - num_layers_to_unfreeze)
            for i in range(start_unfreeze, num_layers):
                for param in self.bert.encoder.layer[i].parameters():
                    param.requires_grad = True
            if self.logger:
                self.logger.info(
                    f"Unfrozen last {min(num_layers_to_unfreeze, num_layers)} BERT encoder layers"
                )

    def preprocess_input(self, x):
        # x: Tensor (B, 2, L) potentially float; convert to long
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        if x.dim() != 3 or x.size(1) != 2:
            raise ValueError(
                f"BertClassifier expects x of shape (B, 2, L); got {tuple(x.shape)}"
            )
        input_ids = x[:, 0, :].long()
        attention_mask = x[:, 1, :].long()
        return input_ids, attention_mask

    def forward(self, x):
        input_ids, attention_mask = self.preprocess_input(x)
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Prefer pooler_output if available; else use CLS token from last_hidden_state
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            pooled = outputs.pooler_output
        else:
            pooled = outputs.last_hidden_state[:, 0]
        pooled = self.dropout(pooled)
        logits = self.head(pooled)
        return logits


@register_network("logreg")
class LogReg(BaseModel):
    def __init__(self, input_dim, logger=None):
        super(LogReg, self).__init__(logger)

        # dim of input_dim
        if isinstance(input_dim, (tuple, list, torch.Size)):
            input_dim = np.prod(input_dim)

        self.model = nn.Linear(input_dim, 1)
        self.logger.debug(f"Created LogReg model with input dimension: {input_dim}")

    def preprocess_input(self, x):
        # First ensure x is a PyTorch tensor
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)

        # Then process according to the dimension
        if x.dim() == 1:
            return x.view(1, -1)
        elif x.dim() == 2:
            return x
        else:
            return x.view(x.size(0), -1)


@register_network("dnn")
class DNN(BaseModel):
    def __init__(self, input_dim, logger=None, m=None):
        super(DNN, self).__init__(logger)
        if m is None:
            m = [8, 8]

        if isinstance(input_dim, (tuple, list, torch.Size)):
            total_input_dim = np.prod(input_dim)
        else:
            total_input_dim = input_dim

        self.model = nn.Sequential(
            nn.Linear(total_input_dim, m[0]),
            nn.ReLU(),
            nn.Linear(m[0], 1),
        )
        self.logger.debug(
            f"Created DNN model with input dimension: {input_dim}, hidden layers: {m}"
        )

    def preprocess_input(self, x):
        return x.view(x.size(0), -1)

    def param_diff(self, other):
        if not isinstance(other, DNN):
            self.logger.error("Can only compare with another DNN instance")
            raise ValueError("Can only compare with another DNN instance")

        diff = {}
        for (name1, param1), (name2, param2) in zip(
            self.named_parameters(), other.named_parameters()
        ):
            if name1 != name2:
                self.logger.error(f"Parameter names do not match: {name1} vs {name2}")
                raise ValueError(f"Parameter names do not match: {name1} vs {name2}")
            diff[name1] = param1.data - param2.data
        return diff

    def param_diff_norm(self, other, norm_type=2):
        diff = self.param_diff(other)
        total_norm = sum(
            torch.norm(diff_tensor, p=norm_type).item() ** norm_type
            for diff_tensor in diff.values()
        )
        return total_norm ** (1 / norm_type)

    def print_param_diff(self, other, threshold=1e-6):
        diff = self.param_diff(other)
        for name, diff_tensor in diff.items():
            if torch.any(torch.abs(diff_tensor) > threshold):
                self.logger.debug(f"Difference in {name}:")
                self.logger.debug(diff_tensor)


@register_network("cnn")
class CNN(BaseModel):
    def __init__(self, input_dim, logger=None, m=[32, 64]):
        super(CNN, self).__init__(logger)
        if not isinstance(input_dim, (tuple, list)):
            raise ValueError(
                "input_dim must be a tuple or list of (channels, height, width)"
            )
        if len(input_dim) != 3:
            raise ValueError(
                "input_dim must have exactly 3 elements (channels, height, width)"
            )
        in_channels, height, width = input_dim
        self.m = m
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, self.m[0], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(self.m[0], self.m[1], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(self.m[1] * (height // 4) * (width // 4), 1),
        )
        self.logger.debug(
            f"Created CNN model with input dimension: {input_dim}, channels: {m}"
        )

    def preprocess_input(self, x):
        self.logger.debug(f"CNN input shape before preprocessing: {x.shape}")
        if x.dim() == 2:
            batch_size = x.size(0)
            x = x.view(batch_size, self.m[0], self.m[1], self.m[2])
        elif x.dim() != 4:
            raise ValueError(f"Unexpected input dimension: {x.shape}")
        self.logger.debug(f"CNN input shape after preprocessing: {x.shape}")
        return x

    def forward(self, x):
        self.logger.debug(f"CNN forward input shape: {x.shape}")

        for i, layer in enumerate(self.model):
            if isinstance(layer, nn.Conv2d):
                self.logger.debug(
                    f"Layer {i} ({type(layer).__name__}) expects input channels: {layer.in_channels}"
                )
            elif isinstance(layer, nn.Linear):
                self.logger.debug(
                    f"Layer {i} ({type(layer).__name__}) expects input features: {layer.in_features}"
                )

            x = layer(x)
            self.logger.debug(
                f"Layer {i} ({type(layer).__name__}) output shape: {x.shape}"
            )

        return x


@register_network("cnn_cifar")
class CNN_CIFAR(BaseModel):
    def __init__(self, input_dim, logger=None, m=[32, 32, 64, 64, 128, 128]):
        super(CNN_CIFAR, self).__init__(logger)
        height, width, in_channels = input_dim  # Correct order for CIFAR data
        self.m = m
        self.conv_layer = nn.Sequential(
            # Conv Layer block 1
            nn.Conv2d(
                in_channels=in_channels, out_channels=m[0], kernel_size=3, padding=1
            ),
            nn.BatchNorm2d(m[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=m[0], out_channels=m[1], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Conv Layer block 2
            nn.Conv2d(in_channels=m[1], out_channels=m[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(m[2]),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=m[2], out_channels=m[3], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Conv Layer block 3
            nn.Conv2d(in_channels=m[3], out_channels=m[4], kernel_size=3, padding=1),
            nn.BatchNorm2d(m[4]),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=m[4], out_channels=m[5], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc_layer = nn.Sequential(
            nn.Linear(4 * 4 * m[5], 1),  # Output 1 value for binary classification
        )

        self.logger.debug(
            f"Created CNN_CIFAR model with input dimension: {input_dim}, channels: {self.m}"
        )

    def forward(self, x):
        self.logger.debug(f"CNN_CIFAR forward input shape: {x.shape}")

        # Detect and adjust input format
        if x.dim() == 4 and x.shape[1] == 3:
            x = x.permute(0, 2, 3, 1)  # Convert to (B,H,W,C)
            self.logger.debug(f"Converted input from (B,C,H,W) to (B,H,W,C): {x.shape}")

        # Normal forward pass flow
        x = x.permute(0, 3, 1, 2)  # Convert to PyTorch standard format (B,C,H,W)
        self.logger.debug(f"Shape after permute: {x.shape}")

        x = self.conv_layer(x)
        self.logger.debug(f"Shape after conv layers: {x.shape}")

        x = x.reshape(x.size(0), -1)
        self.logger.debug(f"Shape after flattening: {x.shape}")

        x = self.fc_layer(x)
        self.logger.debug(f"Final output shape: {x.shape}")

        return x

    def flatten(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.conv_layer(x)
        return x.reshape(x.size(0), -1)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, logger=None):
        super(BasicBlock, self).__init__()
        self.logger = logger or logging.getLogger(__name__)

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels * self.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels * self.expansion),
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(identity)
        out = self.relu(out)

        return out


class ResNet(BaseModel):
    def __init__(self, block, layers, num_classes=1, logger=None):
        super(ResNet, self).__init__(logger)
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, return_indices=False
        )

        # Create layers based on the length of layers list
        if len(layers) == 4:  # ImageNet style
            self.layer1 = self._make_layer(block, 64, layers[0])
            self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512 * block.expansion, num_classes)
        elif len(layers) == 3:  # CIFAR/EMNIST style
            self.layer1 = self._make_layer(block, 64, layers[0])
            self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(256 * block.expansion, num_classes)
        else:
            raise ValueError(
                f"Invalid number of layers: {len(layers)}. Expected 3 or 4."
            )

        self.logger.debug(f"Created ResNet model with {len(layers)} layers")

    def _make_layer(self, block, out_channels, num_blocks, stride=1):
        layers = [block(self.in_channels, out_channels, stride)]
        self.in_channels = out_channels * block.expansion
        layers.extend(
            block(self.in_channels, out_channels) for _ in range(1, num_blocks)
        )
        return nn.Sequential(*layers)

    def forward(self, x):
        self.logger.debug(f"ResNet forward input shape: {x.shape}")

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        if hasattr(self, "layer4"):  # Only for ImageNet style
            x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


@register_network("resnet18")
class ResNet18(ResNet):
    def __init__(self, input_dim, logger=None):
        super(ResNet18, self).__init__(
            BasicBlock, [2, 2, 2, 2], num_classes=1, logger=logger
        )
        # Adjust first conv layer for MNIST (1 channel) or CIFAR10 (3 channels)
        if input_dim[0] == 1:  # MNIST
            self.conv1 = nn.Conv2d(
                1, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
        self.logger.debug(f"Created ResNet-18 model with input dimension: {input_dim}")


@register_network("resnet56")
class ResNet56(ResNet):
    def __init__(self, input_dim, logger=None):
        super(ResNet56, self).__init__(
            BasicBlock, [9, 9, 9, 9], num_classes=1, logger=logger
        )
        # Adjust first conv layer for MNIST (1 channel) or CIFAR10 (3 channels)
        if input_dim[0] == 1:  # MNIST
            self.conv1 = nn.Conv2d(
                1, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
        self.logger.debug(f"Created ResNet-56 model with input dimension: {input_dim}")


@register_network("resnet18_cifar")
class ResNet18CIFAR(ResNet):
    def __init__(self, input_dim, logger=None):
        super(ResNet18CIFAR, self).__init__(
            BasicBlock, [2, 2, 2, 2], num_classes=1, logger=logger
        )
        # Adjust for CIFAR10 input size (32x32)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.maxpool = nn.MaxPool2d(
            kernel_size=2, stride=2, padding=1, return_indices=False
        )
        self.logger.debug(
            f"Created ResNet-18-CIFAR model with input dimension: {input_dim}"
        )


@register_network("resnet56_cifar")
class ResNet56CIFAR(BaseModel):
    def __init__(self, input_dim, logger=None):
        super(ResNet56CIFAR, self).__init__(logger)
        if not isinstance(input_dim, (tuple, list)):
            raise ValueError(
                "input_dim must be a tuple or list of (height, width, channels)"
            )
        if len(input_dim) != 3:
            raise ValueError(
                "input_dim must have exactly 3 elements (height, width, channels)"
            )
        height, width, in_channels = input_dim
        self.in_channels = 16  # Initial number of channels is 16

        # First convolutional layer, do not use stride=2
        self.conv1 = nn.Conv2d(
            in_channels, 16, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        # Standard CIFAR ResNet does not use MaxPool

        # 3 stages, each stage has 9 blocks
        self.layer1 = self._make_layer(BasicBlock, 16, 9, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 32, 9, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 64, 9, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, 1)  # For binary classification tasks, the output is 1

        self.logger.debug(
            f"Created ResNet-56-CIFAR model with input dimension: {input_dim}"
        )

    def _make_layer(self, block, out_channels, num_blocks, stride=1):
        """Create a layer containing multiple blocks"""
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride, self.logger))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        # Ensure correct input format (B, C, H, W)
        if x.dim() == 4 and (x.shape[1] == 32 or x.shape[3] == 3):
            x = x.permute(0, 3, 1, 2)  # Adjust to (B, C, H, W)
        # self.logger.debug(f"ResNet56CIFAR forward input shape: {x.shape}")

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


@register_network("resnet56_emnist")
class ResNet56EMNIST(ResNet):
    def __init__(self, input_dim, logger=None):
        super(ResNet56EMNIST, self).__init__(
            BasicBlock,
            [9, 9, 9],  # 3 layers with 9 blocks each
            num_classes=1,
            logger=logger,
        )
        # Adjust for EMNIST input size (1x28x28)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        # Override the layer creation to use CIFAR-style channel dimensions
        self.in_channels = 16  # Start with 16 channels after first conv
        self.layer1 = self._make_layer(BasicBlock, 16, 9, stride=1)  # 16 -> 16
        self.layer2 = self._make_layer(BasicBlock, 32, 9, stride=2)  # 16 -> 32
        self.layer3 = self._make_layer(BasicBlock, 64, 9, stride=2)  # 32 -> 64

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, 1)  # For binary classification

        self.logger.debug(
            f"Created ResNet-56-EMNIST model with input dimension: {input_dim}"
        )

    def forward(self, x):
        # Ensure input is in correct format (B, C, H, W)
        if x.dim() == 4 and x.shape[1] == 1:
            pass  # Already in correct format
        elif x.dim() == 3:  # (B, H, W)
            x = x.unsqueeze(1)  # Add channel dimension
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")

        self.logger.debug(f"ResNet56EMNIST forward input shape: {x.shape}")

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


@register_network("resnet9")
class ResNet9(ResNet):
    def __init__(self, input_dim, logger=None):
        super(ResNet9, self).__init__(
            BasicBlock,
            [1, 1, 1, 1],  # 9 layers = 2 + (1+1)*4 - 1
            num_classes=1,
            logger=logger,
        )
        # Adjust first conv layer for MNIST (1 channel) or CIFAR10 (3 channels)
        if input_dim[0] == 1:  # MNIST
            self.conv1 = nn.Conv2d(
                1, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
        self.logger.debug(f"Created ResNet-9 model with input dimension: {input_dim}")


@register_network("resnet9_cifar")
class ResNet9CIFAR(ResNet):
    def __init__(self, input_dim, logger=None):
        super(ResNet9CIFAR, self).__init__(
            BasicBlock,
            [1, 1, 1, 1],  # 9 layers = 2 + (1+1)*4 - 1
            num_classes=1,
            logger=logger,
        )
        # Adjust for CIFAR10 input size (32x32)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.maxpool = nn.MaxPool2d(
            kernel_size=2, stride=2, padding=1, return_indices=False
        )
        self.logger.debug(
            f"Created ResNet-9-CIFAR model with input dimension: {input_dim}"
        )


@register_network("tinyvit_emnist")
class TinyViTEMNIST(BaseModel):
    def __init__(self, input_dim, logger=None, num_classes=1):
        super(TinyViTEMNIST, self).__init__(logger)
        if isinstance(input_dim, (tuple, list)):
            channels, height, width = input_dim
        else:
            raise ValueError("input_dim must be a tuple or list of (C, H, W)")

        self.logger.debug(
            f"Creating TinyViT for EMNIST/CIFAR with input_dim={input_dim}"
        )

        self.patch_size = 4  # Patch size smaller than ImageNet version
        self.num_patches = (height // self.patch_size) * (width // self.patch_size)
        embed_dim = 192

        self.patch_embed = nn.Conv2d(
            channels, embed_dim, kernel_size=self.patch_size, stride=self.patch_size
        )
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=3,
                dim_feedforward=embed_dim * 4,
                activation="gelu",
                batch_first=True,
            ),
            num_layers=4,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.position_embed = nn.Parameter(
            torch.zeros(1, 1 + self.num_patches, embed_dim)
        )
        self.head = nn.Linear(embed_dim, num_classes)

    def preprocess_input(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        return x

    def forward(self, x):
        x = self.preprocess_input(x)
        B = x.size(0)
        x = self.patch_embed(x)  # (B, embed_dim, H/patch, W/patch)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.position_embed[:, : x.size(1)]

        x = self.transformer(x)
        x = self.head(x[:, 0])  # Take CLS token output
        return x


@register_network("tinyvit_cifar")
class TinyViTCIFAR(BaseModel):
    def __init__(self, input_dim, logger=None, num_classes=1):
        super(TinyViTCIFAR, self).__init__(logger)
        if isinstance(input_dim, (tuple, list)):
            height, width, channels = input_dim  # CIFAR format is (H, W, C)
        else:
            raise ValueError(
                "input_dim must be a tuple or list of (height, width, channels)"
            )

        self.logger.debug(f"Creating TinyViT for CIFAR with input_dim={input_dim}")

        self.patch_size = 4  # Patch size smaller than ImageNet version
        self.num_patches = (height // self.patch_size) * (width // self.patch_size)
        embed_dim = 192

        self.patch_embed = nn.Conv2d(
            channels, embed_dim, kernel_size=self.patch_size, stride=self.patch_size
        )
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=3,
                dim_feedforward=embed_dim * 4,
                activation="gelu",
                batch_first=True,
            ),
            num_layers=4,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.position_embed = nn.Parameter(
            torch.zeros(1, 1 + self.num_patches, embed_dim)
        )
        self.head = nn.Linear(embed_dim, num_classes)

    def preprocess_input(self, x):
        # Handle CIFAR format (B, H, W, C) -> (B, C, H, W)
        if x.dim() == 4 and x.shape[3] == 3:  # If input is (B, H, W, C)
            x = x.permute(0, 3, 1, 2)  # Convert to (B, C, H, W)
        elif x.dim() == 3:  # If input is (H, W, C)
            x = x.permute(2, 0, 1).unsqueeze(0)  # Convert to (1, C, H, W)
        return x

    def forward(self, x):
        x = self.preprocess_input(x)
        B = x.size(0)
        x = self.patch_embed(x)  # (B, embed_dim, H/patch, W/patch)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.position_embed[:, : x.size(1)]

        x = self.transformer(x)
        x = self.head(x[:, 0])  # Take CLS token output
        return x


@register_network("mobilenetv2_nores")
class MobileNetV2NoRes(BaseModel):
    def __init__(self, input_dim, logger=None, num_classes=1):
        super(MobileNetV2NoRes, self).__init__(logger)
        if isinstance(input_dim, (tuple, list)):
            channels, height, width = input_dim
        else:
            raise ValueError("input_dim must be a tuple or list of (C, H, W)")

        self.logger.debug(
            f"Creating MobileNetV2 (no-residual) with input_dim={input_dim}"
        )

        # Define inverted residual block without shortcut
        def inverted_residual_block(in_channels, out_channels, stride, expand_ratio):
            hidden_dim = in_channels * expand_ratio
            return nn.Sequential(
                # Expansion
                nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # Depthwise convolution
                nn.Conv2d(
                    hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False
                ),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # Projection
                nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        self.features = nn.Sequential(
            nn.Conv2d(channels, 32, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
            inverted_residual_block(32, 64, stride=1, expand_ratio=1),
            inverted_residual_block(64, 128, stride=2, expand_ratio=6),
            inverted_residual_block(128, 128, stride=1, expand_ratio=6),
            inverted_residual_block(128, 256, stride=2, expand_ratio=6),
            inverted_residual_block(256, 256, stride=1, expand_ratio=6),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(256, num_classes)

    def preprocess_input(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        return x

    def forward(self, x):
        x = self.preprocess_input(x)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


@register_network("vit")
class ViTForCIFAR(nn.Module):
    # (B, C, H, W)
    # (C, H, W) -> (B, C, H, W)
    def __init__(self, input_dim, logger=None, model_name="vit_b_16", pretrained=True):
        super(ViTForCIFAR, self).__init__()
        self.logger = logger

        if self.logger:
            self.logger.debug(
                f"Initializing ViT model: {model_name}, pretrained={pretrained}"
            )

        # Load pretrained ViT model (ImageNet-1K)
        if tv_models is None:
            raise ImportError(
                "torchvision is required for ViT models but is not available"
            )
        if model_name == "vit_b_16":
            self.backbone = tv_models.vit_b_16(
                weights="IMAGENET1K_V1" if pretrained else None
            )
        elif model_name == "vit_l_16":
            self.backbone = tv_models.vit_l_16(
                weights="IMAGENET1K_V1" if pretrained else None
            )
        elif model_name == "vit_b_32":
            self.backbone = tv_models.vit_b_32(
                weights="IMAGENET1K_V1" if pretrained else None
            )
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        # Get the input dimension of the classification head
        hidden_size = self.backbone.heads.head.in_features

        # Replace the classification head for binary classification (two classes in CIFAR)
        self.backbone.heads.head = nn.Linear(hidden_size, 1)

        if self.logger:
            self.logger.debug(
                "Modified ViT model head to output 1 class (binary classification)"
            )

    def forward(self, x):
        # Ensure the input tensor format is correct (B, C, H, W)
        if x.dim() == 3:
            x = x.unsqueeze(1)  # Add channel dimension

        # If it is (B, H, W, C), convert to (B, C, H, W)
        if x.size(3) == 3 or x.size(3) == 1:
            x = x.permute(0, 3, 1, 2)

        # ViT requires input to be 224x224
        # If the input is not 3-channel, duplicate it to 3-channel
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)

        # Adjust the input to 224x224 (standard input size for ViT)
        if x.size(2) != 224 or x.size(3) != 224:
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)

        return self.backbone(x)
