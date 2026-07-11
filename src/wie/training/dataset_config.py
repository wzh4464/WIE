# Dictionary to hold configurations for (dataset, network) pairs
DATASET_NETWORK_CONFIG = {
    ("mnist", "logreg"): {
        "num_epoch": 21,
        "batch_size": 64,
        "lr": 0.003,
        "decay": True,
        "n_tr": 256,
        "n_val": 256,
        "n_test": 256,
        # not good
    },
    ("mnist", "dnn"): {
        "num_epoch": 21,
        "batch_size": 64,
        "lr": 0.0005,
        "decay": False,
        "n_tr": 2000,
        "n_val": 256,
        "n_test": 256,
        # used
    },
    ("mnist", "cnn"): {
        "num_epoch": 21,
        "batch_size": 64,
        "lr": 0.0004,
        "decay": False,
        "n_tr": 256,
        "n_val": 256,
        "n_test": 256,
        # not tried
    },
    ("mnist", "resnet56_emnist"): {
        "num_epoch": 10,
        "batch_size": 64,
        "lr": 0.1,
        "decay": True,
        "n_tr": 1024,
        "n_val": 100,
        "n_test": 100,
        # not tried
    },
    # 20news is not good
    ("20news", "logreg"): {
        "num_epoch": 21,
        "batch_size": 64,
        "lr": 0.1,
        "decay": True,
        "n_tr": 256,
        "n_val": 256,
        "n_test": 256,
    },
    ("20news", "dnn"): {
        "num_epoch": 21,
        "batch_size": 64,
        "lr": 0.1,
        "decay": True,
        "n_tr": 200,
        "n_val": 200,
        "n_test": 200,
    },
    ("20news", "cnn"): {
        "num_epoch": 21,
        "batch_size": 64,
        "lr": 0.01,
        "decay": True,
        "n_tr": 200,
        "n_val": 200,
        "n_test": 200,
    },
    ("adult", "logreg"): {
        "num_epoch": 21,
        "batch_size": 64,
        "lr": 0.1,
        "decay": True,
        "n_tr": 256,
        "n_val": 256,
        "n_test": 256,
        # good
    },
    ("adult", "dnn"): {
        "num_epoch": 21,
        "batch_size": 64,
        "lr": 0.3,
        "decay": True,
        "n_tr": 256,
        "n_val": 256,
        "n_test": 256,
        # good
    },
    ("adult", "cnn"): {
        "num_epoch": 21,
        "batch_size": 60,
        "lr": 0.02,
        "decay": True,
        "n_tr": 200,
        "n_val": 200,
        "n_test": 200,
        # not tried
    },
    # cifar is not tried
    ("cifar", "logreg"): {
        "num_epoch": 21,
        "batch_size": 60,
        "lr": 0.01,
        "decay": True,
        "n_tr": 200,
        "n_val": 200,
        "n_test": 200,
    },
    ("cifar", "dnn"): {
        "num_epoch": 21,
        "batch_size": 64,
        "lr": 0.18,
        "decay": True,
        "n_tr": 256,
        "n_val": 1256,
        "n_test": 200,
    },
    ("cifar", "cnn"): {
        "num_epoch": 50,
        "batch_size": 128,
        "lr": 0.01,
        "decay": True,
        "n_tr": 256,
        "n_val": 256,
        "n_test": 256,
    },
    ("emnist", "logreg"): {
        "num_epoch": 21,
        "batch_size": 64,
        "lr": 0.003,
        "decay": True,
        "n_tr": 256,
        "n_val": 256,
        "n_test": 256,
        # well, just so so
    },
    ("emnist", "dnn"): {
        "num_epoch": 21,
        "batch_size": 64,
        "lr": 0.06,
        "decay": True,
        "n_tr": 256,
        "n_val": 256,
        "n_test": 256,
        # good
    },
    ("emnist", "resnet56_emnist"): {
        "num_epoch": 10,
        "batch_size": 64,
        "lr": 0.1,
        "decay": True,
        "n_tr": 8000,
        "n_val": 400,
        "n_test": 400,
        # good
    },
    ("emnist", "cnn"): {
        "num_epoch": 21,
        "batch_size": 64,
        "lr": 0.2,
        "decay": True,
        "n_tr": 256,
        "n_val": 256,
        "n_test": 256,
        # not tried
    },
    ("cifar", "resnet56_cifar"): {
        "num_epoch": 15,
        "batch_size": 16,
        "lr": 0.1,
        "decay": True,
        "n_tr": 800,
        "n_val": 100,
        "n_test": 100,
    },
    ("mnist", "resnet9"): {
        "num_epoch": 10,
        "batch_size": 64,
        "lr": 0.1,
        "decay": True,
        "n_tr": 1024,
        "n_val": 100,
        "n_test": 100,
    },
    ("cifar", "resnet9_cifar"): {
        "num_epoch": 15,
        "batch_size": 16,
        "lr": 0.1,
        "decay": True,
        "n_tr": 800,
        "n_val": 100,
        "n_test": 100,
    },
    ("emnist", "resnet9"): {
        "num_epoch": 10,
        "batch_size": 64,
        "lr": 0.1,
        "decay": True,
        "n_tr": 1024,
        "n_val": 100,
        "n_test": 100,
    },
    ("emnist", "tinyvit_emnist"): {
        "num_epoch": 30,
        "batch_size": 64,
        "lr": 0.3,
        "decay": True,
        "n_tr": 8000,
        "n_val": 100,
        "n_test": 100,
    },
    ("cifar", "tinyvit_cifar"): {
        "num_epoch": 25,
        "batch_size": 64,
        "lr": 0.3,
        "decay": True,
        "n_tr": 8000,
        "n_val": 100,
        "n_test": 100,
    },
    ("emnist", "mobilenetv2_nores"): {
        "num_epoch": 10,
        "batch_size": 64,
        "lr": 0.01,
        "decay": True,
        "n_tr": 8000,
        "n_val": 100,
        "n_test": 100,
    },
    ("cifar", "vit"): {
        "num_epoch": 1,
        "batch_size": 64,
        "lr": 0.1,
        "decay": True,
        "n_tr": 8000,
        "n_val": 100,
        "n_test": 100,
    },
    ("sentiment", "bert"): {
        "num_epoch": 5,  # TextAttack风格参数
        "batch_size": 16,  # TextAttack风格batch size
        "lr": 2e-5,  # TextAttack风格学习率
        "decay": False,  # TextAttack不使用学习率衰减
        "n_tr": 2048,  # TextAttack风格训练样本数
        "n_val": 256,  # TextAttack风格验证样本数
        "n_test": 256,  # TextAttack风格测试样本数
        # TextAttack specific parameters
        "weight_decay": 0.01,
        "max_length": 128,
        "num_warmup_steps": 500,
        "gradient_accumulation_steps": 1,
        "validation_split": 0.1,
        "model_name_or_path": "bert-base-uncased",
        "num_labels": 2,
    },
}


def fetch_training_params(dataset: str, network: str):
    """Retrieve the training parameters for a given dataset and network pair."""
    pair_key = (dataset, network)
    if pair_key not in DATASET_NETWORK_CONFIG:
        raise ValueError(
            f"Configuration for dataset {dataset} with network {network} not found."
        )
    return DATASET_NETWORK_CONFIG[pair_key]
