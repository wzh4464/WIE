import logging
from datetime import datetime
import os


def setup_logging(name, seed, save_dir, gpu=0, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create handlers
    c_handler = logging.StreamHandler()
    os.makedirs(save_dir, exist_ok=True)

    infl_type = name.split("_")[1] + "_" if "infl_" in name else ""
    f_handler = logging.FileHandler(
        os.path.join(
            save_dir,
            f"log_{infl_type}gpu_{gpu}_{seed:03d}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
        ),
        mode="w",
    )
    c_handler.setLevel(level)
    f_handler.setLevel(level)

    # Create formatters and add to handlers
    # Check if MPS is used, if so, display "mps" instead of "GPU{gpu}"
    gpu_identifier = "mps" if gpu == "mps" else f"GPU{gpu}"
    formatter = logging.Formatter(
        f"%(asctime)s - %(name)s - {gpu_identifier} - %(levelname)s - %(message)s"
    )
    c_handler.setFormatter(formatter)
    f_handler.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    return logger
