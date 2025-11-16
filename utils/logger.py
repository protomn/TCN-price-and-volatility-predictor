import logging
import sys
from pathlib import Path
from datetime import datetime

def setup_logger(name: str, log_level: str = "INFO"):

    logger_dir = Path("logs")
    logger_dir.mkdir(exist_ok = True)

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level))

    if logger.handlers:
        return logger
    
    detailed_format = logging.Formatter(
        '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        datefmt = '%Y-%m-%d %H:%M:%S' 
    )

    simple_format = logging.Formatter(
        '%(levelname)s | %(message)s'
    )

    file_handler = logging.FileHandler(
        logger_dir / f"tcn_sys_{datetime.now():%Y%m%d}.log"
    )

    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_format)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(simple_format)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger