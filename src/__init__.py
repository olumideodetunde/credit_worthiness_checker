'''This module implements the logging functionality for the entire project.'''
import os
import sys
import logging

# logging_format = "[%(asctime)s: %(levelname)s: %(module)s] %(message)s"
# log_dir = "logs"
# log_filepath = os.path.join(log_dir, "entire_logs.log")
# os.makedirs(log_dir, exist_ok=True)
# logging.basicConfig(
#     level = logging.INFO,
#     format = logging_format,
#     handlers = [
#         logging.FileHandler(log_filepath),
#         logging.StreamHandler(sys.stdout)
#     ]
# )
# logger = logging.getLogger("credit_worthiness_logger")

def get_logger():
    logging_format = "[%(asctime)s: %(levelname)s: %(module)s] %(message)s"
    log_dir = "logs"
    log_filepath = os.path.join(log_dir, "entire_logs.log")
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger("credit_worthiness_logger")
    logger.setLevel(logging.INFO)

    # Create formatter
    formatter = logging.Formatter(logging_format)

    # Create file handler and set level to info
    file_handler = logging.FileHandler(log_filepath)
    file_handler.setFormatter(formatter)

    # Create stream handler
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger

src_logger = get_logger()
