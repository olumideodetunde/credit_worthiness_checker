'''This module implements the logging functionality for the entire project.'''
import os
import sys
import logging

logging_format = "[%(asctime)s: %(levelname)s: %(module)s] %(message)s"
log_dir = "logs"
log_filepath = os.path.join(log_dir, "entire_logs.log")
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    level = logging.INFO,
    format = logging_format,
    handlers = [
        logging.FileHandler(log_filepath),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("credit_worthiness_logger")


PROJECT_ROOT = "/Users/olumide/Documents/Self Improvement - ML/MLapps/credit_worthiness_checker" 