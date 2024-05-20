'''This module is the main script for preparing ml dataset, visualising the data,
training the model and exporting the model'''
from src import src_logger as logger
from src.data_preprocessing import main as data_prep_main
from src.plotting import main as data_plot_main
from src.feature_engineering import main as feature_eng_main
from src.train import main as train_main

PROJECT_STAGE = "Dataset Preparation"
try:
    logger.info(">>>>>> Started <<<<<< %s", PROJECT_STAGE)
    INPUT_LIST = [
    "artifacts/data_prep/raw/parquet_files/train/train_base.parquet",
    "artifacts/data_prep/raw/parquet_files/train/train_static_cb_0.parquet",
    "artifacts/data_prep/raw/parquet_files/train/train_person_1.parquet",
    "artifacts/data_prep/raw/parquet_files/train/train_applprev_1_1.parquet"
    ]
    OUTPUT_DIR = "artifacts/data_prep/output"
    data_prep_main(input_paths=INPUT_LIST, output_path=OUTPUT_DIR)
    logger.info(">>>>>> Ended <<<<<< %s", PROJECT_STAGE)

except Exception as e:
    logger.exception(e)
    raise e

PROJECT_STAGE = "Data Visualisation"
try:
    logger.info(">>>>> Generating plots for data visualisation")
    data_plot_main(file="artifacts/data_prep/output/train.parquet",
                   output="artifacts/data_prep/plots")
    logger.info(">>>> Plots generated successfully <<<<<<<")
    logger.info(">>>>> Ended process for data visualisation <<<<< %s", PROJECT_STAGE)

except Exception as e:
    logger.exception(e)
    raise e

PROJECT_STAGE = "Feature Engineering"
try:
    logger.info(">>>>>> Started <<<<<< %s", PROJECT_STAGE)
    INPUT_LIST = [
    "artifacts/data_prep/output/train.parquet",
    "artifacts/data_prep/output/dev.parquet"
    ]
    OUTPUT_DIR = "artifacts/feature_eng/output"
    feature_eng_main(input_paths=INPUT_LIST, output_path=OUTPUT_DIR)
    logger.info(">>>>>> Ended <<<<<< %s", PROJECT_STAGE)
except Exception as e:
    logger.exception(e)
    raise e

PROJECT_STAGE = "Model Training and Export"
try:
    logger.info(">>>>>> Started <<<<<< %s", PROJECT_STAGE)
    INPUT_LIST = [
    "artifacts/feature_eng/output/ml_train.parquet",
    "artifacts/feature_eng/output/ml_dev.parquet"
    ]
    OUTPUT_DIR = "artifacts/model_training/output"
    train_main(input_paths=INPUT_LIST, output_path=OUTPUT_DIR)
    logger.info(">>>>>> Ended <<<<<< %s", PROJECT_STAGE)
except Exception as e:
    logger.exception(e)
    raise e
#EOF
