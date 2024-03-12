from src import logger
from src.data_prep.make_dataset import main as data_prep_main
from src.data_prep.plotting import main as data_plot_main

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
    data_plot_main(file="artifacts/data_prep/output/train.parquet", output="artifacts/plots")
    logger.info(">>>> Plots generated successfully <<<<<<<")
    logger.info(">>>>> Ended process for data visualisation <<<<< %s", PROJECT_STAGE)

except Exception as e:
    logger.exception(e)
    raise e
