import time
import logging
import os
from datetime import datetime
import numpy as np
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


# --- Configuration ---
# Define the log file path
# log_directory = "logs" # You can change this
log_filename = "my_application.log"

logging.basicConfig(
    level=logging.INFO,  # Set the minimum logging level (e.g., INFO, DEBUG, WARNING, ERROR, CRITICAL)
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filename=log_filename,
    filemode="a",  # 'a' for append, 'w' for overwrite (default is 'a' if not specified with filename)
)

# Create a logger instance
logger = logging.getLogger(__name__)

# --- Logging Examples ---
# logger.info("This is an informational message.")
# logger.warning("This is a warning message.")
# logger.error("This is an error message.")
# logger.debug("This is a debug message. (Won't show unless level is DEBUG)")


console_handler = logging.StreamHandler()
console_handler.setLevel(
    logging.DEBUG
)  # You can set a different level for console output
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


check_df = pd.read_csv("check_df.csv", dtype=str)
filenames = check_df["filenames"].tolist()
client = OpenAI()


for file in filenames:
    row = check_df[check_df["filenames"] == file].iloc[0]
    # print(row)
    # print(check_df)
    # for _, row in check_df.iterrows():
    if pd.isnull(row["input_file_id"]):
        batch_input_file = client.files.create(
            file=open(row["filenames"], "rb"), purpose="batch"
        )
        batch_input_file_id = batch_input_file.id
        # batch_input_file_id = file + "_input"
        check_df.loc[check_df["filenames"] == file, "input_file_id"] = (
            batch_input_file_id
        )
        check_df.to_csv("check_df.csv", index=False)
        row = check_df[check_df["filenames"] == file].iloc[0]
        logger.info(file + " Sent for batch input file " + batch_input_file_id)
    if pd.isnull(row["batch_id"]):
        batch_obj = client.batches.create(
            input_file_id=row["input_file_id"],
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"description": f"Antiquotadu discuss emotion analysis_{file}"},
        )
        batch_id = batch_obj.id
        # batch_id = file + "_batch"
        check_df.loc[check_df["filenames"] == file, "batch_id"] = batch_id
        check_df.to_csv("check_df.csv", index=False)
        row = check_df[check_df["filenames"] == file].iloc[0]
        logger.info(file + " Started batch " + str(batch_obj))
    if pd.isnull(row["output_file_id"]):
        while True:
            batch_obj = client.batches.retrieve(row["batch_id"])
            if batch_obj.status == "completed":
                break
            if batch_obj.status in ["failed", "expired", "cancelled"]:
                logger.error(file, "Error or problem", str(batch_obj))
                raise Exception(str(batch_obj))
            logger.info(file + " Waiting for batch " + str(batch_obj))
            time.sleep(3600)
        check_df.loc[check_df["filenames"] == file, "output_file_id"] = (
            batch_obj.output_file_id
        )
        check_df.to_csv("check_df.csv", index=False)
        logger.info(file + " Completed Batch " + str(batch_obj))

    #     pass
    # elif pd.isnull()

# logger.info("This message will go to both the file and the console.")
