from src.data_process import process_pdf

import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - [%(name)s] - %(levelname)s - %(message)s"
)

pdf_path = "data/10K.pdf"

if __name__ == "__main__":
    vector_db = process_pdf(pdf_path)
    print("Vector database created successfully.")
