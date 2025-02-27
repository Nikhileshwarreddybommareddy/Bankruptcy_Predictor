import os,sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from pathlib import Path
from src.components.data_transformation import DataTransformation,DataTransformationConfig
from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer
@dataclass
class DataIngestionConfig:
    project_root: Path = Path(__file__).resolve().parent.parent.parent  # Adjust path as needed
    train_data_path: Path = project_root / "data/processed/train.csv"
    test_data_path: Path = project_root / "data/processed/test.csv"
    raw_data_path: Path = project_root / "data/raw/data.csv"

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered Data Ingestion Pipeline")
        try:
            raw_path = self.ingestion_config.raw_data_path  # Fixed access method
            # Check if raw data exists
            if not os.path.exists(raw_path):
                logging.error(f"Raw data file not found: {raw_path}")
                raise FileNotFoundError(f"Raw data file not found: {raw_path}")
            df = pd.read_csv(raw_path)
            df = df.rename(columns={'Bankrupt?':'Bankrupt'})
            df = df[[' Borrowing dependency', ' Current Liability to Current Assets',
                    ' Debt ratio %', " Net Income to Stockholder's Equity",
                    ' Net Value Per Share (A)',
                    ' Net profit before tax/Paid-in capital',
                    ' Operating Gross Margin',
                    ' Per Share Net profit before tax (Yuan ¥)',
                    ' Persistent EPS in the Last Four Seasons',
                    ' ROA(A) before interest and % after tax',
                    ' Working Capital to Total Assets', 'Bankrupt']]
            logging.info(f"Successfully read the raw data from {raw_path}")

            logging.info("Train Test Split Initiated")
            train,test = train_test_split(df,test_size=0.2,random_state=42)
            logging.info("Completed Train Test Split Initiated")
             # Ensure output directory exists
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            train.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info(f"Train & Test set have been saved to {self.ingestion_config.train_data_path} & {self.ingestion_config.test_data_path} ")
            return train,test
        except Exception as e:
            raise CustomException(e,sys)

if __name__ == "__main__":
    ingestion = DataIngestion()
    train,test = ingestion.initiate_data_ingestion()  # Explicitly calling the method
    
    data_transformation = DataTransformation()
    data_transformation.initiate_data_transformation(DataIngestionConfig.train_data_path,DataIngestionConfig.test_data_path)

    modeltrainer=ModelTrainer()
    print(modeltrainer.initiate_model_trainer())