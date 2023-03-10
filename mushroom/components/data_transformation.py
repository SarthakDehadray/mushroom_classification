from mushroom.exception import mushroomException
from mushroom.logger import logging
from mushroom.entity import artifact_entity,config_entity
from scipy.stats import ks_2samp
from typing import Optional
import os,sys
import pandas as pd
from mushroom import utils
import numpy as np 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from imblearn.combine import SMOTETomek
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler 
from mushroom.config import TARGET_COLUMN

class DataTransformation:

    def __init__(self,data_transformation_config:config_entity.DataTransformationConfig,data_ingestion_artifact:artifact_entity.DataIngestionArtifact):
        try:
            logging.info("Data Transformation")
            self.data_transformation_config = data_transformation_config
            self.data_ingestion_artifact = data_ingestion_artifact

        except Exception as e:
            raise mushroomException(e,sys)

    @classmethod
    def get_data_transformer_object(cls)->Pipeline:
        try:
            simple_imputer = SimpleImputer(strategy = "constant",fill_value = "?")
            robust_scaler = RobustScaler()
            pipeline = Pipeline(steps = [("Imputer",simple_imputer),("RobustScaler",robust_scaler)])
            return pipeline

        except Exception as e:
            raise mushroomException(e,sys)

    def initiate_data_transformation(self,)->artifact_entity.DataTransformationArtifact:
        try:
            #reading training and test file
            train_df = pd.read_csv(self.data_ingestion_artifact.train_file_path)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)

            #selecting input feature for train and test dataframe
            input_feature_train_df = df.drop(TARGET_COLUMN,axis = 1)
            input_feature_test_df = df.drop(TARGET_COLUMN,axis = 1)

            #selecting target feature for train and test dataframe

            target_feaute_train_df = train_df[TARGET_COLUMN]
            target_feaute_test_df = test_df[TARGET_COLUMN]

            label_encoder = LabelEncoder()
            label_encoder.fit(target_feature_train_df)

            #transformation on target columns
            target_feature_train_arr = label_encoder.transform(target_feature_train_arr)
            target_feature_test_arr = label_encoder.transform(target_feature_test_arr)

            transformation_pipeline = DataTransformation.get_data_transformer_object()
            transformation_pipeline.fit(input_feature_train_df)

            #transforming input features
            input_feature_train_arr = transformation_pipeline.transform(input_feature_train_df)
            input_feature_test_arr = transformation_pipeline.transform(input_feature_test_df)

            #target encoder 
            train_arr = np.c_[input_feature_train_arr,target_feature_train_arr]
            test_arr = np.c_[input_feature_test_arr,target_feature_test_arr]
        
            #save numpy array
            utils.save_numpy_array_data(file_path = self.data_transformation_config.transformed_train_path,array = train_arr)
            utils.save_numpy_array_data(file_path = self.data_transformation_config.transformed_test_path,array = test_arr)

            utils.save_object(file_path = self.data_transformation_config.transform_object_path,obj = transformation_pipeline)
            utils.save_object(file_path = self.data_transformation_config.target_encoder_path,obj = label_encoder)

            data_transformation_artifact = artifact_entity.DataTransformationArtifact(
                transform_object_path = self.data_transformation_config.transform_object_path,
                transformed_train_path = self.data_transformation_config.transformed_train_path,
                transformed_test_path = self.data_transformation_config.transformed_test_path,
                target_encoder_path = self.data_transformation_config.target_encoder_path
            )

            logging.info(f"Data Transformation object {data_transformation_artifact}")
            return data_transformation_artifact
            
        except Exception as e:
            return mushroomException