from mushroom.predictor import ModelResolver
from mushroom.entity.config_entity import ModelPusherConfig
from mushroom.exception import mushroomException
import os,sys
from mushroom.utils import load_object,save_object
from mushroom.logger import logging
from mushroom.entity.artifact_entity import DataTransformationArtifact,ModelTrainerArtifact,ModelPusherArtifact

class ModelPusher:

    def __init__(self,model_pusher_config:ModelPusherConfig,
    data_transformation_artifact : DataTransformationArtifact,
    model_trainer_artifact :ModelTrainerArtifact):
        try:
            logging.info("Data Transformation")
            self.model_pusher_config = model_pusher_config
            self.data_transformation_artifact =data_transformation_artifact
            self.model_trainer_artifact = model_trainer_artifact
            self.model_resolver = ModelResolver(model_registry =self.model_pusher_config.saved_model_dir)
        except Exception as e:
            raise mushroomException(e, sys)

    def intitate_model_pusher(self,)->ModelPusherArtifact:
        try:
            #load object
            logging.info(f"Loading transformer model and target encoder")
            transformer = load_object(file_path = self.data_transformation_artifact.transform_object_path)
            model = load_object(file_path = self.model_trainer_artifact.model_path)
            target_endoder = load_object(file_path = self.data_transformation_artifact.target_encoder_path)

            #model pusher dir
            logging.info(f"Saving model into model pusher directory")

            save_object(file_path = self.model_pusher_config.pusher_transformer_path,obj = transformer)
            save_object(file_path = self.model_pusher_config.pusher_model_path , obj = model)
            save_object(file_path = model_pusher_config.pusher_target_encoder_path,obj = target_encoder)

            #saved model dir
            logging.info(f"Saving model in saved model dir")
            transformer_path = self.model_resolver.get_latest_save_transformer_path()
            model_path = self.model_resolver.get_latest_save_model_path()
            target_endoder_path = self.model_resolver.get_latest_save_target_encoder_path()

            save_object(file_path = transformer_path,obj = transformer)
            save_object(file_path = model_path,obj = model)
            save_object(file_path = target_endoder_path,obj = target_encoder)

            model_pusher_artifact = ModelPusherArtifact(pusher_model_dir = self.model_pusher_config.pusher_model_dir,
            saved_model_dir =self.model_pusher_config.saved_model_dir)
            logging.info(f"Model Pusher Artifact :{model_pusher_artifact}")
            return model_pusher_artifact

        except Exception as e:
            raise mushroomException(e, sys)
            
