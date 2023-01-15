from mushroom.entity import artifact_entity,config_entity
from mushroom.exception import mushroomException
from mushroom.logger import logging
from typing import Optional
import os,sys
from sklearn.svm import SVC
from mushroom import utils
from sklearn.metrics import f1_score 

class ModelTrainer:
    def __init__(self,model_trainer_config:config_entity.ModelTrainerConfig,
    data_transformation_artifact:artifact_entity.DataTransformationArtifact):
        try:
            logging.info("Model Trainer")
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact

        except Exception as e:
            raise mushroomException(e,sys)

    def fine_tune(self):
        try:
            #write code for Grid Search CV
            pass
        except Exception as e:
            raise mushroomException(e,sys)

    def train_model(self,x,y):
        try:
            svc_clf = SVC()
            svc_clf.fit(x,y)
            return svc_clf
        except Exception as e:
            raise mushroomException(e,sys)

    def initiate_model_trainer(self,)-> artifact_entity.ModelTrainerArtifact:
        try:
            logging.info("Loading train and test array")
            train_arr = utils.load_numpy_array_data(file_path =self.data_transformation_artifact.transformed_train_path)
            test_arr = utils.load_numpy_array_data(file_path =self.data_transformation_artifact.transformed_test_path)
            logging.info(f"Splitting input and target feature from both train and test arrays")
            x_train,y_train = train_arr[:,:-1],train_arr[:,-1]
            x_test,y_test = test_arr[:,:-1],test_arr[:,-1]
            logging.info(f"Train the model")
            model = self,train_model(x = x_train,y = y_train)

            logging.info(f"Calculating the f1 score")
            yhat_train = model.predict(x_train)
            f1_train_score = f1_score(y_true = y_train,y_pred = yhat_train)

            logging.info(f"Calculating the f1 score")
            yhat_test = model.predict(x_test)
            f1_test_score = f1_score(y_true = y_test,y_pred = yhat_test)
            
            logging.info(f"The f1 train score {f1_train_score} and f1 test score {f1_test_score}")

            #check for overfitting or underfitting and expected score

            logging.info(f"Checking id our model is underfitting or not")
            if f1_test_score < self.model_trainer_config.expected_score:
                raise Exception(f"Model is unbale to give expected accuracy {self.model_trainer_config.expected_score} : Actual score {f1_test_score}")
                logging.info(f"Checking if our model is overfitting or not")
                diff = abs(f1_train_score - f1_test_score)

                if diff > self.model_trainer_config.overfiting_threshold:
                    raise Exception(f"Train and test score diff :{diff} is more than overfitting threshold {self.model_trainer_config.overfiting_threshold}")

                #save the trained model
                logging.info(f"Saving mode object")

                utils.save_object(file_path = self.model_trainer_config.model_path,obj = model)

                #prepare artifact
                logging.info(f"Prepare the artifact")
                model_trainer_artifact = artifact_entity.ModelTrainerArtifact(model_path = self.model_trainer_config.model_path, f1_train_score = f1_train_score, f1_test_score = f1_test_score)
                logging.info(f"Model trainer artifact : {model_trainer_artifact}")
                return model_trainer_artifact 
            
        except Exception as e:
            raise mushroomException(e,sys)



