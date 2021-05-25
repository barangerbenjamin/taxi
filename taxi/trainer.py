from taxi.mlflow import MLFlowBase
from taxi.data import get_data, clean_df, holdout   
from taxi.model import get_model
from taxi.pipeline import get_pipeline
from taxi.metrics import compute_rmse
from taxi.paramtrainer import ParamTrainer
import joblib

class Trainer(MLFlowBase):
    def __init__(self):
        super().__init__("[UK] [LDN] [bbaranger] taxifare_recap", "https://mlflow.lewagon.co")

    def fit(self):
        self.pipeline.fit(self.X_train, self.y_train)

    def train(self):
        # GET DATA
        df = get_data()
        nrows = df.shape[0]
        # CLEAN_DATA
        df = clean_df(df)

        # HOLDOUT
        self.X_train, self.X_test, self.y_train, self.y_test = holdout(df)
        
        # GET MODEL
        model = get_model()
        
        # GET PIPELINE
        self.pipeline = get_pipeline(model)
       
        # FIT PIPELINE
        self.fit()
        
        # PREDICT
        y_pred = self.pipeline.predict(self.X_test)
        
        # EVALUATE
        rmse = compute_rmse(y_pred, self.y_test)
        
        # MLFLOW
        # mlflow create run
        self.mlflow_create_run()
        # log params
        self.mlflow_log_param('model_name', model.__class__.__name__)
        self.mlflow_log_param('nrows', nrows)
        # log metrics
        self.mlflow_log_metric('rmse', rmse)
        

        # GRIDSEARCH
        param_trainer = ParamTrainer()
        param_trainer.set_gridsearch(self.pipeline).fit(self.X_train, self.y_train)
        score_grid = param_trainer.score(self.X_test, self.y_test)
        # LOG GRIDSEARCH
        for key, value in param_trainer.grid_search.best_params_.items():
            self.mlflow_log_param(key, value)
        self.mlflow_log_metric('score', score_grid)
        
        # SAVE PIPELINE
        joblib.dump(self.pipeline, 'pipeline.joblib')
        

if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()
    print('training done!')