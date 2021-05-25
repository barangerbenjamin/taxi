from sklearn.model_selection import GridSearchCV
class ParamTrainer():
    def __init__(self):
        pass

    def set_gridsearch(self, pipeline):
        self.grid_search = GridSearchCV(
            pipeline, 
            param_grid={
                'features__distance__standardscaler__copy': [True],
                'model__min_samples_leaf': [3],
                'model__oob_score': [True],
                'model__min_weight_fraction_leaf': [0.0, 0.1]
            },
            cv=5
        )
        return self
    
    def fit(self, X_train, y_train):
        self.grid_search.fit(X_train, y_train)

    def score(self, X_test, y_test):
        self.score = self.grid_search.score(X_test, y_test)
        return self.score