from sklearn import model_selection

class CVTemplate:
    def __init__(self, predict_fun, evaluate_fun):
        """

        :param predict_fun: responsible of fitting a model and return predictions for provided data (X_train, X_test, y_train)
        :param evaluate_fun: responsible of return a string representing a summary of quality of predictions compared to correct labels
        """
        self.predict = predict_fun
        self.evaluate = evaluate_fun

    def test_processing(self, processing_funs, train_data):
        """

        :param processing_funs: a dictionary of functions to be used for generating the training data (X, Y)
        :return:
        """
        for key, p_fun in processing_funs.items():
            print(p_fun['des'])
            X, Y = p_fun['fun'](train_data)
            p_fun[key]['eval_res'] = self._holdout_test(X, Y)

    def _holdout_test(self, X, Y):
        X_train, X_test, y_train, y_test = model_selection.train_test_split(X, Y,
                                                            test_size=0.33,
                                                            random_state=42)

        predictions = self.predict(X_train, X_test, y_train)

        eval_res = self.evaluate(y_test, predictions)
        return eval_res

    def _dummy_test(self, params):
        data = []
        for param in np.arange(64, 1024, step=256):
            predictions = predict_fun(X_train, X_test, y_train, n_estimators)
            res = metrics.f1_score(y_test, predictions, average='weighted')
            data.append((n_estimators, res))
        res = pd.DataFrame(data, columns=['n_estimators', 'eval'])