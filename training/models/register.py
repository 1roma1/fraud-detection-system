from training.models.boosting import CBModel, LGBModel, XGBModel

REGISTER = {"xgboost": XGBModel, "lgboost": LGBModel, "cboost": CBModel}
