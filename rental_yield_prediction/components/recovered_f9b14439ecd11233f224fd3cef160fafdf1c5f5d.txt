from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score
from rental_yield_prediction.entity.artifact_entity import MetricArtifact, CVMetricArtifact
from rental_yield_prediction.exception.exception import CustomException
import sys

def get_metrics(y_true, y_pred)->MetricArtifact:
    try:
        r2_score1 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        metric_artifact = MetricArtifact(
            r2_score=r2_score1,
            mae=mae
        )
        return metric_artifact
    except Exception as e:
        raise CustomException(e, sys)
    

def get_cv_metrics(best_model, X, y)->CVMetricArtifact:
    try:
        cv_r2_score = cross_val_score(estimator=best_model, X=X, y=y, scoring="r2", cv=10).mean()
        cv_mae_score = abs(cross_val_score(estimator=best_model, X=X, y=y, scoring="neg_mean_absolute_error", cv=10).mean())

        cv_metric_artifact = CVMetricArtifact(
            r2_score=cv_r2_score,
            mae=cv_mae_score
        )
        return cv_metric_artifact
    except Exception as e:
        raise CustomException(e, sys)
    