from kedro.pipeline import node, Pipeline, pipeline
from . import nodes

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
         node(
            func=nodes.train_logistic_regression,
            inputs=["train_kobe_dev", "params:session_id"],
            outputs="logistic_model_local",
            name="train_logistic_regression"
        ),
        node(
            func=nodes.train_decision_tree,
            inputs=["train_kobe_dev", "params:session_id"],
            outputs="tree_model_local",
            name="train_decision_tree"
        ),
        node(
            func=nodes.train_best_model,
            inputs=["train_kobe_dev", "params:session_id"],
            outputs="best_model_local",
            name="train_best_model"
        ),
        node(
            func=nodes.evaluate_and_log_models,
            inputs=["logistic_model_local", "tree_model_local", "best_model_local", "test_kobe_dev"],
            outputs=None,
            name="evaluate_and_log_models_node"
        )
    ])
    