"""
This is a boilerplate pipeline 'PreparacaoDados'
generated using Kedro 0.19.12
"""

from kedro.pipeline import node, Pipeline, pipeline  # noqa
from . import nodes

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
                nodes.preprocess_data,
                inputs="raw_kobe_dev",
                outputs="processed_kobe_dev",
                tags=['preprocessing']),
        node(
                nodes.preprocess_data,
                inputs="raw_kobe_prod",
                outputs="processed_kobe_prod",
                tags=['preprocessing']),
        node(
                nodes.prepare_train_and_test_data,
                inputs=["processed_kobe_dev", "params:session_id", "params:test_size"],
                outputs=["train_kobe_dev", "test_kobe_dev"],
                tags=['split_train_test']),
    ])
