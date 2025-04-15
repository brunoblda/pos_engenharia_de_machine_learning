"""
This is a boilerplate pipeline 'Aplicacao'
generated using Kedro 0.19.12
"""

from kedro.pipeline import node, Pipeline, pipeline  # noqa
from . import nodes


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=nodes.model_prod,
            inputs=["logistic_model_local", "processed_kobe_prod"],
            outputs=["metrics_prod_report", "production_model_local"],
            name="model_prod_reporting"
        )
    ])
