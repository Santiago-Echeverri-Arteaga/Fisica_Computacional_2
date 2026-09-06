"""Regenera la colección completa de notebooks sin resultados almacenados."""

from build_classical_notebooks import (
    build_boosting_stacking,
    build_dbscan_meanshift,
    build_explainability,
    build_hierarchical,
    build_imbalance,
    build_kmeans_gmm,
    build_regression,
    build_reproducibility,
    build_trees_forests_bagging,
)
from build_deep_architectures import (
    build_architecture_zoo,
    build_autoencoder,
    build_convolution_numpy,
    build_gan,
    build_lenet_alexnet,
    build_physics_timeseries,
    build_rnn_lstm_gru,
    build_seq2seq,
    build_transfer_learning,
)
from build_initial_notebooks import build_evaluation_notebook, build_knn_svm_notebook
from build_neural_foundations import (
    build_activations,
    build_gradient_backprop,
    build_losses_optimizers_regularization,
    build_neuron_feedforward,
    build_pytorch_mlp,
    build_tensorflow_mlp,
)
from build_topics_project import build_project_template, build_reinforcement_learning, build_transformers_llm


BUILDERS = [
    build_evaluation_notebook,
    build_reproducibility,
    build_regression,
    build_knn_svm_notebook,
    build_trees_forests_bagging,
    build_boosting_stacking,
    build_imbalance,
    build_explainability,
    build_dbscan_meanshift,
    build_kmeans_gmm,
    build_hierarchical,
    build_neuron_feedforward,
    build_activations,
    build_gradient_backprop,
    build_losses_optimizers_regularization,
    build_tensorflow_mlp,
    build_pytorch_mlp,
    build_convolution_numpy,
    build_lenet_alexnet,
    build_architecture_zoo,
    build_transfer_learning,
    build_autoencoder,
    build_gan,
    build_rnn_lstm_gru,
    build_seq2seq,
    build_physics_timeseries,
    build_reinforcement_learning,
    build_transformers_llm,
    build_project_template,
]


if __name__ == "__main__":
    for builder in BUILDERS:
        builder()
    print(f"{len(BUILDERS)} notebooks regenerados.")
