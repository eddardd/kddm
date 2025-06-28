import umap
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm


def visualize_teacher_student_features(
    teacher_model,
    student_model,
    dataloader,
    method="umap",  # "umap" or "tsne"
    n_batches=100,
    label_names=None,
    device="cuda" if torch.cuda.is_available() else "cpu",
    alpha=0.6,
    s=10,
    axes=None,  # Accept external axes
    return_embeddings=False,  # Option to return embeddings,
    filename=None
):
    """
    Visualize teacher and student features side-by-side using true labels
    for coloring.

    Args:
        teacher_model: PyTorch model returning (features, predictions).
        student_model: PyTorch model returning (features, predictions).
        dataloader: PyTorch DataLoader for the dataset.
        method: "umap" or "tsne".
        n_samples: Max number of points to plot.
        label_names: Optional class names for legend.
        device: "cuda" or "cpu".
        alpha: Point transparency.
        s: Point size.
        axes: Optional matplotlib axes array (axes[0], axes[1]). If None,
        creates new figure.
        return_embeddings: If True, returns (t_embeddings, s_embeddings).
    """
    # --- Feature Extraction ---
    teacher_features, true_labels = [], []
    student_features = []

    teacher_model.eval()
    student_model.eval()
    with torch.no_grad():
        for batch_idx, (x, y) in tqdm(
            enumerate(dataloader), total=max(len(dataloader), n_batches)
        ):
            x, y = x.to(device), y.to(device)

            t_features, _ = teacher_model(x)
            s_features, _ = student_model(x)

            teacher_features.append(t_features.cpu().numpy())
            student_features.append(s_features.cpu().numpy())
            true_labels.append(y.cpu().numpy())

            if n_batches and batch_idx >= n_batches - 1:
                break

    teacher_features = np.concatenate(teacher_features, axis=0)
    student_features = np.concatenate(student_features, axis=0)
    true_labels = np.concatenate(true_labels, axis=0)

    # --- Dimensionality Reduction ---
    combined_features = np.vstack([teacher_features, student_features])

    if method == "tsne":
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
    elif method == "umap":
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15)
    else:
        raise ValueError("Method must be 'tsne' or 'umap'.")

    combined_embeddings = reducer.fit_transform(combined_features)
    t_embeddings = combined_embeddings[:len(teacher_features)]  # type: ignore
    s_embeddings = combined_embeddings[len(teacher_features):]  # type: ignore

    # --- Plotting ---
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Teacher features
    scatter = axes[0].scatter(
        t_embeddings[:, 0], t_embeddings[:, 1],  # type: ignore
        c=true_labels, cmap="tab10", alpha=alpha, s=s
    )
    axes[0].set_title("Teacher Features")
    plt.colorbar(
        scatter, ax=axes[0], ticks=np.unique(true_labels), label="Class")
    if label_names:
        axes[0].legend(
            handles=scatter.legend_elements()[0],
            labels=label_names,
            bbox_to_anchor=(1.05, 1),
            loc='upper left'
        )

    # Student features
    scatter = axes[1].scatter(
        s_embeddings[:, 0], s_embeddings[:, 1],  # type: ignore
        c=true_labels, cmap="tab10", alpha=alpha, s=s
    )
    axes[1].set_title("Student Features")
    plt.colorbar(
        scatter, ax=axes[1], ticks=np.unique(true_labels), label="Class")
    plt.tight_layout()

    if filename:
        plt.savefig(filename, bbox_inches='tight')

    if return_embeddings:
        return t_embeddings, s_embeddings
