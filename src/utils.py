import ot
import torch
import typing as t


def kl_divergence(
    logits_student: torch.Tensor,
    logits_teacher: torch.Tensor,
    temperature: float
) -> torch.Tensor:
    return torch.nn.functional.kl_div(
        input=torch.nn.functional.log_softmax(
            logits_teacher / temperature, dim=1),
        target=torch.nn.functional.log_softmax(
            logits_student / temperature, dim=1),
        reduction="batchmean",
        log_target=True
    ) * (temperature ** 2)


def batched_sinkhorn(
    probs_student: torch.Tensor,
    probs_teacher: torch.Tensor,
    ground_cost: torch.Tensor,
    eps: float,
    max_iter: int = 1000,
    tol: float = 1e-6
) -> t.Tuple[torch.Tensor, torch.Tensor]:
    B, c = probs_student.shape

    # Compute Gibbs kernel (broadcasting across B)
    K = torch.exp(-ground_cost / eps).expand(B, c, c)  # Shape (B, c, c)

    # Initialize dual variables
    U = torch.ones_like(probs_student)  # Shape (B, c)
    V = torch.ones_like(probs_teacher)  # Shape (B, c)

    for _ in range(max_iter):
        U_prev = U.clone()

        # Ensure correct batch matrix multiplication
        U = probs_student / (
            torch.bmm(U.unsqueeze(1), K.transpose(1, 2)).squeeze(1) + 1e-8)
        V = probs_teacher / (
            torch.bmm(V.unsqueeze(1), K).squeeze(1) + 1e-8)

        if torch.norm(U - U_prev, p=1) < tol:  # type: ignore
            break

    # Compute dual potentials
    F = eps * torch.log(U + 1e-8)
    G = eps * torch.log(V + 1e-8)

    return F, G


def batched_sinkhorn_loss(
    logits_student: torch.Tensor,
    logits_teacher: torch.Tensor,
    ground_cost: torch.Tensor,
    eps: float,
    temperature: float = 1.0,
    max_iter: int = 1000,
    tol: float = 1e-6
) -> torch.Tensor:
    probs_student = torch.nn.functional.softmax(
        logits_student / temperature, dim=-1)
    probs_teacher = torch.nn.functional.softmax(
        logits_teacher / temperature, dim=-1)
    with torch.no_grad():
        F, G = batched_sinkhorn(
            probs_student, probs_teacher, ground_cost, eps, max_iter, tol)
    return (F * probs_student).sum() + (G * probs_teacher).sum()


def sqeuclidean_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.cdist(x, y, p=2) ** 2


def compute_gaussian(x: torch.Tensor) -> t.Tuple[torch.Tensor, torch.Tensor]:
    return x.mean(dim=0), x.std(dim=0, correction=0)


def __gaussian_kl(
    mean_a: torch.Tensor,
    mean_b: torch.Tensor,
    std_a: torch.Tensor,
    std_b: torch.Tensor
) -> torch.Tensor:
    d = mean_a.shape[0]
    return 0.5 * (
        ((std_a / (std_b + 1e-9)) ** 2).sum() +
        (((mean_a - mean_b) / (std_b + 1e-9)) ** 2).sum() -
        d + 2 * torch.log(std_b / (std_a + 1e-9)).sum()
    )


def __gaussian_w2(
    mean_a: torch.Tensor,
    mean_b: torch.Tensor,
    std_a: torch.Tensor,
    std_b: torch.Tensor
) -> torch.Tensor:
    return (
        torch.linalg.norm(mean_a - mean_b) ** 2 +
        torch.linalg.norm(std_a - std_b) ** 2
    )


def mmd_linear_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute MMD using a linear kernel.

    Args:
        x (torch.Tensor): Samples from distribution P,
        shape (n_samples, n_features).
        y (torch.Tensor): Samples from distribution Q,
        shape (n_samples, n_features).

    Returns:
        torch.Tensor: The MMD value.
    """
    n = x.size(0)
    m = y.size(0)

    # Compute the linear kernel for x and y
    xx = torch.mm(x, x.t())  # (n, n)
    yy = torch.mm(y, y.t())  # (m, m)
    xy = torch.mm(x, y.t())  # (n, m)

    # Compute the MMD
    mmd = (
        (xx.sum() / (n * n)) +
        (yy.sum() / (m * m)) -
        (2 * xy.sum() / (n * m))
    )
    return mmd


def mmd_rbf_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute MMD using a Gaussian (RBF) kernel.

    Args:
        x (torch.Tensor): Samples from distribution P, shape
        (n_samples, n_features).
        y (torch.Tensor): Samples from distribution Q, shape
        (n_samples, n_features).
        sigma (float): The bandwidth parameter for the RBF kernel.

    Returns:
        torch.Tensor: The MMD value.
    """
    n = x.size(0)
    m = y.size(0)

    # Compute pairwise distances
    xx = torch.cdist(x, x, p=2)  # (n, n)
    yy = torch.cdist(y, y, p=2)  # (m, m)
    xy = torch.cdist(x, y, p=2)  # (n, m)

    # Compute the RBF kernel
    with torch.no_grad():
        sigma = torch.cat(
            [xx.flatten(), yy.flatten(), xy.flatten()],
            dim=0
        ).mean()

    gamma = 1.0 / (2 * sigma ** 2)
    k_xx = torch.exp(-gamma * xx ** 2)
    k_yy = torch.exp(-gamma * yy ** 2)
    k_xy = torch.exp(-gamma * xy ** 2)

    # Compute the MMD
    mmd = (
        (k_xx.sum() / (n * n)) +
        (k_yy.sum() / (m * m)) -
        (2 * k_xy.sum() / (n * m))
    )
    return mmd


def empirical_w2(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    n, m = x.shape[0], y.shape[0]
    C = torch.cdist(x, y, p=2) ** 2
    with torch.no_grad():
        a = torch.ones(n) / n
        b = torch.ones(m) / m

        gamma = ot.emd(a, b, C)
        if type(gamma) is torch.Tensor:
            gamma = gamma.to(x.device)
        else:
            gamma = torch.from_numpy(gamma).to(x.device).to(x.dtype)
    return (gamma * C).sum()


def empirical_conditional_w2(
    x: torch.Tensor,
    y: torch.Tensor,
    labels: torch.Tensor
) -> torch.Tensor:
    unique_labels = torch.unique(labels)

    loss = torch.Tensor(0.0, device=x.device)
    for label in unique_labels:
        ind = torch.where(labels == label)[0]

        if len(ind) > 0:
            loss += empirical_w2(x[ind], y[ind])
    return loss


def compute_geodesic_distances(
    X: torch.Tensor,
    sigma: float = 1.0,
    t: float = 1.0
) -> torch.Tensor:
    # Step 1: Pairwise squared Euclidean distances
    X_norm = (X ** 2).sum(dim=1, keepdim=True)
    D2 = X_norm + X_norm.T - 2 * X @ X.T  # shape (n, n)

    # Step 2: Compute heat kernel (fully connected affinity matrix)
    A = torch.exp(-D2 / (2 * sigma**2))  # differentiable affinity

    # Step 3: Normalize to get transition matrix
    D = A.sum(dim=1)  # Degree matrix (diagonal)
    P = A / D[:, None]  # Row-normalized transition matrix

    # Step 4: Diffusion step (t-step random walk)
    P_t = torch.matrix_power(P, int(t))

    # Step 5: Compute diffusion distances
    # Use L2 distance between rows of P^t
    row_norm = (P_t**2).sum(dim=1, keepdim=True)
    dist2 = row_norm + row_norm.T - 2 * P_t @ P_t.T
    dist2 = torch.clamp(dist2, min=0.0)  # numerical stability

    return dist2.sqrt()  # Final diffusion distance matrix


def empirical_gromov_wasserstein(
    x: torch.Tensor,
    y: torch.Tensor,
    dist: t.Literal[
        'euclidean',
        'sqeuclidean',
        'cosine',
        'geodesic'
    ] = 'euclidean'
) -> torch.Tensor:
    n, m = x.shape[0], y.shape[0]

    a = (torch.ones(n) / n).to(x.device)
    b = (torch.ones(m) / m).to(x.device)

    if dist == 'euclidean':
        C_x = torch.cdist(x, x, p=2)
        C_y = torch.cdist(y, y, p=2)
    elif dist == 'sqeuclidean':
        C_x = torch.cdist(x, x, p=2) ** 2
        C_y = torch.cdist(y, y, p=2) ** 2
    elif dist == 'cosine':
        C_x = 1 - torch.nn.functional.cosine_similarity(
            x.unsqueeze(1), x.unsqueeze(0), dim=-1)
        C_y = 1 - torch.nn.functional.cosine_similarity(
            y.unsqueeze(1), y.unsqueeze(0), dim=-1)
    elif dist == 'geodesic':
        C_x = compute_geodesic_distances(x, sigma=1.0, t=1.0)
        C_y = compute_geodesic_distances(y, sigma=1.0, t=1.0)
    else:
        raise ValueError(f"Unknown distance metric: {dist}")

    with torch.no_grad():
        gamma = ot.gromov.gromov_wasserstein(
            C1=C_x,
            C2=C_y,
            p=a,
            q=b,
            loss_fun='square_loss',
            verbose=False,
            tol=1e-5
        )

        if gamma is torch.Tensor:
            gamma = gamma.to(x.device)  # type: ignore
        else:
            gamma = torch.from_numpy(gamma).to(x.device).to(x.dtype)
    constC, hX, hY = ot.gromov._utils.init_matrix(
        C_x, C_y, a, b, loss_fun='square_loss')
    return ot.gromov._utils.gwloss(constC, hX, hY, gamma)


def gaussian_w2(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    mean_x, std_x = compute_gaussian(x)
    mean_y, std_y = compute_gaussian(y)

    return __gaussian_w2(mean_x, mean_y, std_x, std_y)


def gaussian_conditional_w2(
    x: torch.Tensor,
    y: torch.Tensor,
    labels: torch.Tensor
) -> torch.Tensor:
    unique_labels = torch.unique(labels)

    loss = torch.Tensor(0.0, device=x.device)
    for label in unique_labels:
        ind = torch.where(labels == label)[0]

        if len(ind) > 0:
            loss += gaussian_w2(x[ind], y[ind])
    return loss


def gaussian_kl(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    mean_x, std_x = compute_gaussian(x)
    mean_y, std_y = compute_gaussian(y)

    return __gaussian_kl(mean_x, mean_y, std_x, std_y)


def gaussian_conditional_kl(
    x: torch.Tensor,
    y: torch.Tensor,
    labels: torch.Tensor
) -> torch.Tensor:
    unique_labels = torch.unique(labels)

    loss = torch.Tensor(0.0, device=x.device)
    for label in unique_labels:
        ind = torch.where(labels == label)[0]

        if len(ind) > 0:
            loss += gaussian_kl(x[ind], y[ind])
    return loss


def pairwise_kl_div_logits(logits_student, logits_teacher):
    """
    Computes an (n, n) matrix where entry (i, j) corresponds to
    KL divergence:

    KL(softmax(logits_student)[i, :] || softmax(logits_teacher)[j, :]).

    Args:
        logits_student: Tensor of shape (n, n_c), raw scores before
        softmax from the student model
        logits_teacher: Tensor of shape (n, n_c), raw scores before
        softmax from the teacher model

    Returns:
        kl_matrix: Tensor of shape (n, n) containing pairwise KL divergences.
    """
    # Compute log probabilities for stability
    log_probs_student = torch.nn.functional.log_softmax(
        logits_student, dim=-1)  # (n, n_c)
    log_probs_teacher = torch.nn.functional.log_softmax(
        logits_teacher, dim=-1)  # (n, n_c)

    # Compute probability distributions
    probs_student = torch.exp(log_probs_student)  # (n, n_c)

    # Expand tensors for pairwise computation
    probs_student = probs_student.unsqueeze(1)  # (n, 1, n_c)
    log_probs_teacher = log_probs_teacher.unsqueeze(0)  # (1, n, n_c)

    # Compute KL divergence: sum(p_i * (log p_i - log p_j))
    kl_matrix = torch.sum(
        probs_student * (
            log_probs_student.unsqueeze(1) - log_probs_teacher
        ), dim=-1
    )  # (n, n)

    return kl_matrix


class GromovWassersteinDistance(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        features_student: torch.Tensor,
        features_teacher: torch.Tensor,
        labels: torch.Tensor,
        logits_student: t.Optional[torch.Tensor] = None,
        logits_teacher: t.Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return empirical_gromov_wasserstein(features_student, features_teacher)


class WassersteinDistance(torch.nn.Module):
    def __init__(
        self,
        ground_metric_fn=None,
        measure_modeling: t.Literal[
            'empirical',
            'gaussian',
            'cond_gaussian',
            'cond_empirical'
        ] = 'empirical'
    ):
        super().__init__()

        if not ground_metric_fn:
            self.ground_metric_fn = sqeuclidean_distance
        else:
            self.ground_metric_fn = ground_metric_fn

        self.measure_modeling = measure_modeling

    def forward(
        self,
        features_student: torch.Tensor,
        features_teacher: torch.Tensor,
        labels: torch.Tensor,
        logits_student: t.Optional[torch.Tensor] = None,
        logits_teacher: t.Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if self.measure_modeling == 'gaussian':
            return gaussian_w2(
                features_student,
                features_teacher)
        elif self.measure_modeling == 'empirical':
            return empirical_w2(
                features_student,
                features_teacher)
        elif self.measure_modeling == "cond_gaussian":
            return gaussian_conditional_w2(
                features_student,
                features_teacher,
                labels
            )
        elif self.measure_modeling == "cond_empirical":
            return empirical_conditional_w2(
                features_student,
                features_teacher,
                labels
            )
        else:
            raise ValueError(
                f"Unknown measure modeling: {self.measure_modeling}"
            )


class JointWassersteinDistance(torch.nn.Module):
    def __init__(
        self,
        label_metric: t.Literal['sqeuclidean', 'cce'] = 'sqeuclidean'
    ):
        super().__init__()
        self.label_metric = label_metric

    def forward(
        self,
        features_student: torch.Tensor,
        features_teacher: torch.Tensor,
        labels: torch.Tensor,
        logits_student: t.Optional[torch.Tensor] = None,
        logits_teacher: t.Optional[torch.Tensor] = None
    ):
        Cx = torch.cdist(features_student, features_student, p=2) ** 2

        if logits_student is not None and logits_teacher is not None:
            if self.label_metric == 'sqeuclidean':
                Cy = torch.cdist(logits_student, logits_teacher, p=2) ** 2
            elif self.label_metric == 'cce':
                Cy = pairwise_kl_div_logits(logits_student, logits_teacher)
            else:
                raise ValueError(f"Unknown label metric: {self.label_metric}")
        else:
            Cy = torch.zeros_like(Cx)

        with torch.no_grad():
            Cx = Cx / Cx.max()
        C = Cx + Cy

        n, m = features_student.shape[0], features_teacher.shape[0]
        with torch.no_grad():
            a = torch.ones(n) / n
            b = torch.ones(m) / m

            gamma = ot.emd(a, b, C)
            if type(gamma) is torch.Tensor:
                gamma = gamma.to(features_student.device)
            else:
                gamma = torch.from_numpy(
                    gamma).to(
                        features_student.device).to(
                            features_student.dtype)
        return (gamma * C).sum()


class KullbackLeiblerDivergence(torch.nn.Module):
    def __init__(self, conditional=False):
        self.conditional = conditional
        super().__init__()

    def forward(
        self,
        features_student: torch.Tensor,
        features_teacher: torch.Tensor,
        labels: torch.Tensor,
        logits_student: t.Optional[torch.Tensor] = None,
        logits_teacher: t.Optional[torch.Tensor] = None
    ):
        if self.conditional:
            return gaussian_conditional_kl(
                features_student, features_teacher, labels)
        else:
            return gaussian_kl(features_student, features_teacher)


class MaximumMeanDiscrepancy(torch.nn.Module):
    def __init__(
        self,
        conditional: bool = False,
        kernel: t.Literal['linear', 'rbf'] = 'linear',
        sigma=None
    ):
        self.conditional = conditional

        if kernel == 'linear':
            self.mmd_fn = mmd_linear_kernel
        elif kernel == 'rbf':
            self.mmd_fn = mmd_rbf_kernel
        else:
            raise ValueError(f"Invalid {kernel=}")

        super().__init__()

    def __conditional_forward(
        self,
        features_student: torch.Tensor,
        features_teacher: torch.Tensor,
        labels: torch.Tensor
    ):
        unique_labels = torch.unique(labels)

        loss = 0.0
        for label in unique_labels:
            ind = torch.where(labels == label)[0]

            if len(ind) > 0:
                loss += self.mmd_fn(
                    features_student[ind], features_teacher[ind])
        return loss

    def forward(
        self,
        features_student: torch.Tensor,
        features_teacher: torch.Tensor,
        labels: torch.Tensor,
        logits_student: t.Optional[torch.Tensor] = None,
        logits_teacher: t.Optional[torch.Tensor] = None
    ):
        if self.conditional:
            return self.__conditional_forward(
                features_student, features_teacher, labels)
        return self.mmd_fn(features_student, features_teacher)
