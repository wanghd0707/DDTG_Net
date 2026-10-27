"""Feature builders matching the 14-D facet input reported in the paper."""

from __future__ import annotations

import torch
import torch.nn.functional as F


FACE_FEATURE_DIM = 14
FACE_FEATURE_LAYOUT = (
    "relative_position_x",
    "relative_position_y",
    "relative_position_z",
    "normal_x",
    "normal_y",
    "normal_z",
    "face_area",
    "incident_electric_real_x",
    "incident_electric_real_y",
    "incident_electric_real_z",
    "incident_electric_imag_x",
    "incident_electric_imag_y",
    "incident_electric_imag_z",
    "wavenumber",
)


def build_paper_face_features(
    centroids: torch.Tensor,
    normals: torch.Tensor,
    face_area: torch.Tensor,
    face_patch_id: torch.Tensor,
    patch_centers: torch.Tensor,
    incident_electric_field: torch.Tensor,
    wavenumber: float | torch.Tensor,
) -> torch.Tensor:
    """Construct the paper's 14-D facet feature vector.

    The returned columns are
    ``[r_i-c_p(i), n_i, A_i, Re(E_inc_i), Im(E_inc_i), k]``.
    ``incident_electric_field`` must be a complex tensor of shape ``[N, 3]``.
    ``wavenumber`` may be a scalar or one scalar per facet.
    """

    centroids = centroids.float()
    normals = F.normalize(normals.float(), dim=-1)
    face_area = face_area.float().reshape(-1, 1)
    face_patch_id = face_patch_id.long()
    patch_centers = patch_centers.float()

    if centroids.ndim != 2 or centroids.shape[1] != 3:
        raise ValueError("centroids must have shape [N, 3]")
    if normals.shape != centroids.shape:
        raise ValueError("normals must have shape [N, 3]")
    if face_area.shape != (centroids.shape[0], 1):
        raise ValueError("face_area must contain one value per facet")
    if face_patch_id.shape != (centroids.shape[0],):
        raise ValueError("face_patch_id must have shape [N]")
    if patch_centers.ndim != 2 or patch_centers.shape[1] != 3:
        raise ValueError("patch_centers must have shape [P, 3]")
    if not torch.is_complex(incident_electric_field):
        raise TypeError("incident_electric_field must be a complex tensor")
    if incident_electric_field.shape != centroids.shape:
        raise ValueError("incident_electric_field must have shape [N, 3]")
    if face_patch_id.numel() and (
        int(face_patch_id.min()) < 0
        or int(face_patch_id.max()) >= patch_centers.shape[0]
    ):
        raise ValueError("face_patch_id contains an invalid patch index")

    relative_position = centroids - patch_centers[face_patch_id]
    electric_real = incident_electric_field.real.float()
    electric_imag = incident_electric_field.imag.float()

    k = torch.as_tensor(
        wavenumber, dtype=centroids.dtype, device=centroids.device
    )
    if k.ndim == 0:
        k = k.expand(centroids.shape[0])
    else:
        k = k.reshape(-1)
        if k.numel() != centroids.shape[0]:
            raise ValueError("wavenumber must be scalar or contain one value per facet")

    features = torch.cat(
        [
            relative_position,
            normals,
            face_area,
            electric_real,
            electric_imag,
            k.unsqueeze(-1),
        ],
        dim=-1,
    )
    if features.shape[1] != FACE_FEATURE_DIM:
        raise RuntimeError(f"expected 14 features, got {features.shape[1]}")
    return features


def build_paper_edge_features(
    centroids: torch.Tensor,
    normals: torch.Tensor,
    edge_index: torch.Tensor,
) -> torch.Tensor:
    """Construct the paper's 5-D local facet-edge features.

    The returned columns are ``[r_j-r_i, ||r_j-r_i||, n_i dot n_j]``.
    """

    centroids = centroids.float()
    normals = F.normalize(normals.float(), dim=-1)
    edge_index = edge_index.long()
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, E]")
    source, target = edge_index
    displacement = centroids[target] - centroids[source]
    distance = torch.linalg.vector_norm(displacement, dim=-1, keepdim=True)
    normal_alignment = (normals[source] * normals[target]).sum(
        dim=-1, keepdim=True
    )
    return torch.cat([displacement, distance, normal_alignment], dim=-1)


__all__ = [
    "FACE_FEATURE_DIM",
    "FACE_FEATURE_LAYOUT",
    "build_paper_face_features",
    "build_paper_edge_features",
]
