import torch
import lightning
import numpy as np
import matplotlib.pyplot as plt
from mlcolvar.cvs.committor.utils import initialize_committor_masses
import copy
from mlcolvar.data import DictModule, DictDataset
from mlcolvar.utils.trainer import MetricsCallback
from mlcolvar.utils.plot import plot_metrics
from lightning.pytorch.callbacks import EarlyStopping

# GNN utils
def unsorted_segment_sum(
    data: torch.Tensor, segment_ids: torch.Tensor, num_segments: int
) -> torch.Tensor:
    """Function that sums the segments of a matrix. Each row has a non-unique ID and all rows with the same ID are summed such that a matrix with the number of rows equal to the number of unique IDs is obtained.

    :param data: A tensor that contains the data that is to be summed.
    :type data: torch.tensor
    :param segment_ids: An array that has the same number of entries as data has rows which indicates which rows shall be summed.
    :type segment_ids: torch.tensor
    :param num_segments: This is the number of unique IDs, i.e. the dimensionality of the resulting tensor.
    :type num_segments: int
    :return: Returns a tensor shaped num_segments x data.size(1) containing all the segment sums.
    :rtype: torch.Tensor
    """
    result_shape = (num_segments, data.size(1))
    result = data.new_zeros(result_shape)  # Init empty result tensor.
    segment_ids_exp = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    # Use non-inplace scatter_add (returns new tensor, better for 2nd derivatives)
    result = result.scatter_add(0, segment_ids_exp, data)
    return result

def soft_weighted_unsorted_segment_sum(
    data: torch.Tensor, segment_ids: torch.Tensor, seg_weights: torch.Tensor, num_segments: int
) -> torch.Tensor:
    """Function that sums the segments of a matrix. Each row has a non-unique ID and all rows with the same ID are summed such that a matrix with the number of rows equal to the number of unique IDs is obtained.

    This version is numerically stable for second derivatives (Hessian computation).

    :param data: A tensor that contains the data that is to be summed.
    :type data: torch.tensor
    :param segment_ids: An array that has the same number of entries as data has rows which indicates which rows shall be summed.
    :type segment_ids: torch.tensor
    :param num_segments: This is the number of unique IDs, i.e. the dimensionality of the resulting tensor.
    :type num_segments: int
    :return: Returns a tensor shaped num_segments x data.size(1) containing all the segment sums.
    :rtype: torch.Tensor
    """
    # Numerical stabilisation for 2nd derivatives:
    # 1. Clamp logits to prevent exp overflow
    # 2. Use softmax with numerical stability (subtract max per segment)
    # 3. Add small epsilon to denominator instead of using torch.where
    
    sw = seg_weights.squeeze(-1)  # (E,)
    eps = 1e-8  # Small epsilon for numerical stability
    
    # Clamp to prevent exp overflow
    sw = torch.clamp(sw, min=-50.0, max=50.0)
    
    # For numerical stability in softmax, subtract max per segment
    # But this adds complexity for Hessian. Instead, just use stable exp with clamping.
    exp_weights = torch.exp(sw)  # (E,)

    # Sum of exp weights per segment for normalisation
    # Use zeros_like to maintain gradient flow
    w_sum = data.new_zeros((num_segments,))
    if segment_ids.numel() > 0:
        w_sum = w_sum.scatter_add(0, segment_ids, exp_weights)
    
    # CRITICAL: Use smooth regularization instead of torch.where
    # torch.where creates discontinuities that break 2nd derivatives
    # Adding eps ensures we never divide by exactly zero
    w_sum_safe = w_sum + eps  # Now always > 0

    # Aggregate weighted messages  
    result = data.new_zeros((num_segments, data.size(1)))
    if segment_ids.numel() > 0:
        # Normalize weights using safe denominator
        norm_weights = exp_weights / w_sum_safe[segment_ids]  # (E,)
        
        # Weighted aggregation using non-inplace scatter_add
        segment_ids_exp = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
        result = result.scatter_add(0, segment_ids_exp, data * norm_weights.unsqueeze(-1))
    
    return result

from typing import List, Optional, Tuple, Union

import torch
from torch import nn
from mlcolvar.core.transform import Transform
from mlcolvar.core.transform.descriptors.utils import (
    sanitize_positions_shape,
    sanitize_cell_shape,
)


def _apply_pbc_distances(dist_components, pbc_cell):
    """Apply PBC corrections to distance components.
    
    Uses device-aware operations to avoid TorchScript device mismatches.
    All scalar constants are created as tensors on the same device as inputs.
    """
    device = dist_components.device
    dtype = dist_components.dtype
    
    # Ensure pbc_cell is on same device/dtype
    pbc_cell = pbc_cell.to(device=device, dtype=dtype)
    
    # Create scalar constants on correct device (critical for TorchScript)
    one = torch.tensor(1.0, device=device, dtype=dtype)
    two = torch.tensor(2.0, device=device, dtype=dtype)
    
    # Compute PBC shifts using device-aware operations
    # For each dimension d: shift = round(dist / L) * L
    # This uses: round(x) = floor(x + 0.5) = trunc(x + sign(x)*0.5)
    
    # Get half cell lengths
    half_cell = pbc_cell / two
    
    # Compute shifts for each dimension
    # shifts = trunc(dist / (L/2)) then adjust
    # final_shift = trunc((shifts + sign(shifts)) / 2) * L
    
    # Extract cell lengths for broadcasting (reshape to [1, 3, 1, 1] for batched ops)
    Lx = pbc_cell[0].reshape(1, 1, 1)
    Ly = pbc_cell[1].reshape(1, 1, 1)  
    Lz = pbc_cell[2].reshape(1, 1, 1)
    half_Lx = Lx / two
    half_Ly = Ly / two
    half_Lz = Lz / two
    
    # dist_components shape: (B, 3, N, N)
    # Apply PBC separately for each dimension
    dx = dist_components[:, 0:1, :, :]
    dy = dist_components[:, 1:2, :, :]
    dz = dist_components[:, 2:3, :, :]
    
    # PBC correction: d_wrapped = d - round(d/L)*L
    # where round(x) = trunc(x + sign(x)*0.5)
    def wrap_dim(d, L, half_L):
        # Compute number of cell crossings
        n = torch.div(d, half_L, rounding_mode='trunc')
        # Adjust for proper rounding
        n = torch.div(n + torch.sign(n) * one, two, rounding_mode='trunc')
        # Compute shift
        return d - n * L
    
    dx_wrapped = wrap_dim(dx, Lx, half_Lx)
    dy_wrapped = wrap_dim(dy, Ly, half_Ly)
    dz_wrapped = wrap_dim(dz, Lz, half_Lz)
    
    # Concatenate back
    dist_components = torch.cat([dx_wrapped, dy_wrapped, dz_wrapped], dim=1)
    
    return dist_components


def _sanitize_cell_local(cell, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Device-aware cell sanitization that keeps tensor on correct device.
    
    This replaces mlcolvar's sanitize_cell_shape for TorchScript compatibility.
    Creates cell tensor directly on the target device to avoid device mismatches.
    
    Parameters
    ----------
    cell : float, list, or torch.Tensor
        Cell specification in various formats
    device : torch.device
        Target device
    dtype : torch.dtype
        Target dtype
        
    Returns
    -------
    torch.Tensor
        Cell as shape (3,) [Lx, Ly, Lz] on target device
    """
    # Handle tensor input
    if isinstance(cell, torch.Tensor):
        cell = cell.to(device=device, dtype=dtype)
        
        # Flatten to 1D if needed for consistent handling
        if cell.dim() == 0:
            # Scalar -> cubic
            return cell.expand(3).clone()
        
        cell_flat = cell.flatten()
        numel = cell_flat.numel()
        
        if numel == 1:
            return cell_flat.expand(3).clone()
        elif numel == 3:
            return cell_flat
        elif numel == 6:
            # LAMMPS bounds
            Lx = cell_flat[1] - cell_flat[0]
            Ly = cell_flat[3] - cell_flat[2]
            Lz = cell_flat[5] - cell_flat[4]
            return torch.stack([Lx, Ly, Lz])
        elif numel == 9:
            # 3x3 matrix -> diagonal
            mat = cell_flat.view(3, 3)
            return torch.diag(mat)
        else:
            raise ValueError(f"Unsupported cell tensor with {numel} elements")
    
    # Handle scalar
    elif isinstance(cell, (int, float)):
        return torch.tensor([cell, cell, cell], device=device, dtype=dtype)
    
    # Handle list/tuple
    elif isinstance(cell, (list, tuple)):
        cell_t = torch.tensor(cell, device=device, dtype=dtype)
        return _sanitize_cell_local(cell_t, device, dtype)
    
    else:
        raise ValueError(f"Unsupported cell type: {type(cell)}")


def compute_distances_matrix_safe(pos: torch.Tensor,
                                   n_atoms: int,
                                   PBC: bool,
                                   cell: Union[float, list],
                                   vector: bool = False,
                                   scaled_coords: bool = False,
                                   eps: float = 1e-8,
                                  ) -> torch.Tensor:
    """Compute pairwise distances matrix with numerical stability for 2nd derivatives.
    
    This is a modified version of mlcolvar's compute_distances_matrix that adds
    epsilon regularization inside sqrt to prevent NaN in second derivatives when
    atoms are very close together.
    
    The issue: d²sqrt(r²)/dx² ~ 1/r³ which explodes as r→0
    The fix: sqrt(r² + eps) has bounded second derivatives
    
    Parameters
    ----------
    pos : torch.Tensor
        Positions shape (batch, n_atoms, 3) or (batch, n_atoms*3)
    n_atoms : int
        Number of atoms
    PBC : bool
        Use periodic boundary conditions
    cell : Union[float, list]
        Cell dimensions
    vector : bool
        Return vector distances instead of scalar
    scaled_coords : bool
        Coordinates are scaled to [0,1]
    eps : float
        Small value to add inside sqrt for numerical stability (default 1e-8)
        
    Returns
    -------
    torch.Tensor
        Distance matrix (batch, n_atoms, n_atoms)
    """
    pos, batch_size = sanitize_positions_shape(pos, n_atoms)

    _device = pos.device
    _dtype = pos.dtype
    
    # Use local cell sanitization to avoid device mismatches
    # This creates tensors directly on the correct device
    cell = _sanitize_cell_local(cell, _device, _dtype)

    if scaled_coords:
        pbc_cell = torch.tensor([1., 1., 1.], device=_device, dtype=_dtype)
    else:
        pbc_cell = cell
    
    pos = torch.reshape(pos, (batch_size, n_atoms, 3))
    pos = torch.transpose(pos, 1, 2)
    pos = pos.reshape((batch_size, 3, n_atoms))

    pos_expanded = torch.tile(pos, (1, 1, n_atoms)).reshape(batch_size, 3, n_atoms, n_atoms)
    dist_components = pos_expanded - torch.transpose(pos_expanded, -2, -1)

    if PBC:
        dist_components = _apply_pbc_distances(dist_components=dist_components, pbc_cell=pbc_cell)

    if scaled_coords:
        dist_components = torch.einsum('bijk,i->bijk', dist_components, cell)

    if vector: 
        return dist_components
    else:
        # Sum squared components
        dist_sq = torch.sum(torch.pow(dist_components, 2), 1)  # (batch, n_atoms, n_atoms)
        
        # CRITICAL FIX: Add epsilon INSIDE sqrt to prevent NaN in 2nd derivatives
        # sqrt(r² + eps) has bounded 2nd derivatives even when r→0
        # For diagonal (self-distance), we set to 0 after
        dist = torch.sqrt(dist_sq + eps)
        
        # Zero out diagonal (self-distances should be exactly 0, not sqrt(eps))
        diag_mask = torch.eye(n_atoms, dtype=torch.bool, device=_device).unsqueeze(0).expand(batch_size, -1, -1)
        dist = dist.masked_fill(diag_mask, 0.0)
        
        return dist


__all__ = ["GNNTransformerDescriptor"]
# Graph Transformer Convolutional layer
class TransGCL(nn.Module):
    def __init__(self, hidden_nf: int, n_heads: int, act_fn=nn.ReLU()):
        """Defines the Graph convolutional layer for graph-based models. Do not instantiate directly.

        Parameters
        ----------
        hidden_nf : int
            Hidden dimensionality of the latent node representation.
        n_heads : int
            Number of Attention heads, 
        act_fn : torch.nn.modules.activation, optional
            PyTorch activation function to be used in the multi-layer perceptrons, by default nn.ReLU()
        """
        super(TransGCL, self).__init__()
        self.n_heads = n_heads
        self.act_fn = act_fn
        self.dropout = nn.Dropout(p=0.1)
        for i in range(n_heads):
            # Incorporate source-target relation in message (target, target - source)
            self.add_module(
                f"edge_{i}",
                nn.Sequential(
                    nn.Linear(hidden_nf * 2, hidden_nf),
                    act_fn,
                    nn.Linear(hidden_nf, hidden_nf),
                ),
            )
        for i in range(n_heads):
            att_block = nn.Sequential(
                nn.Linear(hidden_nf * 2, hidden_nf),
                act_fn,
                nn.Linear(hidden_nf, 1),
            )
            # Small init to keep logits tame
            nn.init.uniform_(att_block[-1].weight, a=-1e-3, b=1e-3)
            nn.init.zeros_(att_block[-1].bias)
            self.add_module(f"attention_{i}", att_block)

        concat_dim = hidden_nf * (n_heads + 1)
        self.pre_norm = nn.LayerNorm(concat_dim)
        self.node_mlp = nn.Sequential(
            nn.Linear(concat_dim, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
        )
        self.residual = True
        
        # Store modules in ModuleLists for TorchScript compatibility
        self.edge_modules = nn.ModuleList([self._modules[f"edge_{i}"] for i in range(n_heads)])
        self.attention_modules = nn.ModuleList([self._modules[f"attention_{i}"] for i in range(n_heads)])

    def edge_model(self, source: torch.Tensor, target: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        outs: List[torch.Tensor] = []
        att_logits: List[torch.Tensor] = []
        rel = target - source
        cat_src_tgt = torch.cat([source, target], dim=1)
        cat_tgt_rel = torch.cat([target, rel], dim=1)
        for i in range(self.n_heads):
            outs.append(self.edge_modules[i](cat_tgt_rel))
            att_logits.append(self.attention_modules[i](cat_src_tgt))
        return outs, att_logits  # unnormalized logits

    def node_model(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: List[torch.Tensor], att_logits: List[torch.Tensor]) -> torch.Tensor:
        row, _ = edge_index[0], edge_index[1]
        agg_parts: List[torch.Tensor] = [x]
        for i in range(self.n_heads):
            weighted = soft_weighted_unsorted_segment_sum(
                edge_attr[i], row, att_logits[i], num_segments=x.size(0)
            )
            # Scale to control variance across heads
            weighted = weighted / (self.n_heads ** 0.5)
            agg_parts.append(weighted)
        agg = torch.cat(agg_parts, dim=1)
        agg = self.pre_norm(agg)
        out = self.node_mlp(self.dropout(agg))
        if self.residual and out.shape[0] == x.shape[0] and out.shape[1] == x.shape[1]:
            out = out + x
        return out

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        row, col = edge_index[0], edge_index[1]
        if row.numel() == 0:
            # No edges, just return input unchanged
            return h
        edge_feat, att_logits = self.edge_model(h[row], h[col])
        h_out = self.node_model(h, edge_index, edge_feat, att_logits)
        return h_out

# Graph model
class GCL(nn.Module):
    """The graph convolutional layer for the graph-based model. Do not instantiate this directly.

    :param hidden_nf: Hidden dimensionality of the latent node representation.
    :type hidden_nf: int
    :param act_fn: PyTorch activation function to be used in the multi-layer perceptrons, defaults to nn.ReLU()
    :type act_fn: torch.nn.modules.activation, optional
    """

    def __init__(self, hidden_nf: int, act_fn=nn.ReLU()):
        super(GCL, self).__init__()

        self.edge_mlp = nn.Sequential(
            # Only takes the neighbourhood node
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
            # Maps to the same dimension
            nn.Linear(hidden_nf, hidden_nf),
        )

        self.node_mlp = nn.Sequential(
            # Node MLP just takes the current vector and the resulting neighbourhood vector
            nn.Linear(hidden_nf * 2, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
        )

        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

    def edge_model(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

        out = torch.cat([source - target], dim=1)
        out = self.edge_mlp(out)
        return out

    def node_model(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        row = edge_index[0]
        # Get the summed edge vectors for each node
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))
        agg = torch.cat([x, agg], dim=1)
        out = self.node_mlp(agg)

        return out

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        row, col = edge_index[0], edge_index[1]

        edge_feat = self.edge_model(h[row], h[col])
        h = self.node_model(h, edge_index, edge_feat)
        return h

class GNNTransformerDescriptor(Transform):
    """Graph Transformer descriptor usable as a preprocessing step.

    Two modes:
    - mode="graph": outputs a graph-level vector of size out_features per frame.
    - mode="node": outputs node-level predictions for every atom, flattened to (B, N*out_features).

    This is trainable with Lightning (it is a torch.nn.Module). After training, set to eval() and
    use it as a descriptor feeding a downstream CV model.
    """

    def __init__(
        self,
        n_atoms: int,
        out_features: int = 1,
        in_node_nf: int = 3,
        hidden_nf: int = 64,
        n_layers: int = 2,
        n_heads: int = 1,
        PBC: bool = True,
        cell: Union[float, List[float]] = 1.0,
        cutoff: float = 1.0,
        pool: str = "sum",
        mode: str = "graph",  # "graph" or "node"
        device: Optional[Union[str, torch.device]] = None,
    ):
        """Initialize the GNN Transformer descriptor.

        Parameters
        ----------
        n_atoms : int
            Number of atoms in the system.
        out_features : int
            Output features per graph (graph mode) or per node (node mode; flattened in the descriptor).
        in_node_nf : int
            Input node feature dimension (defaults to 3: xyz coordinates).
        hidden_nf : int
            Hidden dimension.
        n_layers : int
            Number of transformer convolutional layers.
        n_heads : int
            Number of attention heads per layer. Setting to 0 deactivates attention and aggregates node features with a constant edge weight of 1.
        PBC : bool
            Whether to use periodic boundary conditions in distance/edge building.
        cell : float | List[float]
            Cell dimensions for PBC handling (orthorhombic).
        cutoff : float
            Distance cutoff to connect edges (in real units; applied on PBC-aware distances).
        pool : str
            Pooling on nodes for graph output: one of {"sum", "mean", "max"}.
        mode : str
            "graph" or "node". Graph returns (B, out_features); node returns (B, n_atoms*out_features).
        device : Optional device
            torch device to place parameters and computations.
        """
        self.n_atoms = int(n_atoms)
        self.mode = str(mode).lower()
        if self.mode not in ("graph", "node"):
            raise ValueError("mode must be either 'graph' or 'node'")

        # Determine descriptor out_features shape for base Transform
        desc_out = int(out_features) if self.mode == "graph" else int(n_atoms * out_features)
        super().__init__(in_features=int(n_atoms * 3), out_features=desc_out)

        self.out_features = int(out_features)
        self.in_node_nf = int(in_node_nf)
        self.hidden_nf = int(hidden_nf)
        self.n_layers = int(n_layers)
        self.n_heads = int(n_heads)
        self.PBC = bool(PBC)
        # Register cell as buffer so it moves with .to(device)
        if isinstance(cell, torch.Tensor):
            self.register_buffer('cell', cell.clone())
        elif cell is not None:
            self.register_buffer('cell', torch.tensor(cell, dtype=torch.float32))
        else:
            self.register_buffer('cell', None)
        self.cutoff = float(cutoff)
        self.device = torch.device(device) if device is not None else None

        # Pooling function
        pool = pool.lower()
        if pool == "sum":
            self._pool_fn = torch.sum
        elif pool == "mean":
            self._pool_fn = torch.mean
        elif pool == "max":
            self._pool_fn = torch.amax
        else:
            raise ValueError("pool must be one of {'sum','mean','max'}")

        # Encoder from node input features to hidden
        self.embedding = torch.nn.Linear(self.in_node_nf, self.hidden_nf)

        # Transformer conv layers
        if self.n_heads == 0:
            self.layers = torch.nn.ModuleList([GCL(self.hidden_nf, act_fn=torch.nn.ReLU()) for _ in range(self.n_layers)])
        else:
            self.layers = torch.nn.ModuleList([TransGCL(self.hidden_nf, self.n_heads, act_fn=torch.nn.ReLU()) for _ in range(self.n_layers)])

        # Node-level head (pre-pooling)
        self.node_head = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_nf, self.hidden_nf),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_nf, self.hidden_nf),
        )

        # Graph-level head (post-pooling)
        self.graph_head = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_nf, self.hidden_nf),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_nf, self.out_features),
        )

        # Node-level prediction head (if mode=node)
        if self.mode == "node":
            self.node_pred = torch.nn.Linear(self.hidden_nf, self.out_features)
        else:
            self.node_pred = None

    # ---------------- helpers ----------------
    def _ensure_device_dtype(
        self, 
        pos: torch.Tensor, 
        cell_override: Optional[torch.Tensor] = None
    ) -> Tuple[torch.device, torch.dtype, torch.Tensor]:
        """Ensure device/dtype consistency and prepare cell matrix.
        
        Parameters
        ----------
        pos : torch.Tensor
            Input positions tensor (determines device and dtype)
        cell_override : torch.Tensor, optional
            Runtime cell for NPT simulations. If provided, uses this instead of self.cell.
            Can be:
            - (3,) or (1, 3): [Lx, Ly, Lz] orthorhombic box lengths
            - (6,) or (1, 6): [xlo, xhi, ylo, yhi, zlo, zhi] LAMMPS bounds
            - (9,) or (1, 9): Flattened 3x3 cell matrix
            - (3, 3) or (1, 3, 3): Full cell matrix
            - (B, 3), (B, 6), (B, 9), (B, 3, 3): Per-batch cells for NPT
            
        Returns
        -------
        Tuple[device, dtype, box]
            Device, dtype, and properly shaped cell tensor as [Lx, Ly, Lz]
        """
        # Use model's device (from embedding layer weights), not input device
        # This ensures inputs are moved to match the model, not vice versa
        try:
            device = next(self.parameters()).device
            dtype = next(self.parameters()).dtype
        except StopIteration:
            # Fallback to input device if no parameters
            device = pos.device
            dtype = pos.dtype
        B = pos.shape[0]
        
        if cell_override is not None:
            # Use runtime cell (NPT mode)
            if isinstance(cell_override, torch.Tensor):
                box = cell_override.to(device=device, dtype=dtype)
            else:
                box = torch.tensor(cell_override, device=device, dtype=dtype)
            
            # Convert various formats to [Lx, Ly, Lz] shape (3,) or (B, 3)
            box = self._convert_cell_to_lengths(box, B, device, dtype)
        else:
            # Use stored cell (NVT mode) - use local sanitization for device safety
            box = _sanitize_cell_local(self.cell, device, dtype)
        
        return device, dtype, box
    
    def _convert_cell_to_lengths(
        self, 
        cell: torch.Tensor, 
        B: int,
        device: torch.device,
        dtype: torch.dtype
    ) -> torch.Tensor:
        """Convert various cell formats to [Lx, Ly, Lz] box lengths.
        
        Parameters
        ----------
        cell : torch.Tensor
            Cell tensor in various formats
        B : int
            Batch size (for validation)
        device : torch.device
            Target device
        dtype : torch.dtype
            Target dtype
            
        Returns
        -------
        torch.Tensor
            Cell as [Lx, Ly, Lz] shape (3,) 
        """
        # Handle different input shapes
        if cell.dim() == 0:
            # Scalar -> cubic box
            return cell.expand(3)
        
        elif cell.dim() == 1:
            if cell.shape[0] == 1:
                # Single value -> cubic box
                return cell.expand(3)
            elif cell.shape[0] == 3:
                # Already [Lx, Ly, Lz]
                return cell
            elif cell.shape[0] == 6:
                # LAMMPS bounds: [xlo, xhi, ylo, yhi, zlo, zhi]
                Lx = cell[1] - cell[0]
                Ly = cell[3] - cell[2]
                Lz = cell[5] - cell[4]
                return torch.stack([Lx, Ly, Lz])
            elif cell.shape[0] == 9:
                # Flattened 3x3 -> extract diagonal
                mat = cell.view(3, 3)
                return torch.diag(mat)
            else:
                raise ValueError(f"Unsupported 1D cell shape: {cell.shape}")
        
        elif cell.dim() == 2:
            # Batched cells (B, ...)
            if cell.shape[0] == B:
                if cell.shape[1] == 3:
                    # (B, 3) - per-batch box lengths, use first sample for edge building
                    # Note: edge_index is shared across batch, so use representative box
                    return cell[0]
                elif cell.shape[1] == 6:
                    # (B, 6) - per-batch LAMMPS bounds, use first sample
                    Lx = cell[0, 1] - cell[0, 0]
                    Ly = cell[0, 3] - cell[0, 2]
                    Lz = cell[0, 5] - cell[0, 4]
                    return torch.stack([Lx, Ly, Lz])
                elif cell.shape[1] == 9:
                    # (B, 9) - per-batch flattened matrix, use first sample
                    mat = cell[0].view(3, 3)
                    return torch.diag(mat)
            elif cell.shape == (3, 3):
                # Full 3x3 cell matrix
                return torch.diag(cell)
            elif cell.shape == (1, 3):
                return cell.squeeze(0)
            elif cell.shape == (1, 6):
                cell_1d = cell.squeeze(0)
                Lx = cell_1d[1] - cell_1d[0]
                Ly = cell_1d[3] - cell_1d[2]
                Lz = cell_1d[5] - cell_1d[4]
                return torch.stack([Lx, Ly, Lz])
            else:
                raise ValueError(f"Unsupported 2D cell shape: {cell.shape}")
        
        elif cell.dim() == 3 and cell.shape[-2:] == (3, 3):
            # (B, 3, 3) - per-batch full matrix, use first sample
            return torch.diag(cell[0])
        
        else:
            raise ValueError(f"Unsupported cell tensor shape: {cell.shape}")

    def _build_edges(self, pos: torch.Tensor, box: torch.Tensor) -> torch.Tensor:
        """Build a batched edge index for a cutoff graph.

        Returns a 2 x E tensor of integer indices over the flattened nodes (batchwise offset by b*N).
        
        This implementation is fully tensorized to be TorchScript-compatible (no Python loops/lists).
        """
        B, N, _ = pos.shape
        device = pos.device
        # distances: (B, N, N) - use safe version with epsilon for 2nd derivative stability
        D = compute_distances_matrix_safe(pos=pos, n_atoms=N, PBC=self.PBC, cell=box, scaled_coords=False)
        # mask edges below cutoff, exclude self
        self_mask = ~torch.eye(N, dtype=torch.bool, device=device).unsqueeze(0).expand(B, -1, -1)
        cut = (D > 0) & (D <= self.cutoff) & self_mask  # (B, N, N)
        
        # Fully tensorized edge building (TorchScript compatible)
        # Get all (batch, row, col) indices where cut is True
        indices = cut.nonzero(as_tuple=False)  # (E, 3) where columns are [batch, row, col]
        
        if indices.shape[0] == 0:
            return torch.zeros((2, 0), dtype=torch.long, device=device)
        
        batch_idx = indices[:, 0]
        row_idx = indices[:, 1]
        col_idx = indices[:, 2]
        
        # Add batch offset: node i in batch b becomes i + b*N
        offset = batch_idx * N
        row = row_idx + offset
        col = col_idx + offset
        
        edge_index = torch.stack([row, col], dim=0).to(torch.long)
        return edge_index

    def _prepare_node_features(self, pos: torch.Tensor) -> torch.Tensor:
        """Default node features: raw xyz coordinates per node."""
        # pos: (B, N, 3) -> (B*N, 3)
        B, N, _ = pos.shape
        x = pos.reshape(B * N, 3)
        if self.in_node_nf != 3:
            # project to requested input size with a linear layer if needed
            proj = getattr(self, "_proj_in", None)
            if proj is None:
                self._proj_in = torch.nn.Linear(3, self.in_node_nf)
                proj = self._proj_in
            x = proj(x)
        return x

    # --------------- forward ----------------
    def forward(
        self, 
        X: torch.Tensor, 
        edge_index: Optional[torch.Tensor] = None,
        cell: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with optional runtime cell for NPT simulations.
        
        Parameters
        ----------
        X : torch.Tensor
            Input positions (B, N*3) or (B, N, 3)
        edge_index : torch.Tensor, optional
            Pre-computed edge indices (2, E) for Verlet list caching
        cell : torch.Tensor, optional
            Runtime cell for NPT. If None, uses self.cell.
            Shapes: (3,), (6,), (9,), (3,3), or batched (B, ...) for per-frame cells.
            
        Returns
        -------
        torch.Tensor
            Descriptor output (B, out_features) for graph mode
        """
        # X is positions: (B, N*3) or (B, N, 3)
        pos, _ = sanitize_positions_shape(pos=X, n_atoms=self.n_atoms)
        B, N, D = pos.shape
        if N != self.n_atoms or D != 3:
            raise ValueError(f"Expected positions of shape (B, {self.n_atoms}, 3), got {tuple(pos.shape)}")

        device, dtype, box = self._ensure_device_dtype(pos, cell_override=cell)
        pos = pos.to(device=device, dtype=dtype)

        # Build graph
        if edge_index is None:
            edge_index = self._build_edges(pos, box)
        else:
            edge_index = edge_index.to(device)

        # Node features and encoder
        x = self._prepare_node_features(pos)  # (B*N, in_node_nf)
        h = self.embedding(x)

        # Apply transformer layers
        for layer in self.layers:
            h = layer(h, edge_index)

        # Node head
        h = self.node_head(h)  # (B*N, hidden)

        if self.mode == "graph":
            # reshape and pool: (B, N, hidden) -> (B, hidden)
            h_b = h.view(B, N, self.hidden_nf)
            if self._pool_fn is torch.amax:
                pooled = self._pool_fn(h_b, dim=1)
            else:
                pooled = self._pool_fn(h_b, dim=1)
            out = self.graph_head(pooled)  # (B, out_features)
            return out

        # node mode: per-node predictions, then flatten to (B, N*out_features)
        assert self.node_pred is not None
        node_out = self.node_pred(h)  # (B*N, out_features)
        node_out = node_out.view(B, N * self.out_features)
        return node_out


# ---------------- Jacobian Normalization for GNN ----------------

def compute_gnn_jacobian_frobenius_norm(
    gnn: nn.Module,
    positions: torch.Tensor,
    n_atoms: int,
    batch_size: int = 32,
    device: Optional[torch.device] = None,
) -> Tuple[float, float]:
    """Compute the average Frobenius norm of the GNN Jacobian (d_descriptor/d_positions).
    
    This is used to normalize the GNN output so that different model instances
    trained on the same data produce gradients of similar magnitude.
    
    Parameters
    ----------
    gnn : nn.Module
        The GNN descriptor model (e.g., GNNTransformerDescriptor)
    positions : torch.Tensor
        Dataset of positions, shape (N_samples, n_atoms*3) or (N_samples, n_atoms, 3)
    n_atoms : int
        Number of atoms
    batch_size : int
        Batch size for processing (to manage memory)
    device : torch.device, optional
        Device for computation. If None, uses the GNN's device.
        
    Returns
    -------
    Tuple[float, float]
        (mean_frobenius_norm, std_frobenius_norm) over the dataset
    """
    if device is None:
        device = next(gnn.parameters()).device
    
    gnn = gnn.to(device).eval()
    
    # Ensure positions are (N, n_atoms*3)
    pos, _ = sanitize_positions_shape(positions, n_atoms)
    pos = pos.view(pos.shape[0], -1)  # (N, n_atoms*3)
    
    n_samples = pos.shape[0]
    frobenius_norms = []
    
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch_pos = pos[start:end].to(device).requires_grad_(True)
        
        # Forward pass through GNN
        desc = gnn(batch_pos)  # (B, out_features)
        
        # Compute Jacobian via backward pass for each output dimension
        # Frobenius norm: ||J||_F = sqrt(sum_ij J_ij^2)
        # We compute this as sqrt(sum over outputs of ||grad_i||^2)
        B, D_out = desc.shape
        D_in = batch_pos.shape[1]  # n_atoms * 3
        
        # Accumulate squared gradients
        jacobian_sq_sum = torch.zeros(B, device=device)
        
        for j in range(D_out):
            # Gradient of j-th output w.r.t. all inputs
            grad_j = torch.autograd.grad(
                outputs=desc[:, j].sum(),
                inputs=batch_pos,
                retain_graph=(j < D_out - 1),  # Keep graph for all but last
                create_graph=False
            )[0]  # (B, D_in)
            
            # Sum of squared gradients for this output dimension
            jacobian_sq_sum += (grad_j ** 2).sum(dim=1)  # (B,)
        
        # Frobenius norm for each sample in batch
        batch_frob_norms = torch.sqrt(jacobian_sq_sum).detach().cpu()
        frobenius_norms.append(batch_frob_norms)
        
        # Clear gradients
        batch_pos.grad = None
    
    all_norms = torch.cat(frobenius_norms)
    mean_norm = all_norms.mean().item()
    std_norm = all_norms.std().item()
    
    return mean_norm, std_norm


class JacobianNormalizedGNN(nn.Module):
    """Wrapper that normalizes the GNN output so that the Jacobian has unit Frobenius norm on average.
    
    The scaling is: output = gnn(x) / jacobian_scale
    
    This means: d(output)/d(x) = d(gnn(x))/d(x) / jacobian_scale
    
    If jacobian_scale = mean(||J||_F), then the normalized Jacobian has average Frobenius norm ~1.
    
    Parameters
    ----------
    gnn : nn.Module
        The GNN descriptor model to wrap
    jacobian_scale : float
        The normalization constant (typically the mean Frobenius norm computed on training data)
    """
    
    def __init__(self, gnn: nn.Module, jacobian_scale: float):
        super().__init__()
        self.gnn = gnn
        # Store as buffer so it gets saved/loaded with the model
        self.register_buffer('jacobian_scale', torch.tensor(jacobian_scale, dtype=torch.float32))
    
    def forward(
        self, 
        x: torch.Tensor, 
        edge_index: Optional[torch.Tensor] = None,
        cell: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with optional pre-computed edge_index and runtime cell.
        
        Parameters
        ----------
        x : torch.Tensor
            Input positions
        edge_index : torch.Tensor, optional
            Pre-computed edge indices for Verlet list support
        cell : torch.Tensor, optional
            Runtime cell for NPT simulations
        
        Returns
        -------
        torch.Tensor
            Normalized GNN output
        """
        return self.gnn(x, edge_index=edge_index, cell=cell) / self.jacobian_scale
    
    # Delegate attribute access to wrapped GNN for compatibility
    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.gnn, name)


def create_normalized_gnn(
    gnn: nn.Module,
    positions: torch.Tensor,
    n_atoms: int,
    batch_size: int = 32,
    device: Optional[torch.device] = None,
) -> Tuple[JacobianNormalizedGNN, float, float]:
    """Create a Jacobian-normalized GNN wrapper.
    
    Computes the mean Frobenius norm of the Jacobian over the provided positions
    and returns a wrapped GNN that divides its output by this scale factor.
    
    Parameters
    ----------
    gnn : nn.Module
        The GNN descriptor model
    positions : torch.Tensor
        Training positions to compute the normalization constant from
    n_atoms : int
        Number of atoms
    batch_size : int
        Batch size for Jacobian computation
    device : torch.device, optional
        Device for computation
        
    Returns
    -------
    Tuple[JacobianNormalizedGNN, float, float]
        (normalized_gnn, mean_jacobian_norm, std_jacobian_norm)
    """
    print("Computing GNN Jacobian Frobenius norms over dataset...")
    mean_norm, std_norm = compute_gnn_jacobian_frobenius_norm(
        gnn=gnn,
        positions=positions,
        n_atoms=n_atoms,
        batch_size=batch_size,
        device=device,
    )
    print(f"  Mean ||J||_F = {mean_norm:.6f}")
    print(f"  Std  ||J||_F = {std_norm:.6f}")
    print(f"  Coefficient of variation = {std_norm/mean_norm:.2%}")
    
    normalized_gnn = JacobianNormalizedGNN(gnn, jacobian_scale=mean_norm)
    
    return normalized_gnn, mean_norm, std_norm


# ---------------- Precomputation helpers ----------------
@torch.no_grad()
def parse_lammps_bounds(bounds: Union[List[float], np.ndarray, torch.Tensor]) -> torch.Tensor:
    """Convert LAMMPS-style bounds to box lengths.
    
    Parameters
    ----------
    bounds : List[float] | np.ndarray | torch.Tensor
        LAMMPS bounds in format [xlo, xhi, ylo, yhi, zlo, zhi] (length 6)
        OR box lengths [Lx, Ly, Lz] (length 3)
        OR single cubic box length (scalar or length 1)
        
    Returns
    -------
    torch.Tensor
        Box lengths [Lx, Ly, Lz] as a 1D tensor of shape (3,)
        
    Examples
    --------
    >>> parse_lammps_bounds([0.0, 28.5, 0.0, 28.5, 0.0, 28.5])
    tensor([28.5, 28.5, 28.5])
    
    >>> parse_lammps_bounds([-0.3, 28.2, -0.3, 28.2, -0.3, 28.2])
    tensor([28.5, 28.5, 28.5])
    
    >>> parse_lammps_bounds([28.5, 28.5, 28.5])  # Already box lengths
    tensor([28.5, 28.5, 28.5])
    """
    if isinstance(bounds, (int, float)):
        return torch.tensor([bounds, bounds, bounds], dtype=torch.float32)
    
    if isinstance(bounds, np.ndarray):
        bounds = torch.from_numpy(bounds).float()
    elif isinstance(bounds, list):
        bounds = torch.tensor(bounds, dtype=torch.float32)
    elif isinstance(bounds, torch.Tensor):
        bounds = bounds.float()
    
    bounds = bounds.flatten()
    
    if len(bounds) == 1:
        # Cubic box, single value
        return bounds.repeat(3)
    elif len(bounds) == 3:
        # Already box lengths [Lx, Ly, Lz]
        return bounds
    elif len(bounds) == 6:
        # LAMMPS format [xlo, xhi, ylo, yhi, zlo, zhi]
        xlo, xhi, ylo, yhi, zlo, zhi = bounds
        Lx = xhi - xlo
        Ly = yhi - ylo
        Lz = zhi - zlo
        return torch.tensor([Lx, Ly, Lz], dtype=torch.float32)
    else:
        raise ValueError(
            f"Invalid bounds format. Expected length 1, 3, or 6, got {len(bounds)}. "
            f"Supported formats: scalar, [Lx, Ly, Lz], or [xlo, xhi, ylo, yhi, zlo, zhi]"
        )


def build_graphs_for_positions(
    pos: torch.Tensor,
    n_atoms: int,
    PBC: bool,
    cell: Union[float, List[float], np.ndarray],
    cutoff: float,
) -> List[torch.Tensor]:
    """Precompute cutoff graphs (edge_index) for each structure in a batch of positions.

    Parameters
    ----------
    pos : torch.Tensor
        Positions tensor of shape (B, n_atoms, 3) or (B, n_atoms*3).
    n_atoms : int
        Number of atoms per structure.
    PBC : bool
        Whether to apply periodic boundary conditions.
    cell : float | List[float] | np.ndarray
        Cell dimensions used for PBC. Accepts multiple formats:
        - Scalar: cubic box with this side length
        - [Lx, Ly, Lz]: orthorhombic box lengths (length 3)
        - [xlo, xhi, ylo, yhi, zlo, zhi]: LAMMPS-style bounds (length 6)
    cutoff : float
        Distance cutoff for connecting edges.

    Returns
    -------
    List[torch.Tensor]
        A list of length B, where each element is a (2, E_i) LongTensor with local indices [0..n_atoms-1].
    """
    pos, _ = sanitize_positions_shape(pos=pos, n_atoms=n_atoms)
    B, N, D = pos.shape
    if N != n_atoms or D != 3:
        raise ValueError(f"Expected positions of shape (B, {n_atoms}, 3), got {tuple(pos.shape)}")

    # Convert LAMMPS bounds or other formats to box lengths [Lx, Ly, Lz]
    box_lengths = parse_lammps_bounds(cell)
    box = box_lengths.to(device=pos.device, dtype=pos.dtype)
    
    Dmat = compute_distances_matrix_safe(pos=pos, n_atoms=n_atoms, PBC=PBC, cell=box, scaled_coords=False)
    self_mask = ~torch.eye(n_atoms, dtype=torch.bool, device=pos.device).unsqueeze(0).expand(B, -1, -1)
    cutmask = (Dmat > 0) & (Dmat <= float(cutoff)) & self_mask

    graphs: List[torch.Tensor] = []
    for b in range(B):
        idx = cutmask[b].nonzero(as_tuple=False)
        if idx.numel() == 0:
            graphs.append(torch.zeros((2, 0), dtype=torch.long))
        else:
            row = idx[:, 0].to(torch.long).cpu()
            col = idx[:, 1].to(torch.long).cpu()
            graphs.append(torch.stack([row, col], dim=0))
    return graphs


@torch.no_grad()
def build_graphs_for_dataset(
    dataset,
    n_atoms: int,
    PBC: bool,
    cell: Union[float, List[float], np.ndarray],
    cutoff: float,
    per_frame_bounds: Optional[Union[List, np.ndarray]] = None,
) -> List[torch.Tensor]:
    """Precompute graphs for all structures in a DictDataset-like object.

    The `dataset` is expected to have a 'data' key with positions. Returns a list of edge_index tensors
    (2, E_i) with local indices per structure. You can then assign it back as dataset['graph'] = graphs.
    
    Parameters
    ----------
    dataset : DictDataset-like
        Dataset with 'data' key containing positions
    n_atoms : int
        Number of atoms per structure
    PBC : bool
        Whether to apply periodic boundary conditions
    cell : float | List[float] | np.ndarray
        Cell dimensions used for PBC if per_frame_bounds is None.
        Accepts multiple formats (see build_graphs_for_positions).
    cutoff : float
        Distance cutoff for connecting edges
    per_frame_bounds : List | np.ndarray, optional
        Per-frame cell bounds for NPT simulations. Shape (n_frames, 6) where each
        row is [xlo, xhi, ylo, yhi, zlo, zhi], OR shape (n_frames, 3) for box lengths.
        If provided, overrides the `cell` argument.
        
    Returns
    -------
    List[torch.Tensor]
        A list of length B, where each element is a (2, E_i) LongTensor with local indices.
    """
    pos = dataset["data"]
    n_frames = len(pos)
    
    if per_frame_bounds is not None:
        # NPT mode: build graphs frame-by-frame with varying cell
        graphs = []
        for i in range(n_frames):
            frame_pos = pos[i:i+1]  # Keep batch dimension
            frame_cell = per_frame_bounds[i]
            frame_graphs = build_graphs_for_positions(
                pos=frame_pos, n_atoms=n_atoms, PBC=PBC, cell=frame_cell, cutoff=cutoff
            )
            graphs.extend(frame_graphs)
        return graphs
    else:
        # NVT mode: single cell for all frames
        return build_graphs_for_positions(pos=pos, n_atoms=n_atoms, PBC=PBC, cell=cell, cutoff=cutoff)


class LightningGNNTransformer(lightning.LightningModule):
    """Thin LightningModule wrapper around GNNTransformerDescriptor to enable Trainer.fit().

    Usage:
      - Pretraining: create the wrapper with a descriptor (or descriptor args), fit with a datamodule
        producing batches with keys 'data' (positions) and 'labels' (targets).
      - After training: take `module.descriptor.eval()` and use it as a preprocessing Transform.
      
    Curvature Regularization (CR):
      - Optionally adds a physics-informed regularization that correlates model gradient space
        with potential energy space for more physically meaningful learned representations.
      - Enable by setting lambda_cr > 0 and providing cr_energy_calculator and cr_reference_histogram.
    """

    def __init__(
        self,
        descriptor: Optional[GNNTransformerDescriptor] = None,
        # If descriptor is None, the following are used to build one
        n_atoms: Optional[int] = None,
        out_features: int = 1,
        in_node_nf: int = 3,
        hidden_nf: int = 64,
        n_layers: int = 2,
        n_heads: int = 1,
        PBC: bool = True,
        cell: Union[float, List[float]] = 1.0,
        cutoff: float = 1.0,
        pool: str = "sum",
        mode: str = "graph",
        device: Optional[Union[str, torch.device]] = None,
        # Optimization
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        loss: str = "mse",  # 'mse'
        options: dict = {},
        # Curvature Regularization (CR) parameters
        lambda_cr: float = 0.0,
        cr_energy_calculator: Optional[nn.Module] = None,
        cr_reference_histogram: Optional[torch.Tensor] = None,
        cr_step_scale: float = 0.05,
        cr_n_steps: int = 5,
        cr_decay_alpha: float = 0.9,
        cr_beta: float = 1.0,
        cr_max_loss: float = 10.0,
        cr_batch_fraction: float = 1.0,
    ):
        """Initialize the Lightning GNN Transformer.
        
        Parameters
        ----------
        descriptor : GNNTransformerDescriptor, optional
            Pre-built descriptor. If None, one is created from other params.
        n_atoms : int, optional
            Number of atoms (required if descriptor is None)
        out_features : int
            Number of output features
        in_node_nf : int
            Input node feature dimension
        hidden_nf : int
            Hidden dimension
        n_layers : int
            Number of GNN layers
        n_heads : int
            Number of attention heads
        PBC : bool
            Use periodic boundary conditions
        cell : float or list
            Cell dimensions
        cutoff : float
            Distance cutoff for edges
        pool : str
            Pooling method ('sum', 'mean', 'max')
        mode : str
            'graph' or 'node'
        device : str or torch.device, optional
            Device
        lr : float
            Learning rate
        weight_decay : float
            Weight decay
        loss : str
            Loss function ('mse')
        options : dict
            Additional options for optimizer/scheduler
        lambda_cr : float
            Weight for CR loss (0 = disabled)
        cr_energy_calculator : nn.Module, optional
            Energy calculator (e.g., EAM_FS) for CR loss
        cr_reference_histogram : torch.Tensor, optional
            Mean histogram of target state B, shape (out_features,)
        cr_step_scale : float
            Position step size for CR path (in Angstroms)
        cr_n_steps : int
            Number of steps along CR path
        cr_decay_alpha : float
            Decay factor for distant steps
        cr_beta : float
            Energy scaling factor
        cr_max_loss : float
            Maximum CR loss per step
        cr_batch_fraction : float
            Fraction of batches to apply CR loss (0.0 to 1.0)
        """
        super().__init__()
        if descriptor is None:
            if n_atoms is None:
                raise ValueError("n_atoms is required when descriptor is not provided")
            descriptor = GNNTransformerDescriptor(
                n_atoms=n_atoms,
                out_features=out_features,
                in_node_nf=in_node_nf,
                hidden_nf=hidden_nf,
                n_layers=n_layers,
                n_heads=n_heads,
                PBC=PBC,
                cell=cell,
                cutoff=cutoff,
                pool=pool,
                mode=mode,
                device=device,
            )
        self.descriptor = descriptor
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        # OPTIM
        self._optimizer_name = "Adam"
        self.optimizer_kwargs = {}
        self.lr_scheduler_kwargs = {}
        for o in options.keys():
            if o == "optimizer":
                self.optimizer_kwargs.update(options[o])
            elif o == "lr_scheduler":
                self.lr_scheduler_kwargs.update(options[o])

        loss = loss.lower()
        if loss == "mse":
            self.criterion = torch.nn.MSELoss()
        else:
            raise ValueError("Unsupported loss: choose from {'mse'}")
        
        # Curvature Regularization setup
        self.lambda_cr = float(lambda_cr)
        self.cr_batch_fraction = float(cr_batch_fraction)
        self.cr_loss_fn: Optional[CurvatureRegularizationLoss] = None
        
        if self.lambda_cr > 0:
            if cr_energy_calculator is None:
                raise ValueError("cr_energy_calculator required when lambda_cr > 0")
            if cr_reference_histogram is None:
                raise ValueError("cr_reference_histogram required when lambda_cr > 0")
            
            self.cr_loss_fn = CurvatureRegularizationLoss(
                energy_calculator=cr_energy_calculator,
                reference_histogram=cr_reference_histogram,
                n_atoms=descriptor.n_atoms,
                step_scale=cr_step_scale,
                n_steps=cr_n_steps,
                decay_alpha=cr_decay_alpha,
                beta=cr_beta,
                max_loss=cr_max_loss,
            )
            # Store energy calculator as submodule for device management
            self.cr_energy_calculator = cr_energy_calculator

    def forward(
        self, 
        X: torch.Tensor, 
        edge_index: Optional[torch.Tensor] = None,
        cell: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with optional pre-computed edge_index and runtime cell.
        
        Parameters
        ----------
        X : torch.Tensor
            Input positions
        edge_index : torch.Tensor, optional
            Pre-computed edge indices for Verlet list support
        cell : torch.Tensor, optional
            Runtime cell for NPT simulations
            
        Returns
        -------
        torch.Tensor
            Descriptor output
        """
        return self.descriptor(X, edge_index=edge_index, cell=cell)

    def _prepare_targets(self, y: torch.Tensor, B: int) -> torch.Tensor:
        # Reshape targets to match descriptor outputs depending on mode
        mode = getattr(self.descriptor, "mode", "graph")
        n_atoms = getattr(self.descriptor, "n_atoms", None)
        out_features = getattr(self.descriptor, "out_features", 1)

        if mode == "graph":
            # Expect (B, out_features) or (B,) -> (B, out_features)
            y = y.reshape(B, -1)
            if y.shape[1] == 1 and out_features > 1:
                y = y.expand(B, out_features)
            return y
        else:
            # Node mode: expect (B, n_atoms*out_features) or (B, n_atoms, out_features)
            if y.dim() == 3 and y.shape[1] == n_atoms and y.shape[2] == out_features:
                y = y.reshape(B, n_atoms * out_features)
            elif y.dim() == 2 and y.shape[1] == n_atoms:
                # Single scalar per node -> tile to out_features if >1
                if out_features > 1:
                    y = y.unsqueeze(-1).expand(B, n_atoms, out_features).reshape(B, n_atoms * out_features)
            return y

    def _bounds_to_cell(self, bounds: torch.Tensor) -> torch.Tensor:
        """Convert LAMMPS-style bounds to cell dimensions.
        
        Parameters
        ----------
        bounds : torch.Tensor
            Shape (B, 6) for LAMMPS format [xlo, xhi, ylo, yhi, zlo, zhi]
            or (B, 3) already as cell dimensions [Lx, Ly, Lz]
            
        Returns
        -------
        torch.Tensor
            Cell dimensions shape (B, 3) as [Lx, Ly, Lz]
        """
        if bounds.dim() == 1:
            bounds = bounds.unsqueeze(0)
        
        if bounds.shape[-1] == 6:
            # LAMMPS format: [xlo, xhi, ylo, yhi, zlo, zhi]
            Lx = bounds[:, 1] - bounds[:, 0]
            Ly = bounds[:, 3] - bounds[:, 2]
            Lz = bounds[:, 5] - bounds[:, 4]
            return torch.stack([Lx, Ly, Lz], dim=-1)  # (B, 3)
        elif bounds.shape[-1] == 3:
            # Already cell dimensions
            return bounds
        else:
            raise ValueError(f"Unsupported bounds shape: {bounds.shape}. Expected (B, 6) or (B, 3)")

    def training_step(self, batch, batch_idx):
        # Expect a Dict-like with 'data' (positions) and 'labels' (targets)
        X = batch["data"]
        edge_index = None
        if "graph" in batch:
            g = batch["graph"]
            # Support list of per-sample graphs or pre-batched tensor
            if isinstance(g, (list, tuple)):
                # Concatenate with batch offsets
                B = X.shape[0]
                N = self.descriptor.n_atoms
                parts = []
                offset = 0
                for b in range(B):
                    gi = g[b]
                    if gi is None:
                        gi = torch.zeros((2, 0), dtype=torch.long)
                    parts.append(gi + offset)
                    offset += N
                edge_index = torch.cat(parts, dim=1) if len(parts) > 0 else None
            elif isinstance(g, torch.Tensor):
                # Accept either a single batched edge_index (2, E_tot) or padded per-sample (B, 2, E_max)
                if g.dim() == 2 and g.shape[0] == 2:
                    edge_index = g
                elif g.dim() == 3 and g.shape[1] == 2:
                    B = X.shape[0]
                    N = self.descriptor.n_atoms
                    parts = []
                    offset = 0
                    # Optional lengths per-sample
                    glen = batch.get("graph_len", None)
                    for b in range(B):
                        gb = g[b]
                        if glen is not None:
                            L = int(glen[b])
                            gb = gb[:, :L]
                        else:
                            # filter -1 padding if present
                            if gb.numel() == 0:
                                parts.append(torch.zeros((2, 0), dtype=torch.long))
                                offset += N
                                continue
                            mask = (gb[0] >= 0) & (gb[1] >= 0)
                            gb = gb[:, mask]
                        parts.append(gb.to(torch.long) + offset)
                        offset += N
                    edge_index = torch.cat(parts, dim=1) if len(parts) > 0 else None
                else:
                    # Unsupported tensor shape
                    edge_index = None
        y = batch["labels"].to(dtype=X.dtype, device=X.device)
        B = X.shape[0]
        
        # Extract bounds for NPT support (if present in batch)
        bounds = batch.get("bounds", None)
        if bounds is not None:
            bounds = bounds.to(device=X.device, dtype=X.dtype)
        
        preds = self.descriptor(X, edge_index=edge_index, cell=bounds)
        y = self._prepare_targets(y, B)
        loss_mse = self.criterion(preds, y)
        
        # Curvature Regularization loss
        loss_cr = torch.tensor(0.0, device=X.device, dtype=X.dtype)
        if self.lambda_cr > 0 and self.cr_loss_fn is not None:
            # Apply CR loss to a fraction of batches (controlled by cr_batch_fraction)
            apply_cr = (self.cr_batch_fraction >= 1.0) or (torch.rand(1).item() < self.cr_batch_fraction)
            if apply_cr:
                loss_cr = self.cr_loss_fn(
                    descriptor=self.descriptor,
                    positions=X,
                    current_predictions=preds,
                    bounds=bounds,  # Pass bounds for NPT support
                )
                self.log("train_loss_cr", loss_cr, prog_bar=True, on_step=True, on_epoch=True)
        
        # Total loss
        loss = loss_mse + self.lambda_cr * loss_cr
        
        self.log("train_loss_mse", loss_mse, prog_bar=False, on_step=True, on_epoch=True)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        X = batch["data"]
        edge_index = None
        if "graph" in batch:
            g = batch["graph"]
            if isinstance(g, (list, tuple)):
                B = X.shape[0]
                N = self.descriptor.n_atoms
                parts = []
                offset = 0
                for b in range(B):
                    gi = g[b]
                    if gi is None:
                        gi = torch.zeros((2, 0), dtype=torch.long)
                    parts.append(gi + offset)
                    offset += N
                edge_index = torch.cat(parts, dim=1) if len(parts) > 0 else None
            elif isinstance(g, torch.Tensor):
                if g.dim() == 2 and g.shape[0] == 2:
                    edge_index = g
                elif g.dim() == 3 and g.shape[1] == 2:
                    B = X.shape[0]
                    N = self.descriptor.n_atoms
                    parts = []
                    offset = 0
                    glen = batch.get("graph_len", None)
                    for b in range(B):
                        gb = g[b]
                        if glen is not None:
                            L = int(glen[b])
                            gb = gb[:, :L]
                        else:
                            if gb.numel() == 0:
                                parts.append(torch.zeros((2, 0), dtype=torch.long))
                                offset += N
                                continue
                            mask = (gb[0] >= 0) & (gb[1] >= 0)
                            gb = gb[:, mask]
                        parts.append(gb.to(torch.long) + offset)
                        offset += N
                    edge_index = torch.cat(parts, dim=1) if len(parts) > 0 else None
                else:
                    edge_index = None
        y = batch["labels"].to(dtype=X.dtype, device=X.device)
        B = X.shape[0]
        
        # Extract bounds for NPT support (if present in batch)
        bounds = batch.get("bounds", None)
        if bounds is not None:
            bounds = bounds.to(device=X.device, dtype=X.dtype)
        
        preds = self.descriptor(X, edge_index=edge_index, cell=bounds)
        y = self._prepare_targets(y, B)
        loss = self.criterion(preds, y)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    @property
    def optimizer_name(self) -> str:
        """Optimizer name. Options can be set using optimizer_kwargs. Actual optimizer will be return during training from configure_optimizer function."""
        return self._optimizer_name

    @optimizer_name.setter
    def optimizer_name(self, optimizer_name: str):
        if not hasattr(torch.optim, optimizer_name):
            raise AttributeError(
                f"torch.optim does not have a {optimizer_name} optimizer."
            )
        self._optimizer_name = optimizer_name

    def configure_optimizers(self):
        """
        Initialize the optimizer based on self._optimizer_name and self.optimizer_kwargs.

        Returns
        -------
        torch.optim
            Torch optimizer
        """

        optimizer = getattr(torch.optim, self._optimizer_name)(
            self.parameters(), **self.optimizer_kwargs
        )

        if self.lr_scheduler_kwargs:
            scheduler_cls = self.lr_scheduler_kwargs['scheduler']
            scheduler_kwargs = {k: v for k, v in self.lr_scheduler_kwargs.items() if k != 'scheduler'}
            lr_scheduler = scheduler_cls(optimizer, **scheduler_kwargs)
            return [optimizer] , [lr_scheduler]
        else: 
            return optimizer



def _to_cell_matrix_single(cell_entry, device=None, dtype=None):
    """Convert a single cell representation to a 3x3 lattice matrix.

    Supported formats:
    - (3,) lengths: [Lx, Ly, Lz] -> diag(Lx, Ly, Lz)
    - (6,) LAMMPS bounds: [xlo, xhi, ylo, yhi, zlo, zhi] -> diag(xhi-xlo, yhi-ylo, zhi-zlo)
    - (9,) flattened 3x3 (row-major) -> reshape to (3,3)
    - (3,3) lattice matrix (rows are lattice vectors)
    """
    tt = torch.as_tensor(cell_entry, device=device, dtype=dtype if dtype is not None else torch.get_default_dtype())
    if tt.ndim == 2 and tt.shape == (3, 3):
        return tt
    flat = tt.view(-1)
    n = flat.numel()
    if n == 3:
        return torch.diag(flat)
    elif n == 6:
        xlo, xhi, ylo, yhi, zlo, zhi = flat.tolist()
        L = torch.tensor([xhi - xlo, yhi - ylo, zhi - zlo], device=tt.device, dtype=tt.dtype)
        return torch.diag(L)
    elif n == 9:
        return flat.view(3, 3)
    else:
        raise ValueError("Unsupported cell format; expected 3-lengths, 6-bounds, 9-matrix or 3x3")

def _prepare_cell_matrix(cell, B, device, dtype):
    """Prepare lattice matrix/matrices for batch.

    If `cell` is a per-frame list/array of length B, stack into (B,3,3),
    otherwise return a single (3,3) matrix broadcastable across batch.
    """
    import numpy as _np
    # per-batch list/array
    if isinstance(cell, (list, tuple)) and len(cell) > 0 and isinstance(cell[0], (list, tuple, _np.ndarray, torch.Tensor)):
        if len(cell) != B:
            raise ValueError("Per-batch cell list/array must have length equal to batch size")
        mats = [ _to_cell_matrix_single(cell[i], device=device, dtype=dtype) for i in range(B) ]
        return torch.stack(mats, dim=0)  # (B,3,3)
    # torch tensor with leading batch dim
    if isinstance(cell, torch.Tensor) and cell.ndim >= 2 and cell.shape[0] == B:
        # convert each entry to 3x3 then stack
        mats = [ _to_cell_matrix_single(cell[i], device=device, dtype=dtype) for i in range(B) ]
        return torch.stack(mats, dim=0)
    # single cell entry
    return _to_cell_matrix_single(cell, device=device, dtype=dtype)  # (3,3)

def _min_image_general(dr: torch.Tensor, cell_mat: torch.Tensor):
    """Apply minimum image using lattice matrices.

    dr: (B, N, N, 3)
    cell_mat: (3,3) or (B,3,3) with rows as lattice vectors (Cartesian)
    """
    B = dr.shape[0]
    if cell_mat.ndim == 2:
        cell_mat = cell_mat.unsqueeze(0).expand(B, -1, -1)
    inv = torch.linalg.inv(cell_mat)  # (B,3,3)
    # reshape for batched matmul
    drb = dr.view(B, -1, 3)
    s = torch.bmm(drb, inv)  # fractional coordinates
    s_wrapped = s - torch.round(s)
    dr_min = torch.bmm(s_wrapped, cell_mat).view(B, dr.shape[1], dr.shape[2], 3)
    return dr_min

def _pairwise_displacements(pos: torch.Tensor, box):
    rij = pos.unsqueeze(2) - pos.unsqueeze(1)
    if box is not None:
        B = pos.shape[0]
        cell_mat = _prepare_cell_matrix(box, B, device=pos.device, dtype=pos.dtype)
        rij = _min_image_general(rij, cell_mat)
    return rij

import numpy as np

def read_all_entropy_energy(filename):
    """
    Reads all timesteps from a LAMMPS dump file containing per-atom entropy and potential energy.

    Parameters:
        filename (str): Path to the LAMMPS dump file.

    Returns:
        entropy_array (np.ndarray): Shape (n_frames, n_atoms), per-atom entropy.
        energy_array  (np.ndarray): Shape (n_frames, n_atoms), per-atom energy.
    """
    entropy_list = []
    energy_list = []

    with open(filename, 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        # Detect start of a frame
        if "ITEM: TIMESTEP" in lines[i]:
            # Move to number of atoms
            num_atoms = int(lines[i + 3].strip())

            # Move to start of atom lines
            atom_start = i + 9
            atom_end = atom_start + num_atoms

            entropy = []
            energy = []

            for line in lines[atom_start:atom_end]:
                parts = line.strip().split()
                c_ent = float(parts[-2])
                c_peatom = float(parts[-1])
                entropy.append(c_ent)
                energy.append(c_peatom)

            entropy_list.append(entropy)
            energy_list.append(energy)

            # Jump to next frame
            i = atom_end
        else:
            i += 1

    # Convert to numpy arrays
    entropy_array = np.array(entropy_list)
    energy_array = np.array(energy_list)

    return entropy_array, energy_array

def lammps_to_numpy(traj_filename):
    """
    Parse a LAMMPS trajectory file and extract coordinates and per-frame box bounds.

    Parameters:
        traj_filename (str): Path to the LAMMPS trajectory file.

    Returns:
        coords (np.ndarray): Array of shape (n_frames, n_atoms, 3) with atomic positions.
        bounds (List[List[float]]): Per-frame list of [xlo, xhi, ylo, yhi, zlo, zhi].
    """
    coords = []
    bounds = []
    #peatoms = []

    with open(traj_filename, 'r') as traj_file:
        while True:
            line = traj_file.readline()
            if not line:
                break  # End of file

            if "ITEM: TIMESTEP" in line:
                traj_file.readline()  # Skip timestep value

                traj_file.readline()  # ITEM: NUMBER OF ATOMS
                num_atoms = int(traj_file.readline().strip())

                bb_line = traj_file.readline()  # ITEM: BOX BOUNDS ...
                if not bb_line.startswith("ITEM: BOX BOUNDS"):
                    raise ValueError("Expected 'ITEM: BOX BOUNDS' line in trajectory")
                # Read three lines of bounds; each may contain 2 or 3 values (triclinic includes tilt)
                xyz_bounds = []
                for _ in range(3):
                    bline = traj_file.readline().strip().split()
                    if len(bline) < 2:
                        raise ValueError("Malformed BOX BOUNDS line; expected at least two floats")
                    lo = float(bline[0]); hi = float(bline[1])
                    xyz_bounds.append((lo, hi))
                xlo, xhi = xyz_bounds[0]
                ylo, yhi = xyz_bounds[1]
                zlo, zhi = xyz_bounds[2]
                bounds.append([xlo, xhi, ylo, yhi, zlo, zhi])

                atom_header = traj_file.readline().strip()
                if not atom_header.startswith("ITEM: ATOMS"):
                    raise ValueError("Unexpected format in ATOMS section")

                headers = atom_header.split()[2:]
                col_x = headers.index("x")
                col_y = headers.index("y")
                col_z = headers.index("z")
                #col_peatom = headers.index("c_peatom")

                frame_coords = np.zeros((num_atoms, 3), dtype=np.float64)
                #frame_peatoms = np.zeros((num_atoms,), dtype=np.float64)

                for i in range(num_atoms):
                    atom_data = traj_file.readline().strip().split()
                    frame_coords[i, 0] = float(atom_data[col_x])
                    frame_coords[i, 1] = float(atom_data[col_y])
                    frame_coords[i, 2] = float(atom_data[col_z])
                    #frame_peatoms[i] = float(atom_data[col_peatom])

                coords.append(frame_coords)
                #peatoms.append(frame_peatoms)

    return np.array(coords), bounds# , np.array(peatoms)

from mlcolvar.core.transform import Transform
from mlcolvar.core.transform.tools.utils import easy_KDE

class LogHistogram(Transform):
    """
    Compute continuous histogram using Gaussian kernels
    """

    def __init__(self,
                 in_features: int,
                 min: float,
                 max: float,
                 bins: int,
                 sigma_to_center: float = 1.0) -> torch.Tensor :
        """Computes the continuous histogram of a quantity using Gaussian kernels

        Parameters
        ----------
        in_features : int
            Number of inputs
        min : float
            Minimum value of the histogram
        max : float
            Maximum value of the histogram
        bins : int
            Number of bins of the histogram
        sigma_to_center : float, optional
            Sigma value in bin_size units, by default 1.0


        Returns
        -------
        torch.Tensor
            Values of the histogram for each bin
        """

        super().__init__(in_features=in_features, out_features=bins)

        self.min = min
        self.max = max
        self.bins = bins
        self.sigma_to_center = sigma_to_center

    def compute_hist(self, x):
        hist = easy_KDE(x=x,
                        n_input=self.in_features,
                        min_max=[self.min, self.max],
                        n=self.bins,
                        sigma_to_center=self.sigma_to_center)
        return hist

    def forward(self, x: torch.Tensor):
        x = torch.log(self.compute_hist(x) + 1e-10) - -23.025850929940457  # add small value to avoid log(0)
        return x


#ent_cryst_lammps, e_cryst = read_all_entropy_energy("unbiased/crystal/just_entropy_test.lammpstrj")
ent_melt_lammps, e_melt = read_all_entropy_energy("../../Sodium/unbiased/melt/just_entropy_test.lammpstrj")

n_at = 559 
traj_melt, bounds_melt = lammps_to_numpy("../../Sodium/unbiased/melt/dump.Na.lammpstrj")

unbiased_data = np.concatenate([traj_melt])
unbiased_labels = np.concatenate([ent_melt_lammps])

from mlcolvar.core.transform.tools import ContinuousHistogram
hist_ent = ContinuousHistogram(in_features=n_at, min=-6, max=0, bins=40)

histo_labels = hist_ent(torch.Tensor(unbiased_labels.reshape(-1,n_at)))

unbiased_bounds = np.concatenate([bounds_melt])

# Create the dataset
ds_unbiased = DictDataset({"data": torch.Tensor(unbiased_data.reshape(-1,n_at*3)), "labels": histo_labels, "bounds": torch.Tensor(unbiased_bounds)})

graphs = []
max_edges = 0
for i in range(len(ds_unbiased)):
    pos = ds_unbiased['data'][i]
    graph = build_graphs_for_positions(pos, n_atoms=n_at, PBC=True, cell=unbiased_bounds[i], cutoff=3.75)[0]
    graphs.append(graph)
    max_edges = max(max_edges, graph.shape[1])

for i in range(len(graphs)):
    # Pad with -1 to max_edges
    if graphs[i].shape[1] < max_edges:
        pad_size = max_edges - graphs[i].shape[1]
        pad = torch.full((2, pad_size), -1, dtype=torch.long)
        graph_padded = torch.cat([graphs[i], pad], dim=1)
        graphs[i] = graph_padded

ds_unbiased["graph"] = torch.stack(graphs)

datamodule = DictModule(ds_unbiased, lengths=[1], shuffle=True, batch_size=32)

# initialise lr scheduler
lr_scheduler = torch.optim.lr_scheduler.ExponentialLR

# create options dictionary
options = {'optimizer' : {'lr': 1e-3, 'weight_decay': 1e-5},
           'lr_scheduler' : {'scheduler' : lr_scheduler, 'gamma' : 0.99999},
            'nn' : {'activation' : 'tanh'}}

# number of descriptors, which is the size of the input layer
n_at = 559

# Compute example cell from first sample's bounds for GNN initialization
# The GNN will receive per-sample bounds at runtime for NPT support
example_bounds = unbiased_bounds[0]  # [xlo, xhi, ylo, yhi, zlo, zhi]
example_cell = [example_bounds[1] - example_bounds[0],  # Lx
                example_bounds[3] - example_bounds[2],  # Ly
                example_bounds[5] - example_bounds[4]]  # Lz

# initialise model
model = GNNTransformerDescriptor(
    n_atoms=n_at,
    out_features=40,
    in_node_nf=3,
    hidden_nf=64,
    n_layers=2,
    n_heads=1,
    PBC=True,
    cell=example_cell,  # Use example cell; actual cell passed at runtime for NPT
    cutoff=3.75, # Match cutoff with cutoff in the entropy calculation
    pool="sum",
    mode="graph",
    device="cuda" if torch.cuda.is_available() else "cpu",
)

metrics = MetricsCallback()

T = 350
kb = 0.0083144621 # kJ/mol
beta = 1/(kb*T)

model_train = LightningGNNTransformer(
    descriptor=model,
    options=options)


# initialize trainer, for testing the number of epochs is low, change this to something like 5000 or 100000
trainer = lightning.Trainer(callbacks=[metrics],
                            max_epochs=10000,
                            logger=False,
                            enable_checkpointing=False,
                            limit_val_batches=0,    # this is to skip validation
                            num_sanity_val_steps=0,  # this is to skip validation
                            accelerator='gpu',
                            devices=1,
                            enable_progress_bar=False
                            )

# fit model
model_train = model_train.cuda()  # Move model to GPU
trainer.fit(model_train, datamodule)

from mlcolvar.utils.plot import paletteFessa, plot_metrics
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1,1,figsize=(4,3))
ax.plot(metrics.metrics['train_loss'], label='Train Loss', color="fessa1")
ax.plot(metrics.metrics['train_loss_mse'], label="MSE Loss", color="fessa3")
ax.set_yscale('log')
plt.savefig("iter_0/desc_crnorm_conv.png", dpi=300)

out = []
model_train = model_train.cuda()  # Move model to GPU

for i in range(len(ds_unbiased["data"])):
    data = ds_unbiased["data"][i].to("cuda")
    out.append(model_train(data).detach().cpu().numpy())

fig, ax = plt.subplots(1,1,figsize=(4,3))
ax.scatter(ds_unbiased["labels"].flatten(), np.array(out).flatten(), color="fessa1", alpha=0.5, s=2)
plt.savefig("iter_0/desc_crnorm_parity.png", dpi=300)



#################################################
# Insert Generator Training
#################################################

# For committor training, reuse the trajectory data and bounds already loaded above
# (traj_cryst, traj_melt, bounds_cryst, bounds_melt, unbiased_bounds are already available)
n_at = 559

from mlcolvar.core.transform.descriptors.utils import sanitize_positions_shape

def compute_descriptors_derivatives(dataset, 
                                    descriptor_function, 
                                    n_atoms : int, 
                                    separate_boundary_dataset = True, 
                                    positions_noise : float = 0.0,
                                    batch_size : int = None):
    """Compute the derivatives of a set of descriptors wrt input positions in a dataset for committor optimization

    Parameters
    ----------
    dataset :
        DictDataset with the positions under the 'data' key
    descriptor_function : torch.nn.Module
        Transform module for the computation of the descriptors
    n_atoms : int
        Number of atoms in the system
    separate_boundary_dataset : bool, optional
            Switch to exculde boundary condition labeled data from the variational loss, by default True
    positions_noise : float
        Order of magnitude of small noise to be added to the positions to avoid atoms having the exact same coordinates on some dimension and thus zero derivatives, by default 0.
        Ideally the smaller the better, e.g., 1e-6 for single precision, even lower for double precision.
    batch_size : int
        Size of batches to process data, useful for heavy computation to avoid memory overflows, if None a singel batch is used, by default None 

    Returns
    -------
    pos : torch.Tensor
        Positions tensor (detached)
    desc : torch.Tensor
        Computed descriptors (detached)
    d_desc_d_pos : torch.Tensor
        Derivatives of desc wrt to pos (detached)
    """
    
    # apply noise if given
    if positions_noise > 0:
        noise = torch.rand_like(dataset['data'], )*positions_noise
        dataset['data'] = dataset['data'] + noise

    # get and prepare positions
    pos = dataset['data']
    labels = dataset['labels']
    pos = sanitize_positions_shape(pos=pos, n_atoms=n_atoms)[0]
    pos.requires_grad = True
    
    # Check for NPT bounds in dataset
    bounds = dataset["bounds"]
    if isinstance(bounds, np.ndarray):
        bounds = torch.from_numpy(bounds).float()
    elif not isinstance(bounds, torch.Tensor):
        bounds = torch.tensor(bounds, dtype=torch.float32)
            
    # get_device 
    device = pos.device

    # check if to separate boundary data
    if separate_boundary_dataset:
        mask_var = labels.squeeze() > 1
        if mask_var.sum()==0:
            raise(ValueError('No points left after separating boundary and variational datasets. \n If you are using only unbiased data set separate_boundary_dataset=False here and in Committor or don\'t use SmartDerivatives!!'))
    else:
        mask_var = torch.ones_like(labels.squeeze()).to(torch.bool)
    
    # check batches size for calculation
    if batch_size is None or batch_size == -1:
        batch_size = len(pos)
    else:
        if batch_size <= 0:
            raise ( ValueError(f"Batch size must be larger than zero if set! Found {batch_size}"))
    n_batches = int(np.ceil(len(pos) / batch_size))

    # compute descriptors and derivatives
    # we loop over batches and compute everything only for that part of the data, inside we loop over descriptors
    # we save lists and make them proper tensors later
    batch_aux_stack = []
    batch_desc_stack = []
    batch_count = 0
    while batch_count * batch_size + 1 <= len(pos):
        print(f"Processing batch {batch_count}/{n_batches}", end='\r')

        # get batch slicing indexes, they don't need to be all of the same size
        batch_start, batch_stop = batch_count*batch_size, (batch_count+1) * batch_size
        
        batch_mask_var = mask_var[batch_start:batch_stop]   # separate_dataset mask
        batch_pos = pos[batch_start:batch_stop]             # batch positions
        batch_bounds = bounds[batch_start:batch_stop]       # batch bounds for NPT support
        batch_pos = batch_pos[batch_mask_var, :, :]         # batch_positions for variational dataset only
        batch_bounds = batch_bounds[batch_mask_var, :]     # batch_bounds for variational dataset only
        
        if len(batch_pos) > 0:
            batch_desc = descriptor_function(batch_pos, cell=batch_bounds)

            # loop over descriptors, #TODO maybe can be done with jacobians?
            # we store things always on the cpu
            batch_aux = []
            for i in range(len(batch_desc[0])):
                aux_der = torch.autograd.grad(batch_desc[:,i], batch_pos, grad_outputs=torch.ones_like(batch_desc[:,i]), retain_graph=True )[0]
                batch_aux.append(aux_der.detach().cpu())
            
            batch_d_desc_d_pos = torch.stack(batch_aux, axis=2)         # derivatives of this batch
            batch_aux_stack.append(batch_d_desc_d_pos.detach().cpu())   # derivatives of all batches
            batch_desc_stack.append(batch_desc.detach().cpu())         # descriptors of all batches

            # cleanup
            del aux_der    
            del batch_pos
            del batch_desc

            # to be sure, clean the gpu cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        batch_count += 1
    
    print(f"Processed all data in {n_batches} batches!")

    if batch_count == 1:
        d_desc_d_pos = batch_d_desc_d_pos
        desc = batch_desc_stack
    else:
        d_desc_d_pos = torch.cat(batch_aux_stack, dim=0)
        desc = torch.cat(batch_desc_stack, dim=0)
    
    # we compute the descriptors on the whole dataset to always have all of them, no need for grads   
    #with torch.no_grad():
    #    print(pos.shape)
    #    desc = descriptor_function(pos)

    # detach and move back to original device
    pos = pos.detach().to(device)
    desc = desc.detach().to(device)
    d_desc_d_pos = d_desc_d_pos.detach().to(device)

    # to be sure, clean the gpu cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return pos, desc, d_desc_d_pos.squeeze(-1)

labels_melt = torch.zeros(len(ds_unbiased))

# Create the dataset with bounds for NPT support
ds_unbiased = DictDataset({
    "data": torch.Tensor(unbiased_data.reshape(-1, n_at*3)), 
    "labels": torch.tensor(labels_melt),
    "bounds": torch.Tensor(unbiased_bounds)  # Include bounds for NPT
})

from mlcolvar.data import DictModule, DictDataset
#from mlcolvar.core.loss.committor_loss import compute_descriptors_derivatives, SmartDerivatives
from mlcolvar.cvs.committor.utils import compute_committor_weights


# zeroth iteration should be unbiased, we thus initialise the bias as zero
bias = torch.zeros(len(ds_unbiased))

# compute weights
ds_unbiased = compute_committor_weights(dataset=ds_unbiased,
                                    bias=bias,
                                    data_groups=[0],
                                    beta=beta)

model_train.eval()
model_train.freeze()

# ============================================================================
# Jacobian Normalization: Wrap the GNN BEFORE training the Committor
# This ensures the Committor is trained with the same descriptor magnitudes
# that will be used at inference/export time.
# ============================================================================
print("\n" + "="*70)
print("JACOBIAN NORMALIZATION (before Committor training)")
print("="*70)

# Compute normalization constant over training data
n_samples_for_norm = min(1000, len(ds_unbiased['data']))
positions_for_norm = ds_unbiased['data'][:n_samples_for_norm]

normalized_gnn, mean_jac_norm, std_jac_norm = create_normalized_gnn(
    gnn=model_train,
    positions=positions_for_norm,
    n_atoms=n_at,
    batch_size=32,
    device=next(model_train.parameters()).device,
)

print(f"\nGNN Jacobian normalization applied:")
print(f"  Scale factor: {mean_jac_norm:.6f}")
print(f"  The GNN output is now divided by {mean_jac_norm:.6f}")
print(f"  This makes the average ||d(desc)/d(pos)||_F ≈ 1")
print("="*70 + "\n")

# Use the normalized GNN for VJPDerivatives setup
# This ensures Committor training uses normalized descriptors

device=next(model_train.parameters()).device
pos, desc, d_desc_d_pos = compute_descriptors_derivatives(ds_unbiased, normalized_gnn, n_at, separate_boundary_dataset = False, batch_size=128)
dataset = DictDataset({"data":desc.clone().detach().to(device), "weights":ds_unbiased["weights"].to(device),"derivatives":d_desc_d_pos.clone().detach().to(device)})

# Store the normalized GNN for later use at export time
model_normalized = normalized_gnn

atomic_masses = torch.ones(559)*22.98976928  # Sodium atomic mass in atomic mass units

gamma = 1/0.05
friction = np.zeros(n_at*3)
print(friction.shape)
for i_atom in range(n_at):
    friction[3*i_atom:3*i_atom+3] = np.array([kb*T / (gamma*atomic_masses[i_atom])]*3) 
#cell = torch.Tensor([3.0233, 3.0233, 3.0233]).to(device)
#cell = torch.ones(91*3).to(device)*3.961
friction = torch.tensor(friction, device=device,dtype=torch.float32)

###########################
# New Training Block
###########################

# =============================================================================
# GENERATOR CLASSES AND TRAINING
# =============================================================================

from mlcolvar.cvs import BaseCV
from mlcolvar.core import FeedForward
from mlcolvar.core.loss.generator_loss import GeneratorLoss, compute_eigenfunctions


class GeneratorActivation(BaseCV, lightning.LightningModule):
    """Generator CV: trains a representation of the resolvent eigenfunctions.

    forward_cv = exp(-nn(x)).
    Call compute_eigenfunctions post-training to get the eigenfunction projection.
    """

    BLOCKS = ["nn"]

    def __init__(self, layers, eta, r, alpha=20, friction=None, options=None, **kwargs):
        super().__init__(in_features=layers[0], out_features=r, **kwargs)
        self.loss_fn = GeneratorLoss(eta=eta, alpha=alpha, cell=None, friction=friction, n_cvs=r)
        self.r = r
        self.eta = eta
        self.friction = friction
        self.cell = None
        self.evecs = None
        self.evals = None
        options = self.parse_options(options or {})
        o = "nn"
        if "activation" not in options[o]:
            options[o]["activation"] = "tanh"
        self.nn = FeedForward(layers, **options[o])

    def compute_eigenfunctions(self, dataset, friction=None, eta=None, cell=None,
                               tikhonov_reg=1e-4, recompute=False):
        if friction is None:
            friction = self.friction
        if eta is None:
            eta = self.eta
        if cell is None:
            cell = self.cell
        if recompute or self.evecs is None:
            dataset["data"].requires_grad = True
            output = self.forward(dataset["data"])
            desc_derivs = dataset["derivatives"] if "derivatives" in dataset.keys else None
            eigenfunctions, evals, evecs = compute_eigenfunctions(
                dataset["data"], output, dataset["weights"],
                friction, eta, self.r, cell, tikhonov_reg,
                descriptors_derivatives=desc_derivs,
            )
            self.evals = evals
            self.evecs = evecs
            return eigenfunctions, evals, evecs
        else:
            eigenfunctions = self.forward(dataset["data"]) @ self.evecs.real
            return eigenfunctions, self.evals, self.evecs

    def forward_cv(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(-self.nn(x))

    def training_step(self, train_batch, batch_idx):
        """Compute and return the training loss and record metrics."""
        torch.set_grad_enabled(True)
        # =================get data===================
        x = train_batch["data"]
        # check data are have shape (n_data, -1)
        x = x.reshape((x.shape[0], -1))

        x.requires_grad = True

        weights = train_batch["weights"]
        if "derivatives" in train_batch.keys():
            derivatives = train_batch["derivatives"]
        else:
            derivatives = None

        # =================forward====================
        # we use forward and not forward_cv to also apply the preprocessing (if present)
        q = self.forward(x)
        # ===================loss=====================
        if self.training:
            loss, loss_ef, loss_ortho = self.loss_fn(x, q, weights, derivatives)
        else:
            loss, loss_ef, loss_ortho = self.loss_fn(x, q, weights, derivatives)
        # ====================log=====================+
        name = "train" if self.training else "valid"
        self.log(f"{name}_loss", loss, on_epoch=True)
        self.log(f"{name}_loss_var", loss_ef, on_epoch=True)
        self.log(f"{name}_loss_ortho", loss_ortho, on_epoch=True)
        return loss


class GeneratorTrivial(BaseCV, lightning.LightningModule):
    """Trivial generator CV: forward_cv = exp(-nn(x)).

    NN weights are copied from a trained GeneratorActivation.
    Eigenvector coefficients are stored but the NN already encodes the right direction.
    Used for TorchScript export (standalone or wrapped with GNN).
    """

    BLOCKS = ["nn"]

    def __init__(self, layers, eta, r, alpha=20, friction=None, options=None, coeffs=None, **kwargs):
        super().__init__(in_features=layers[0], out_features=1, **kwargs)
        self.loss_fn = GeneratorLoss(eta=eta, alpha=alpha, cell=None, friction=friction, n_cvs=r)
        options = self.parse_options(options or {})
        o = "nn"
        if "activation" not in options[o]:
            options[o]["activation"] = "tanh"
        self.nn = FeedForward(layers, **options[o])
        self.coeffs = coeffs

    def forward_cv(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(-self.nn(x))


class BiasModel(torch.nn.Module):
    """Gradient-norm-based bias model for estimating the PLUMED LAMBDA parameter.

    bias(x) = -l * (log(||grad q(x)||^2 + e) - log(e))
    """

    def __init__(self, input_model, e=1e-6, l=1):
        super().__init__()
        self.input_model = input_model
        self.l = l
        if not isinstance(e, torch.Tensor):
            e = torch.tensor([e], dtype=torch.float32)
        self.e = e.to("cpu")

    def forward(self, x):
        x = x.detach().float().requires_grad_(True)
        q = self.input_model(x)
        grad_outputs = torch.ones_like(q)
        grads = torch.autograd.grad(q, x, grad_outputs, retain_graph=True)[0]
        grads_sq = torch.sum(torch.pow(grads, 2), dim=1)
        return -self.l * (torch.log(grads_sq + self.e) - torch.log(self.e))



# TorchScript export of Committor with pretrained GNN preprocessing (device-agnostic)
import torch, gc
from torch import nn
from typing import Optional, Tuple

def make_lightning_traceable(module: nn.Module):
    """Make Lightning modules traceable by JIT by setting dummy trainer.

    PyTorch JIT's trace_module uses hasattr() to check for exported methods,
    which triggers LightningModule.trainer property causing "not attached to Trainer" error.
    This function sets a dummy _trainer attribute to prevent that error.

    Applies recursively to all submodules.
    """
    # Apply to root module
    if hasattr(module, '_trainer') or hasattr(module.__class__, 'trainer'):
        try:
            _ = module.trainer  # Test if it causes error
        except Exception:
            module._trainer = object()  # Set dummy to prevent error

    # Apply recursively to all submodules
    for name, submodule in module.named_modules():
        if submodule is module:
            continue
        if hasattr(submodule, '_trainer') or hasattr(submodule.__class__, 'trainer'):
            try:
                _ = submodule.trainer
            except Exception:
                submodule._trainer = object()

    return module



def create_and_save_ghost_npt(model, layers, r, friction, coeffs, iteration, eta=0.05,
                              alpha=20, out_dir="iter_0", example_desc=None):
    """Copy trained NN weights into a GeneratorTrivial and save a standalone TorchScript.

    Does NOT attach GNN preprocessing here; the NPT export wraps this model later.

    Parameters
    ----------
    model : GeneratorActivation
        The trained generator model (on any device).
    layers : list
        NN layer sizes (must match model).
    r : int
        Number of eigenfunctions (1 for first eigenfunction).
    friction : torch.Tensor
        Friction tensor used during training.
    coeffs : torch.Tensor
        Real part of eigenvectors, shape (r, r).  coeffs[:, 0] selects first eigenfunction.
    iteration : int
        Seed/iteration index used in the output filename.
    out_dir : str
        Directory for saved models.
    example_desc : torch.Tensor, optional
        Example descriptor of shape (1, layers[0]) for tracing.  If None, uses zeros.

    Returns
    -------
    GeneratorTrivial
        Trivial model on CPU with copied weights, ready for NPT export.
    """
    import os
    os.makedirs(out_dir, exist_ok=True)

    trivial = GeneratorTrivial(
        layers=layers, eta=eta, r=r, alpha=alpha,
        friction=friction.cpu(), coeffs=coeffs[:, 0]
    ).to("cpu").to(torch.float32)

    # Copy NN weights from trained model
    trivial.nn = copy.deepcopy(model.nn).to("cpu").to(torch.float32)

    # Save standalone descriptor-input torchscript for inspection
    if example_desc is None:
        example_desc = torch.zeros(1, layers[0], dtype=torch.float32)
    make_lightning_traceable(trivial)
    traced = torch.jit.trace(trivial, example_desc.to("cpu").float())
    torch.jit.save(traced, f"{out_dir}/model_trivial_{iteration}.pt")
    print(f"  Saved standalone generator: {out_dir}/model_trivial_{iteration}.pt")

    return trivial



# =============================================================================
# Generator Training Loop (4 seeds)
# =============================================================================

gen_options = {"nn": {"activation": "tanh"},
               "optimizer": {"lr": 5e-4, "weight_decay": 1e-5}}

num_descriptors_gen = dataset["data"].shape[1]
gen_layers = [num_descriptors_gen, 32, 32, 1]

# Store trivial models for later NPT export
gen_trivial_models = []

for i_gen in range(0,5):
    torch.manual_seed(i_gen)
    print(f"\n" + "="*70)
    print(f"GENERATOR TRAINING - Seed {i_gen}")
    print("="*70)

    dataset_gen = DictDataset({
        "data": dataset["data"].detach().clone(),
        "weights": dataset["weights"].detach().clone(),
        "derivatives": dataset["derivatives"].detach().clone(),
    })

    gen_model = GeneratorActivation(
        layers=gen_layers, eta=0.05, r=1, alpha=20, friction=friction, options=gen_options
    )

    gen_datamodule = DictModule(dataset_gen, lengths=[0.8, 0.2], batch_size=512, shuffle=True)
    gen_metrics = MetricsCallback()
    gen_early_stop = EarlyStopping(monitor="train_loss", min_delta=1e-4, patience=500, verbose=False)

    gen_trainer = lightning.Trainer(
        callbacks=[gen_metrics, gen_early_stop],
        max_epochs=10000,
        enable_checkpointing=False,
        logger=False,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        accelerator="gpu",
        devices=1,
    )

    gen_model = gen_model.cuda()
    gen_trainer.fit(gen_model, gen_datamodule)

    # Plot training metrics
    fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    ax = plot_metrics(gen_metrics.metrics,
                      keys=["train_loss", "train_loss_var", "train_loss_ortho"],
                      colors=["fessa1", "fessa3", "fessa4"],
                      ax=ax, yscale="log")
    plt.savefig(f"iter_0/Training_generator_seed{i_gen}.png", dpi=300)
    plt.close(fig)

    # Compute eigenfunctions
    gen_model = gen_model.to(device)
    dataset_gen_dev = DictDataset({
        "data": dataset_gen["data"].to(device),
        "weights": dataset_gen["weights"].to(device),
        "derivatives": dataset_gen["derivatives"].to(device),
    })
    g, evals, evecs = gen_model.compute_eigenfunctions(dataset_gen_dev, recompute=True)
    coeffs = evecs.cpu().detach().real
    print(f"  Eigenvalues: {evals.cpu().detach().numpy()}")

    # Create trivial model and save standalone TorchScript
    example_desc = dataset_gen["data"][0:1].cpu().float()
    gen_trivial = create_and_save_ghost_npt(
        gen_model, gen_layers, 1, friction, coeffs, i_gen,
        eta=0.05, alpha=20, out_dir="iter_0", example_desc=example_desc
    )
    gen_trivial_models.append(gen_trivial)

    # Estimate PLUMED LAMBDA via BiasModel
    bias_model = BiasModel(gen_trivial.to("cpu"), l=1, e=1e-7)
    desc_cpu = dataset_gen["data"].cpu().detach().float()
    bias_vals = bias_model(desc_cpu).detach()
    bias_range = (bias_vals.max() - bias_vals.min()).item()
    lambda_value = 200.0 / bias_range if bias_range > 0 else 1.0
    peak_value = lambda_value * bias_vals.max().item()
    print(f"  lambda_value = {lambda_value:.4f}  (normalize bias range to 40)")
    print(f"  peak_bias    = {peak_value:.4f}")

    # Cleanup GPU memory
    gen_model = gen_model.to("cpu")
    del gen_model, gen_trainer, gen_datamodule, dataset_gen, gen_metrics, gen_early_stop
    del g, evals, evecs, coeffs, bias_model, bias_vals
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

print(f"\nGenerator training complete. {len(gen_trivial_models)} trivial models ready for NPT export.")


#############################
# Export Block
#############################

# ============================================================================
# Verlet Neighbor List for NPT Production Inference
# ============================================================================

class VerletNeighborListDynamic(nn.Module):
    """
    TorchScript-compatible Verlet neighbor list with dynamic box for NPT simulations.
    
    Unlike VerletNeighborListInline, this version accepts the box as a runtime parameter
    in the forward pass, making it suitable for NPT simulations where the box changes.
    
    The neighbor list is rebuilt when:
    - Any atom moves more than skin/2 from its reference position, OR
    - The box strain exceeds a threshold (default 1%)
    
    Parameters
    ----------
    cutoff : float
        GNN interaction cutoff
    skin : float
        Verlet skin distance. Larger = fewer rebuilds but more edges.
    n_atoms : int
        Number of atoms
    box_strain_threshold : float
        Fractional box strain threshold for rebuilding (default 0.01 = 1%)
    """
    
    def __init__(
        self,
        cutoff: float,
        skin: float,
        n_atoms: int,
        box_strain_threshold: float = 0.01,
    ):
        super().__init__()
        self.cutoff = cutoff
        self.cutoff_sq = cutoff * cutoff
        self.skin = skin
        self.cutoff_with_skin = cutoff + skin
        self.cutoff_with_skin_sq = self.cutoff_with_skin ** 2
        self.skin_half_sq = (skin / 2.0) ** 2
        self.n_atoms = n_atoms
        self.box_strain_threshold = box_strain_threshold
        
        # Cached state (will be set at first forward)
        self.register_buffer('_cached_edge_index', torch.zeros(2, 0, dtype=torch.long))
        self.register_buffer('_ref_pos', torch.zeros(n_atoms, 3, dtype=torch.float32))
        self.register_buffer('_ref_box', torch.zeros(3, 3, dtype=torch.float32))
        self.register_buffer('_initialized', torch.tensor(False))
    
    def _min_image_dynamic(self, dr: torch.Tensor, box: torch.Tensor) -> torch.Tensor:
        """Apply minimum image convention with dynamic box.
        
        Parameters
        ----------
        dr : torch.Tensor
            Displacement vectors, shape (..., 3)
        box : torch.Tensor
            Cell matrix, shape (3, 3) with rows as lattice vectors
            
        Returns
        -------
        torch.Tensor
            Wrapped displacements in same shape as dr
        """
        # For orthorhombic boxes, use diagonal
        L = torch.diag(box)  # (3,)
        return dr - torch.round(dr / L) * L
    
    def _compute_box_strain(self, box_new: torch.Tensor, box_ref: torch.Tensor) -> float:
        """Compute relative strain between two boxes."""
        diff_norm = torch.norm(box_new - box_ref).item()
        ref_norm = torch.norm(box_ref).item()
        if ref_norm < 1e-10:
            return 0.0
        return diff_norm / ref_norm
    
    def _check_rebuild(self, pos: torch.Tensor, box: torch.Tensor) -> bool:
        """Check if neighbor list needs rebuilding."""
        if not self._initialized.item():
            return True
        
        # Check box strain
        strain = self._compute_box_strain(box, self._ref_box)
        if strain > self.box_strain_threshold:
            return True
        
        # Check atom displacement
        dr = self._min_image_dynamic(pos - self._ref_pos, box)
        disp_sq = (dr * dr).sum(dim=-1)
        max_disp_sq = disp_sq.max().item()
        
        return max_disp_sq > self.skin_half_sq
    
    def _build_edges(self, pos: torch.Tensor, box: torch.Tensor) -> torch.Tensor:
        """Build neighbor list with cutoff + skin using dynamic box."""
        # Pairwise displacements with minimum image
        dr = pos.unsqueeze(1) - pos.unsqueeze(0)  # (N, N, 3)
        dr = self._min_image_dynamic(dr, box)
        dist_sq = (dr * dr).sum(dim=-1)  # (N, N)
        
        # Neighbors within cutoff + skin, excluding self
        mask = (dist_sq < self.cutoff_with_skin_sq) & (dist_sq > 0)
        
        # Get edge indices
        edge_index = mask.nonzero(as_tuple=False).t().contiguous()  # (2, E)
        
        # Update reference state
        self._ref_pos.copy_(pos)
        self._ref_box.copy_(box)
        self._initialized.fill_(True)
        
        return edge_index
    
    def _filter_to_cutoff(self, pos: torch.Tensor, edge_index: torch.Tensor, box: torch.Tensor) -> torch.Tensor:
        """Filter edges to actual cutoff distance with current box."""
        if edge_index.shape[1] == 0:
            return edge_index
        src, dst = edge_index[0], edge_index[1]
        dr = pos[dst] - pos[src]
        dr = self._min_image_dynamic(dr, box)
        dist_sq = (dr * dr).sum(dim=-1)
        mask = dist_sq < self.cutoff_sq
        return edge_index[:, mask]
    
    def forward(self, pos: torch.Tensor, box: torch.Tensor) -> torch.Tensor:
        """
        Get neighbor list edge_index with dynamic box.
        
        Parameters
        ----------
        pos : torch.Tensor
            Atomic positions (n_atoms, 3)
        box : torch.Tensor
            Cell matrix (3, 3) with rows as lattice vectors
            
        Returns
        -------
        torch.Tensor
            Edge indices (2, E) for message passing, filtered to actual cutoff
        """
        # Check if rebuild is needed
        if self._check_rebuild(pos, box):
            self._cached_edge_index = self._build_edges(pos, box)
        
        # Filter to actual cutoff with current box
        return self._filter_to_cutoff(pos, self._cached_edge_index, box)
    
# NPT (VARIABLE BOX) WRAPPERS FOR DEPLOYMENT
# ============================================================================

class FlattenGraphPreprocessNPT(nn.Module):
    """
    GNN wrapper for NPT simulations that accepts box as a second input.
    
    This wrapper is designed for deployment in NPT simulations where the
    box size changes at each step. The box is passed from PLUMED at runtime.
    
    PLUMED centers coordinates around origin (-L/2 to L/2), but the model
    was trained on LAMMPS-style coordinates (0 to L). This wrapper automatically
    computes and applies the coordinate shift from the box diagonal at runtime.
    
    Model signature: forward(positions[1, N*3], box[1, 9]) -> features
    
    Parameters
    ----------
    gnn : nn.Module
        The GNN model (e.g., JacobianNormalizedGNN wrapping GNNTransformerDescriptor)
    n_nodes : int
        Number of atoms
    feat_dim : int
        Feature dimension per node (typically 3)
    """
    
    def __init__(
        self, 
        gnn: nn.Module, 
        n_nodes: int, 
        feat_dim: int,
    ):
        super().__init__()
        self.gnn = gnn
        self.n_nodes = int(n_nodes)
        self.feat_dim = int(feat_dim)
    
    def _box_flat_to_matrix(self, box_flat: torch.Tensor) -> torch.Tensor:
        """Convert flattened 9-element box to 3x3 matrix.
        
        Parameters
        ----------
        box_flat : torch.Tensor
            Shape (1, 9) or (9,) - flattened 3x3 cell matrix (row-major)
            
        Returns
        -------
        torch.Tensor
            Shape (3, 3) cell matrix
        """
        if box_flat.dim() == 2:
            box_flat = box_flat.squeeze(0)  # (9,)
        return box_flat.view(3, 3)
    
    def _compute_half_box(self, cell: torch.Tensor) -> torch.Tensor:
        """Compute half box dimensions from cell matrix diagonal.
        
        For orthorhombic boxes, this returns [Lx/2, Ly/2, Lz/2].
        For triclinic boxes, uses the diagonal elements as approximation.
        
        Parameters
        ----------
        cell : torch.Tensor
            Shape (3, 3) cell matrix
            
        Returns
        -------
        torch.Tensor
            Shape (3,) half box dimensions
        """
        # Extract diagonal: [cell[0,0], cell[1,1], cell[2,2]]
        # Use device-aware constant to avoid TorchScript issues
        two = torch.tensor(2.0, device=cell.device, dtype=cell.dtype)
        return torch.diag(cell) / two
    
    def forward(self, x: torch.Tensor, box: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with runtime box for NPT simulations.
        
        Automatically shifts coordinates by half-box to convert from PLUMED's
        centered coordinates (-L/2 to L/2) to LAMMPS-style (0 to L).
        
        Parameters
        ----------
        x : torch.Tensor
            Flat positions (1, N*3) or (N*3,) from PLUMED (centered at origin)
        box : torch.Tensor
            Flattened box (1, 9) or (9,) - 3x3 cell matrix in row-major order
            
        Returns
        -------
        torch.Tensor
            GNN output features
        """
        # Handle input shape
        if x.dim() == 1:
            x = x.view(1, -1)
        B, D = x.shape
        expected = self.n_nodes * self.feat_dim
        assert D == expected, f"Expected flattened size {expected}, got {D}"
        
        # Convert box to cell matrix
        cell = self._box_flat_to_matrix(box)  # (3, 3)
        
        # Compute half-box shift from box diagonal (dynamic for NPT)
        # PLUMED centers coords at origin, LAMMPS uses 0 to L
        half_box = self._compute_half_box(cell)  # (3,)
        
        # Reshape positions to (B, N, 3) for per-dimension shift
        x_nodes = x.view(B, self.n_nodes, self.feat_dim)  # (B, N, 3)
        
        # Apply coordinate shift: add half_box to each atom's xyz
        # half_box is (3,), broadcasts over (B, N, 3)
        x_nodes = x_nodes + half_box
        
        # Call GNN with runtime cell for NPT
        out = self.gnn(x_nodes, cell=cell)
        
        if out.dim() == 2:
            return out
        elif out.dim() == 3 and out.size(1) == 1:
            return out.squeeze(1)
        else:
            return out.view(B, -1)


class CommittorNPTWrapper(nn.Module):
    """
    Committor wrapper for NPT simulations that accepts box as second input.
    
    Model signature: forward(positions[1, N*3], box[1, 9]) -> CV[1, 1]
    
    This is designed for use with PYTORCH_MODEL_BIAS_VERLET_BOX which reads
    the box from PLUMED's getBox() and passes it to the model.
    """
    
    def __init__(self, committor: nn.Module, preprocess: nn.Module):
        super().__init__()
        self.comm = committor
        self.pre = preprocess
        
        # Clear preprocessing in committor (we handle it)
        if hasattr(self.comm, 'preprocessing'):
            self.comm.preprocessing = None
        if hasattr(self.comm, 'postprocessing') and self.comm.postprocessing is not None:
            self.comm.postprocessing = None
        if hasattr(self.comm, 'sigmoid') and self.comm.sigmoid is not None:
            self.comm.sigmoid = None
        
    def forward(self, x: torch.Tensor, box: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with runtime box for NPT.
        
        Parameters
        ----------
        x : torch.Tensor
            Flat positions (1, N*3)
        box : torch.Tensor
            Flattened box (1, 9) - 3x3 cell matrix in row-major order
            
        Returns
        -------
        torch.Tensor
            CV output (1, 1)
        """
        # Preprocessing with runtime box
        feats = self.pre(x, box)
        
        # Committor forward
        out = self.comm.forward_cv(feats) if hasattr(self.comm, 'forward_cv') else self.comm(feats)
        return out


# NPT (Variable Box) Export Functions
# ============================================================================


# Use the SAME normalized GNN that was used during training
# This ensures consistency between training and inference
# Force deep clone to CPU to avoid lingering CUDA references
import copy
normalized_gnn_cpu = copy.deepcopy(model_normalized).cpu()
# Use a REAL sample from the dataset for tracing/scripting
example_pos = ds_unbiased['data'][0:1].cpu()  # Take first sample

# Get example box from dataset bounds (for NPT tracing)
# The bounds are in LAMMPS format: [xlo, xhi, ylo, yhi, zlo, zhi]
example_bounds = unbiased_bounds[0]  # First sample's bounds
# Convert to cell matrix: orthorhombic box [[Lx,0,0], [0,Ly,0], [0,0,Lz]]
Lx = example_bounds[1] - example_bounds[0]
Ly = example_bounds[3] - example_bounds[2]
Lz = example_bounds[5] - example_bounds[4]
example_box = torch.tensor([
    [Lx, 0, 0],
    [0, Ly, 0],
    [0, 0, Lz]
], dtype=torch.float32).view(1, 9)  # Flatten to (1, 9) for model input

# Get the GNN cutoff for Verlet list
gnn_cutoff = normalized_gnn_cpu.gnn.cutoff if hasattr(normalized_gnn_cpu.gnn, 'cutoff') else 5.0

# =============================================================================
# GENERATOR NPT EXPORT
# Export each trained trivial generator model wrapped with the NPT GNN.
# Signature: forward(positions[1, N*3], box[1, 9]) -> CV[1, 1]
# =============================================================================

def export_generator_npt(
    generator_trivial: nn.Module,
    gnn: nn.Module,
    out_path: str,
    n_nodes: int,
    feat_dim: int = 3,
    device: str = "cpu",
    example_input: Optional[torch.Tensor] = None,
    example_box: Optional[torch.Tensor] = None,
) -> torch.jit.ScriptModule:
    """Export a GeneratorTrivial wrapped with NPT GNN preprocessing for PLUMED.

    Produces a model with signature:
        forward(positions[1, N*3], box[1, 9]) -> Tensor[1, 1]

    The model uses the same FlattenGraphPreprocessNPT + CommittorNPTWrapper
    infrastructure as the committor, replacing the committor's forward_cv with
    the generator trivial model's forward_cv = exp(-nn(x)).

    Parameters
    ----------
    generator_trivial : nn.Module
        Trained GeneratorTrivial instance (on any device; will be deep-copied to `device`).
    gnn : nn.Module
        Normalized GNN on CPU (will be deep-copied to `device`).
    out_path : str
        Output .pt file path.
    n_nodes : int
        Number of atoms.
    feat_dim : int
        Spatial dimensions per atom (3).
    device : str
        Tracing device ('cpu' recommended for PLUMED compatibility).
    example_input : torch.Tensor, optional
        Example positions (1, N*3) for tracing.
    example_box : torch.Tensor, optional
        Example box (1, 9) for tracing.

    Returns
    -------
    torch.jit.ScriptModule
    """
    dev = torch.device(device)

    # Deep-copy and move to device
    gen_triv = copy.deepcopy(generator_trivial).to(dev).eval()
    gnn_dev = copy.deepcopy(gnn).to(dev).eval()

    # Make all Lightning submodules traceable
    make_lightning_traceable(gen_triv)
    make_lightning_traceable(gnn_dev)

    # Preprocessing wrapper: positions + box -> GNN descriptors
    flat_pre = FlattenGraphPreprocessNPT(gnn=gnn_dev, n_nodes=n_nodes, feat_dim=feat_dim).to(dev)
    make_lightning_traceable(flat_pre)

    # CommittorNPTWrapper calls self.comm.forward_cv(feats); GeneratorTrivial has forward_cv
    wrapper = CommittorNPTWrapper(gen_triv, flat_pre).to(dev).eval()
    make_lightning_traceable(wrapper)

    # Force all parameters to correct device
    for name, param in wrapper.named_parameters():
        if param.device != dev:
            param.data = param.data.to(dev)

    # Prepare example inputs
    if example_input is not None:
        dummy_pos = example_input.to(dev).view(1, -1)
    else:
        dummy_pos = torch.randn(1, n_nodes * feat_dim, dtype=torch.float32, device=dev)
    if example_box is not None:
        dummy_box = example_box.to(dev).view(1, 9)
    else:
        dummy_box = (torch.eye(3, device=dev) * 10.0).view(1, 9)

    # Test before tracing
    with torch.no_grad():
        test_out = wrapper(dummy_pos, dummy_box)
    print(f"  Test CV value: {test_out[0, 0].item():.6f}")

    # Trace and save
    ts = torch.jit.trace(wrapper, (dummy_pos, dummy_box), strict=False)
    torch.jit.save(ts, out_path)
    print(f"  Saved: {out_path}")
    return ts


print("\n" + "="*70)
print("EXPORT: Generator NPT models (variable box)")
print("="*70)
for i_gen, gen_triv in enumerate(gen_trivial_models):
    out_path = f"iter_0/model_trivial_npt_{i_gen}.pt"
    print(f"\nExporting generator seed {i_gen} -> {out_path}")
    export_generator_npt(
        generator_trivial=gen_triv,
        gnn=normalized_gnn_cpu,
        out_path=out_path,
        n_nodes=559,
        feat_dim=3,
        device="cuda",
        example_input=example_pos,
        example_box=example_box,
    )

print("\n" + "="*70)
print("EXPORT COMPLETE")
print("="*70)
print("\nExported models:")
print(f"  iter_0/model_z_crnorm_npt.pt        - Committor NPT model (variable box)")
for i_gen in range(len(gen_trivial_models)):
    print(f"  iter_0/model_trivial_npt_{i_gen}.pt    - Generator NPT model, seed {i_gen} (variable box)")
    print(f"  iter_0/model_trivial_{i_gen}.pt        - Generator standalone (descriptor input)")
print("\nModel signature: forward(positions[1, 1677], box[1, 9]) -> CV[1, 1]")
print("\nCoordinate handling:")
print("  - PLUMED provides coordinates centered at origin (-L/2 to L/2)")
print("  - Model automatically shifts by half_box = diag(box)/2")
print("\n" + "="*70)
print("USAGE: Use with PYTORCH_MODEL_BIAS_VERLET_BOX in PLUMED")
print("="*70)
