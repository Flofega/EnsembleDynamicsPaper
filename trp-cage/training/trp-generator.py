import torch
import lightning
from mlcolvar.cvs import BaseCV
from mlcolvar.core import FeedForward
import sys
import numpy as np
import mlcolvar
from mlcolvar.data import DictDataset,DictModule
from mlcolvar.core.transform import PairwiseDistances
from mlcolvar.cvs.committor.utils import initialize_committor_masses, compute_committor_weights
from mlcolvar.utils.io import load_dataframe,create_dataset_from_files
from mlcolvar.utils.plot import plot_metrics, paletteFessa, paletteCortina
from mlcolvar.utils.fes import compute_fes
from mlcolvar.utils.trainer import MetricsCallback
from mlcolvar.cvs.generator import Generator

torch.set_default_dtype(torch.float64)

print("Import done")

__all__ = ["Generator"]
torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from mlcolvar.core.loss import GeneratorLoss,compute_eigenfunctions
class Generator_activation(BaseCV, lightning.LightningModule):
    """
    Baseclass for learning a representation for the eigenfunctions of the generator.
    The representation is expressed as a concatenation of the output of r neural networks.
    **Data**: for training it requires a DictDataset with the keys 'data', and 'weights'
              and optionally 'derivatives' which should contain the descriptors derivatives
    **Loss**: Minimize the representation loss and the orthonormalization loss

    """

    BLOCKS = ["nn"]

    def __init__(
        self,
        layers: list,
        eta: float,
        r: int,
        alpha: float,
        friction=None,
        cell: float = None,
        options: dict = None,
        **kwargs,
    ):
        """Define a NN-based generator model

        Parameters
        ----------
        layers : list
            Number of neurons per layer
        eta : float
            Hyperparameter for the shift to define the resolvent. $(\eta I-_mathcal{L})^{-1}$
        r : int
            Hyperparamer for the number of eigenfunctions wanted
        alpha : float
            Hyperparamer that scales the orthonormality loss
        friction: torch.tensor
            Langevin friction which should contain \sqrt{k_B*T/(gamma*m_i)}
        cell : float, optional
            CUBIC cell size length, used to scale the positions from reduce coordinates to real coordinates, by default None

        options : dict[str, Any], optional
            Options for the building blocks of the model, by default {}.
            Available blocks: ['nn'] .
        """
        super().__init__(in_features=layers[0], out_features=r, **kwargs)

        # =======  LOSS  =======
        self.loss_fn = GeneratorLoss(
            eta=eta, alpha=alpha, cell=cell, friction=friction, n_cvs=r
        )
        self.r = 1
        self.eta = eta
        self.friction = friction
        self.cell = cell
        self.evecs = None
        self.evals = None
        # ======= OPTIONS =======
        # parse and sanitize
        options = self.parse_options(options)

        # ======= BLOCKS =======
        # initialize NN turning
        o = "nn"
        # set default activation to tanh
        if "activation" not in options[o]:
            options[o]["activation"] = "tanh"
        self.nn = FeedForward(layers, **options[o])

    def compute_eigenfunctions(
        self,
        dataset,
        friction=None,
        eta=None,
        cell=None,
        tikhonov_reg=1e-4,
        recompute=False,
    ):
        """Computes the eigenfunctions based on the representation learned given by the neural networks.

        Parameters
        ----------
        dataset : DictDataset
        Dictionary containing:
            - 'data' (torch.Tensor, shape (N, d)): Input descriptors or positions.
            - 'weights' (torch.Tensor, shape (N,)): Biasing weights associated with the data points.
            - 'derivatives', optional, (torch.Tensor, shape (N,natoms, d, 3)): derivatives of the descriptors with respect to the atomic positions
        friction:torch.tensor, optional
            If different from the one used in training: Langevin friction which should contain \sqrt{k_B*T/(gamma*m_i)}
        eta : float, optional
            If different from the one used in training, Hyperparameter for the shift to define the resolvent. $(\eta I-_mathcal{L})^{-1}$

        cell : float, optional
            If different form the one used in training, CUBIC cell size length, used to scale the positions from reduce coordinates to real coordinates, by default None

        tikhonov_reg: float, optional
            Hyperparameter for the regularization of the inverse (Ridge regression parameter)
        recompute: Boolean, optional
            Is used to know if the eigenvectors are needed to be recomputed or not
        """
        if friction is None:
            friction = self.friction
        if eta is None:
            eta = self.eta
        if cell is None:
            cell = self.cell
        if (
            recompute or self.evecs is None
        ):  # If the calculation has not been done previously, or we want to compute again the eigenpairs due to a change of parameters
            dataset["data"].requires_grad = True
            output = self.forward(dataset["data"])
            if "derivatives" in dataset.keys:
                descriptors_derivatives = dataset["derivatives"]
            else:
                descriptors_derivatives = None
            eigenfunctions, evals, evecs = compute_eigenfunctions(
                dataset["data"],
                output,
                dataset["weights"],
                friction,
                eta,
                self.r,
                cell,
                tikhonov_reg,
                descriptors_derivatives=descriptors_derivatives,
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

cell = torch.Tensor([3.77103, 3.77103, 3.77103]).to(device)
print('Cell: ', cell)
# temperature in Kelvin
T = 290
# Boltzmann factor in the RIGHT ENRGY UNITS!
kb = 0.0083144621
beta = 1/(kb*T)
kT = 1/beta
print(f'Beta: {beta} \n1/beta: {1/beta}')

import torch
import lightning
from mlcolvar.cvs import BaseCV
from mlcolvar.core import FeedForward
from mlcolvar.core.loss.generator_loss import GeneratorLoss
from mlcolvar.core.loss.committor_loss import SmartDerivatives, compute_descriptors_derivatives

__all__ = ["Generator"]

from mlcolvar.core.transform import Transform
from mlcolvar.core.transform.tools.utils import easy_KDE

class LogHistogram32(Transform):
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
                        sigma_to_center=self.sigma_to_center).to(torch.float32)
        return hist

    def forward(self, x: torch.Tensor):
        x = torch.log(self.compute_hist(x) + 1e-10) - -23.025850929940457  # add small value to avoid log(0)
        return x

class ghostCV_combi_trivial(BaseCV, lightning.LightningModule):
    """Base class for data-driven learning of committor function.
    The committor function q is expressed as the output of a neural network optimized with a self-consistent
    approach based on the Kolmogorov's variational principle for the committor and on the imposition of its boundary conditions. 

    **Data**: for training it requires a DictDataset with the keys 'data', 'labels' and 'weights'

    **Loss**: Minimize Kolmogorov's variational functional of q and impose boundary condition on the metastable states (CommittorLoss)
    
    References
    ----------
    .. [*] P. Kang, E. Trizio, and M. Parrinello, "Computing the committor using the committor to study the transition state ensemble", Nat. Comput. Sci., 2024, DOI: 10.1038/s43588-024-00645-0

    See also
    --------
    mlcolvar.core.loss.CommittorLoss
        Kolmogorov's variational optimization of committor and imposition of boundary conditions
    mlcolvar.cvs.committor.utils.compute_committor_weights
        Utils to compute the appropriate weights for the training set
    mlcolvar.cvs.committor.utils.initialize_committor_masses
        Utils to initialize the masses tensor for the training
    """

    BLOCKS = ["nn", "sigmoid"]

    def __init__(
        self, 
        layers: list,
        eta: float,
        r: int,
        alpha: float = 10000,
        cell: float = None,
        friction = None,
        options: dict = None,
        coeffs = None,
        min =None,
        max =None,
        **kwargs,
    ):
        """Define a NN-based committor model

        Parameters
        ----------
        layers : list
            Number of neurons per layer
        mass : torch.Tensor
            List of masses of all the atoms we are using, for each atom we need to repeat three times for x,y,z.
            The mlcolvar.cvs.committor.utils.initialize_committor_masses can be used to simplify this.
        alpha : float
            Hyperparamer that scales the boundary conditions contribution to loss, i.e. alpha*(loss_bound_A + loss_bound_B)
        gamma : float, optional
            Hyperparamer that scales the whole loss to avoid too small numbers, i.e. gamma*(loss_var + loss_bound), by default 10000
        delta_f : float, optional
            Delta free energy between A (label 0) and B (label 1), units is kBT, by default 0. 
            State B is supposed to be higher in energy.
        cell : float, optional
            CUBIC cell size length, used to scale the positions from reduce coordinates to real coordinates, by default None
        options : dict[str, Any], optional
            Options for the building blocks of the model, by default {}.
            Available blocks: ['nn'] .
        """
        super().__init__(in_features=layers[0], out_features=layers[-1], **kwargs) 
        
        # =======  LOSS  =======
        self.loss_fn = GeneratorLoss(
                                     eta=eta,
                                     alpha=alpha,
                                     cell=cell,
                                     friction=friction,
                                     n_cvs=r
        )

        # ======= OPTIONS =======
        # parse and sanitize
        options = self.parse_options(options)

        # ======= BLOCKS =======
        # initialize NN turning
        o = "nn"
        # set default activation to tanh
        if "activation" not in options[o]: 
            options[o]["activation"] = "tanh"
        self.nn = FeedForward(layers, **options[o])
        self.coeffs = coeffs
        print(min)

        
    def forward_cv(self, x: torch.Tensor) -> (torch.Tensor):
        output = self.nn(x)
        return torch.exp(-output)

class ghostCV_combi(BaseCV, lightning.LightningModule):
    """Base class for data-driven learning of committor function.
    The committor function q is expressed as the output of a neural network optimized with a self-consistent
    approach based on the Kolmogorov's variational principle for the committor and on the imposition of its boundary conditions. 

    **Data**: for training it requires a DictDataset with the keys 'data', 'labels' and 'weights'

    **Loss**: Minimize Kolmogorov's variational functional of q and impose boundary condition on the metastable states (CommittorLoss)
    
    References
    ----------
    .. [*] P. Kang, E. Trizio, and M. Parrinello, "Computing the committor using the committor to study the transition state ensemble", Nat. Comput. Sci., 2024, DOI: 10.1038/s43588-024-00645-0

    See also
    --------
    mlcolvar.core.loss.CommittorLoss
        Kolmogorov's variational optimization of committor and imposition of boundary conditions
    mlcolvar.cvs.committor.utils.compute_committor_weights
        Utils to compute the appropriate weights for the training set
    mlcolvar.cvs.committor.utils.initialize_committor_masses
        Utils to initialize the masses tensor for the training
    """

    BLOCKS = ["nn", "sigmoid"]

    def __init__(
        self, 
        layers: list,
        eta: float,
        r: int,
        alpha: float = 10000,
        cell: float = None,
        friction = None,
        options: dict = None,
        coeffs = None,
        min =None,
        max =None,
        **kwargs,
    ):
        """Define a NN-based committor model

        Parameters
        ----------
        layers : list
            Number of neurons per layer
        mass : torch.Tensor
            List of masses of all the atoms we are using, for each atom we need to repeat three times for x,y,z.
            The mlcolvar.cvs.committor.utils.initialize_committor_masses can be used to simplify this.
        alpha : float
            Hyperparamer that scales the boundary conditions contribution to loss, i.e. alpha*(loss_bound_A + loss_bound_B)
        gamma : float, optional
            Hyperparamer that scales the whole loss to avoid too small numbers, i.e. gamma*(loss_var + loss_bound), by default 10000
        delta_f : float, optional
            Delta free energy between A (label 0) and B (label 1), units is kBT, by default 0. 
            State B is supposed to be higher in energy.
        cell : float, optional
            CUBIC cell size length, used to scale the positions from reduce coordinates to real coordinates, by default None
        options : dict[str, Any], optional
            Options for the building blocks of the model, by default {}.
            Available blocks: ['nn'] .
        """
        super().__init__(in_features=layers[0], out_features=layers[-1], **kwargs) 
        
        # =======  LOSS  =======
        self.loss_fn = GeneratorLoss(
                                     eta=eta,
                                     alpha=alpha,
                                     cell=cell,
                                     friction=friction,
                                     n_cvs=r
        )

        # ======= OPTIONS =======
        # parse and sanitize
        options = self.parse_options(options)

        # ======= BLOCKS =======
        # initialize NN turning
        o = "nn"
        # set default activation to tanh
        if "activation" not in options[o]: 
            options[o]["activation"] = "tanh"
        self.nn =FeedForward(layers, **options[o]) 
        self.coeffs = coeffs
        print(min)

        
    def forward_cv(self, x: torch.Tensor) -> (torch.Tensor):
        output = torch.exp(-self.nn(x))#torch.cat([nn(x) for nn in self.nn], dim=1)
        return output

def convert_model(model_name, n_input):
    loaded_model = torch.jit.load(model_name).to(torch.device('cpu')).to(torch.float32)
    fake_input = torch.rand(1,n_input,dtype=torch.float32).to(torch.device('cpu')).to(torch.float32)
    loaded_model(fake_input)
    frozen_model = torch.jit.trace(loaded_model, fake_input)
    torch.jit.save(frozen_model, model_name)
class BiasModel(torch.nn.Module):
    def __init__(self, input_model, e=1e-6, l=1) -> None:
        super().__init__()
        self.input_model = input_model
        self.l = l
        if type(e) is not torch.Tensor:
            e = torch.tensor([e],dtype=torch.float32,device=torch.device("cpu"))
        self.e = e.to("cpu")

    def forward(self, x):
        x.requires_grad = True
        q = self.input_model(x)
        print(q.device)
        grad_outputs = torch.ones_like(q,device="cpu",dtype=torch.float32)
        grads = torch.autograd.grad(q, x, grad_outputs, retain_graph=True)[0]
        grads_squared = torch.sum(torch.pow(grads, 2), 1)
        bias =  -self.l* (torch.log( grads_squared + self.e ) - torch.log(self.e))
        return bias

def create_and_save_ghost(model, layers, r, friction, coeffs,iteration ):
    new_model = ghostCV_combi(layers=layers,eta=0.1,r=r,cell=None,alpha=200,friction=friction,coeffs=coeffs).to("cpu")
    new_model_trivial = ghostCV_combi_trivial(layers=layers,eta=0.1,r=r,cell=None,alpha=200,friction=friction,coeffs=coeffs[:,0]).to("cpu")
    new_model.nn = model.nn.to("cpu")
    new_model_trivial.nn = model.nn.to("cpu")
    new_model.to(torch.float32)
    new_model_trivial.to(torch.float32)
    new_model.preprocessing = LogHistogram32(in_features=20, min=-500, max=-200, bins=10)
    new_model_trivial.preprocessing = LogHistogram32(in_features=20, min=-500, max=-200, bins=10)
    traced_model = new_model.to("cpu").to_torchscript(file_path=f'models_heavy_it2/model_{iteration}.pt', method='trace')
    traced_model = new_model_trivial.to("cpu").to_torchscript(file_path=f'models_heavy_it2/model_{iteration}_trivial.pt', method='trace')

    convert_model(f'models_heavy_it2/model_{iteration}.pt',20)
    convert_model(f'models_heavy_it2/model_{iteration}_trivial.pt',20)
    return new_model, new_model_trivial
class BiasModel(torch.nn.Module):
    def __init__(self, input_model, e=1e-6, l=1) -> None:
        super().__init__()
        self.input_model = input_model
        self.l = l
        if type(e) is not torch.Tensor:
            e = torch.tensor([e],dtype=torch.float32,device=torch.device("cpu"))
        self.e = e.to("cpu")

    def forward(self, x):
        x.requires_grad = True
        q = self.input_model(x)
        print(q.device)
        grad_outputs = torch.ones_like(q,device="cpu",dtype=torch.float32)
        grads = torch.autograd.grad(q, x, grad_outputs, retain_graph=True)[0]
        grads_squared = torch.sum(torch.pow(grads, 2), 1)
        bias =  -self.l* (torch.log( grads_squared + self.e ) - torch.log(self.e))
        return bias

def load_data(filenames, regexp, load_args, index_iteration, beta, ComputeDescriptors, n_atoms):
    dataset, dataframe = create_dataset_from_files(file_names = filenames,
                                               folder = None,
                                               create_labels = True,
                                               filter_args={'regex' : regexp},
                                               return_dataframe = True,
                                               load_args=load_args,
                                               verbose = True)

    bias = torch.zeros(len(dataset))
    dataset = compute_committor_weights(dataset, bias, list(range(index_iteration+1)), beta)    
    pos, desc, d_desc_d_pos = compute_descriptors_derivatives(dataset, ComputeDescriptors, n_atoms, separate_boundary_dataset = False)
    dataset = DictDataset({"data":desc.clone().detach().to(device), "weights":dataset["weights"].to(device),"derivatives":d_desc_d_pos.clone().detach().to(device)})#30 2500 epochs
    return dataset, dataframe

import torch
from typing import List, Tuple, Optional, Dict
from mlcolvar.core.transform import Transform

# ...helper functions remain unchanged above...

def _ensure_tensor(x, device=None, dtype=None):
    if x is None:
        return None
    t = torch.as_tensor(x, device=device, dtype=dtype if dtype is not None else torch.get_default_dtype())
    return t

def _min_image(d, box: Optional[torch.Tensor]):
    if box is None:
        return d
    if box.ndim == 0:
        L = box
        return d - torch.round(d / L) * L
    elif box.ndim == 1:
        return d - torch.round(d / box) * box
    else:
        return d

def _pairwise_displacements(pos: torch.Tensor, box: Optional[torch.Tensor]):
    rij = pos.unsqueeze(2) - pos.unsqueeze(1)
    if box is not None:
        rij = _min_image(rij, box)
    return rij

def _angle(v1: torch.Tensor, v2: torch.Tensor, eps=1e-12):
    n1 = torch.linalg.norm(v1, dim=-1).clamp_min(eps)
    n2 = torch.linalg.norm(v2, dim=-1).clamp_min(eps)
    cos_th = (v1 * v2).sum(dim=-1) / (n1 * n2)
    cos_th = cos_th.clamp(-1.0, 1.0)
    return torch.acos(cos_th)

def _dihedral_pbc(pi, pj, pk, pl, box: Optional[torch.Tensor], eps=1e-12):
    b1 = pj - pi
    b2 = pk - pj
    b3 = pl - pk
    if box is not None:
        b1 = _min_image(b1, box)
        b2 = _min_image(b2, box)
        b3 = _min_image(b3, box)
    n1 = torch.cross(b1, b2, dim=-1)
    n2 = torch.cross(b2, b3, dim=-1)
    n1u = n1 / torch.linalg.norm(n1, dim=-1).unsqueeze(-1).clamp_min(eps)
    n2u = n2 / torch.linalg.norm(n2, dim=-1).unsqueeze(-1).clamp_min(eps)
    b2u = b2 / torch.linalg.norm(b2, dim=-1).unsqueeze(-1).clamp_min(eps)
    m1 = torch.cross(n1u, b2u, dim=-1)
    x = (n1u * n2u).sum(dim=-1)
    y = (m1 * n2u).sum(dim=-1)
    return torch.atan2(y, x)

class MMEnergy(Transform):
    """Molecular mechanics style energy (bond, angle, dihedral, LJ, Coulomb) with optional multi-replica batching.

    Accepted input shapes (with n_replicas = R, n_atoms = N, ndim = D):
      1. (B, N*D*R)                           flattened all replicas
      2. (B, R, N*D)
      3. (B, R, N, D)
      4. (B, N*D) (only if R == 1)
      5. (B, R*N, D)  NEW: replicas concatenated along atom dimension

    Output:
      (B, R) if R > 1 else (B, 1)
    """
    def __init__(
        self,
        n_atoms: int,
        bonds: Optional[List[Tuple[int,int,float,float]]] = None,
        angles: Optional[List[Tuple[int,int,int,float,float]]] = None,
        dihedrals: Optional[List[Tuple[int,int,int,int,float,int,float]]] = None,
        ndim: int = 3,
        use_lj: bool = False,
        lj_epsilon: Optional[float] = None,
        lj_sigma: Optional[float] = None,
        lj_cutoff: Optional[float] = None,
        lj_sigma_i: Optional[List[float]] = None,
        lj_epsilon_i: Optional[List[float]] = None,
        lj_comb: str = "LB",
        use_coulomb: bool = False,
        charges: Optional[List[float]] = None,
        epsilon_r: float = 1.0,
        coulomb_cutoff: float = 0.0,
        exclude12: bool = True,
        exclude13: bool = True,
        scale14: float = 1.0,
        pbc: bool = True,
        box = None,
        force_atoms: Optional[List[int]] = None,
        n_replicas: int = 1,
    ):
        self.n_atoms = n_atoms
        self.ndim = ndim
        self.n_replicas = int(n_replicas) if n_replicas is not None else 1
        in_features = n_atoms*ndim*self.n_replicas
        out_features = self.n_replicas if self.n_replicas>1 else 1
        super().__init__(in_features=in_features, out_features=out_features)
        self.bonds = bonds or []
        self.angles = angles or []
        self.dihedrals = dihedrals or []
        self.use_lj = use_lj
        self.lj_epsilon = lj_epsilon
        self.lj_sigma = lj_sigma
        self.lj_cutoff = lj_cutoff if lj_cutoff is not None else 0.0
        self.lj_peratom = (lj_sigma_i is not None) and (lj_epsilon_i is not None)
        self.lj_sigma_i = lj_sigma_i
        self.lj_epsilon_i = lj_epsilon_i
        self.lj_comb = lj_comb.upper()
        self.use_coulomb = use_coulomb
        self.charges = charges
        self.epsilon_r = float(epsilon_r)
        self.coulomb_cutoff = float(coulomb_cutoff)
        self.exclude12 = exclude12
        self.exclude13 = exclude13
        self.scale14 = float(scale14)
        self.pbc = pbc
        self.box = None if (box is None or not pbc) else _ensure_tensor(box)
        self._build_topology_maps()
        self.ke = torch.tensor(138.935458111, dtype=torch.get_default_dtype())
        if force_atoms is None or len(force_atoms) == 0:
            self.force_mask = None
        else:
            mask = torch.zeros(self.n_atoms, dtype=torch.get_default_dtype())
            mask[torch.as_tensor(force_atoms, dtype=torch.long)] = 1.0
            self.force_mask = mask
        if self.use_lj and not self.lj_peratom:
            if not (self.lj_epsilon and self.lj_sigma and self.lj_cutoff and self.lj_epsilon>0 and self.lj_sigma>0 and self.lj_cutoff>0):
                raise ValueError("LJ requires LJ_EPSILON>0, LJ_SIGMA>0 and LJ_CUTOFF>0 (or per-atom params + cutoff).")
        if self.use_lj and self.lj_peratom and (self.lj_cutoff is None or self.lj_cutoff<=0):
            raise ValueError("When using per-atom LJ, LJ_CUTOFF must be > 0.")
        if self.use_coulomb:
            if self.charges is None or len(self.charges)!=self.n_atoms:
                raise ValueError("Coulomb requires charges list of length n_atoms.")
            if self.epsilon_r <= 0:
                raise ValueError("EPSILON_R must be > 0.")

    def _build_topology_maps(self):
        N = self.n_atoms
        bonded12 = [set() for _ in range(N)]
        bonded13 = [set() for _ in range(N)]
        for (i,j,_,_) in self.bonds:
            bonded12[i].add(j); bonded12[j].add(i)
        for (i,j,k,_,_) in self.angles:
            bonded13[i].add(k); bonded13[k].add(i)
        self._excl12 = bonded12
        self._excl13 = bonded13
        pairs14 = set()
        scale14 = {}
        if len(self.dihedrals) > 0:
            for (i,j,k,l,_,_,_) in self.dihedrals:
                a, b = (i, l) if i<l else (l, i)
                pairs14.add((a,b))
                if self.scale14 != 1.0:
                    scale14[(a,b)] = self.scale14
        self._pairs14 = pairs14
        self._scale14 = scale14

    @staticmethod
    def from_files(
        n_atoms: int,
        bonds_file: str,
        angles_file: Optional[str] = None,
        dihedrals_file: Optional[str] = None,
        charges_file: Optional[str] = None,
        lj_peratom_file: Optional[str] = None,
        n_replicas: int = 1,
        **kwargs
    ):
        def read_tokens(path):
            out = []
            with open(path, "r") as f:
                for line in f:
                    s = line.strip()
                    if not s or s.startswith("#"):
                        continue
                    toks = s.split()
                    out.append(toks)
            return out
        bonds = []
        for toks in read_tokens(bonds_file):
            if len(toks) < 4:
                continue
            i,j,k,r0 = int(toks[0])-1, int(toks[1])-1, float(toks[2]), float(toks[3])
            if i<0 or j<0:
                raise ValueError("BONDS_FILE indices must be 1-based within GROUP")
            bonds.append((i,j,k,r0))
        angles = []
        if angles_file:
            for toks in read_tokens(angles_file):
                if len(toks) < 5:
                    continue
                i,j,k,kk,t0deg = int(toks[0])-1, int(toks[1])-1, int(toks[2])-1, float(toks[3]), float(toks[4])
                if i<0 or j<0 or k<0:
                    raise ValueError("ANGLES_FILE indices must be 1-based within GROUP")
                angles.append((i,j,k,kk, t0deg))
        dihedrals = []
        if dihedrals_file:
            for toks in read_tokens(dihedrals_file):
                if len(toks) < 7:
                    continue
                i,j,k,l,kk,n,p0deg = int(toks[0])-1, int(toks[1])-1, int(toks[2])-1, int(toks[3])-1, float(toks[4]), int(toks[5]), float(toks[6])
                if i<0 or j<0 or k<0 or l<0:
                    raise ValueError("DIHEDRALS_FILE indices must be 1-based within GROUP")
                dihedrals.append((i,j,k,l,kk,n,p0deg))
        charges = None
        if charges_file:
            seq = []
            idx_map: Dict[int,float] = {}
            mixed_indexed = False
            mixed_seq = False
            for toks in read_tokens(charges_file):
                if len(toks)==1:
                    if mixed_indexed:
                        raise ValueError("CHARGES_FILE mixes formats")
                    mixed_seq = True
                    seq.append(float(toks[0]))
                elif len(toks)==2:
                    if mixed_seq:
                        raise ValueError("CHARGES_FILE mixes formats")
                    mixed_indexed = True
                    i,q = int(toks[0])-1, float(toks[1])
                    if i<0 or i>=n_atoms:
                        raise ValueError("CHARGES_FILE index out of range")
                    idx_map[i] = q
                else:
                    raise ValueError("CHARGES_FILE: invalid line")
            if mixed_indexed:
                if len(idx_map)!=n_atoms:
                    raise ValueError("CHARGES_FILE indexed mode: one entry per atom required")
                charges = [idx_map[i] for i in range(n_atoms)]
            else:
                if len(seq)!=n_atoms:
                    raise ValueError("CHARGES_FILE sequential mode: one charge per atom required")
                charges = seq
        lj_sigma_i = None
        lj_epsilon_i = None
        if lj_peratom_file:
            lj_sigma_i = [0.0] * n_atoms
            lj_epsilon_i = [0.0] * n_atoms
            filled = [False] * n_atoms
            mode_seq = False
            mode_indexed = False
            with open(lj_peratom_file, "r") as f:
                for line in f:
                    s = line.strip()
                    if not s or s.startswith("#"):
                        continue
                    toks = s.split()
                    if len(toks) == 2:
                        if mode_indexed:
                            raise ValueError("LJ_PERATOM_FILE mixes formats")
                        mode_seq = True
                        idx = sum(filled)
                        if idx >= n_atoms:
                            raise ValueError("LJ_PERATOM_FILE has more lines than atoms (sequential mode)")
                        sig = float(toks[0]); eps = float(toks[1])
                        lj_sigma_i[idx] = sig
                        lj_epsilon_i[idx] = eps
                        filled[idx] = True
                    elif len(toks) == 3:
                        if mode_seq:
                            raise ValueError("LJ_PERATOM_FILE mixes formats")
                        mode_indexed = True
                        idx = int(toks[0]) - 1
                        if idx < 0 or idx >= n_atoms:
                            raise ValueError("LJ_PERATOM_FILE index out of range (1..n_atoms)")
                        sig = float(toks[1]); eps = float(toks[2])
                        lj_sigma_i[idx] = sig
                        lj_epsilon_i[idx] = eps
                        filled[idx] = True
                    else:
                        raise ValueError("LJ_PERATOM_FILE invalid line; expected 2 or 3 tokens")
            if mode_indexed:
                if not all(filled):
                    missing = [i+1 for i,v in enumerate(filled) if not v]
                    raise ValueError(f"LJ_PERATOM_FILE missing entries for atoms: {missing}")
            else:
                if sum(filled) != n_atoms:
                    raise ValueError("LJ_PERATOM_FILE sequential mode must have exactly one line per atom")
        kdict = dict(kwargs)
        if lj_sigma_i is not None:
            kdict.update(lj_sigma_i=lj_sigma_i, lj_epsilon_i=lj_epsilon_i)
        return MMEnergy(
            n_atoms=n_atoms, bonds=bonds, angles=angles, dihedrals=dihedrals,
            charges=charges, n_replicas=n_replicas, **kdict
        )

    def _apply_force_mask(self, pos: torch.Tensor):
        if self.force_mask is None:
            return pos
        mask = self.force_mask.to(pos.device, dtype=pos.dtype).view(1, 1, self.n_atoms, 1)
        return pos * mask + pos.detach() * (1 - mask)

    def _pairwise_terms(self, pos: torch.Tensor, box: Optional[torch.Tensor]):
        B, N, D = pos.shape
        rij = _pairwise_displacements(pos, box)
        r = torch.linalg.norm(rij, dim=-1).clamp_min(1e-12)
        iu, ju = torch.triu_indices(N, N, offset=1, device=pos.device)
        r_ij = r[:, iu, ju]
        excl = torch.zeros((N, N), dtype=torch.bool, device=pos.device)
        if self.exclude12:
            for i in range(N):
                for j in self._excl12[i]:
                    excl[i, j] = True; excl[j, i] = True
        if self.exclude13:
            for i in range(N):
                for j in self._excl13[i]:
                    excl[i, j] = True; excl[j, i] = True
        scale = torch.ones((N, N), dtype=pos.dtype, device=pos.device)
        if len(self._scale14) > 0:
            for (i,j), s in self._scale14.items():
                scale[i,j] = s; scale[j,i] = s
        excl_u = excl[iu, ju]
        scale_u = scale[iu, ju].view(1, -1)
        return iu, ju, r_ij, excl_u, scale_u

    def _compute_total_energy_batch(self, pos: torch.Tensor):
        device = pos.device
        dtype = pos.dtype
        box = None if (self.box is None or not self.pbc) else self.box.to(device=device, dtype=dtype)
        if self.force_mask is not None:
            fm = self.force_mask.to(device=device, dtype=dtype).view(1, self.n_atoms, 1)
            pos = pos * fm + pos.detach() * (1 - fm)
        Btot = pos.shape[0]
        total = torch.zeros(Btot, dtype=dtype, device=device)
        if len(self.bonds)>0:
            idx_i = torch.tensor([b[0] for b in self.bonds], device=device)
            idx_j = torch.tensor([b[1] for b in self.bonds], device=device)
            k = torch.tensor([b[2] for b in self.bonds], device=device, dtype=dtype).view(1,-1)
            r0 = torch.tensor([b[3] for b in self.bonds], device=device, dtype=dtype).view(1,-1)
            ri = pos[:, idx_i, :]; rj = pos[:, idx_j, :]
            rij = ri - rj
            if box is not None: rij = _min_image(rij, box)
            r = torch.linalg.norm(rij, dim=-1).clamp_min(1e-12)
            Eb = 0.5 * k * (r - r0)**2
            total += Eb.sum(dim=1)
        if len(self.angles)>0:
            idx_i = torch.tensor([a[0] for a in self.angles], device=device)
            idx_j = torch.tensor([a[1] for a in self.angles], device=device)
            idx_k = torch.tensor([a[2] for a in self.angles], device=device)
            kk = torch.tensor([a[3] for a in self.angles], device=device, dtype=dtype).view(1,-1)
            theta0 = torch.tensor([a[4] for a in self.angles], device=device, dtype=dtype) * (torch.pi/180.0)
            theta0 = theta0.view(1,-1)
            ri = pos[:, idx_i, :]; rj = pos[:, idx_j, :]; rk = pos[:, idx_k, :]
            vji = ri - rj; vjk = rk - rj
            if box is not None:
                vji = _min_image(vji, box); vjk = _min_image(vjk, box)
            th = _angle(vji, vjk)
            Ea = 0.5 * kk * (th - theta0)**2
            total += Ea.sum(dim=1)
        if len(self.dihedrals)>0:
            idx_i = torch.tensor([d[0] for d in self.dihedrals], device=device)
            idx_j = torch.tensor([d[1] for d in self.dihedrals], device=device)
            idx_k = torch.tensor([d[2] for d in self.dihedrals], device=device)
            idx_l = torch.tensor([d[3] for d in self.dihedrals], device=device)
            k_t = torch.tensor([d[4] for d in self.dihedrals], device=device, dtype=dtype).view(1,-1)
            n   = torch.tensor([d[5] for d in self.dihedrals], device=device, dtype=dtype).view(1,-1)
            phi0 = torch.tensor([d[6] for d in self.dihedrals], device=device, dtype=dtype) * (torch.pi/180.0)
            phi0 = phi0.view(1,-1)
            pi = pos[:, idx_i, :]; pj = pos[:, idx_j, :]; pk = pos[:, idx_k, :]; pl = pos[:, idx_l, :]
            phi = _dihedral_pbc(pi, pj, pk, pl, box)
            Ed = k_t * (1.0 - torch.cos(n * (phi - phi0)))
            total += Ed.sum(dim=1)
        if self.use_lj or self.use_coulomb:
            iu, ju, r_ij, excl_u, scale_u = self._pairwise_terms(pos, box)
            if self.use_lj:
                if self.lj_peratom:
                    sig_i = _ensure_tensor(self.lj_sigma_i, device=pos.device, dtype=pos.dtype)
                    eps_i = _ensure_tensor(self.lj_epsilon_i, device=pos.device, dtype=pos.dtype)
                    if self.lj_comb == "LB":
                        sij = 0.5 * (sig_i[iu] + sig_i[ju])
                        eij = torch.sqrt((eps_i[iu] * eps_i[ju]).clamp_min(0))
                    else:
                        sij = torch.sqrt((sig_i[iu] * sig_i[ju]).clamp_min(0))
                        eij = torch.sqrt((eps_i[iu] * eps_i[ju]).clamp_min(0))
                else:
                    m = iu.numel()
                    sij = torch.full((m,), float(self.lj_sigma), device=pos.device, dtype=pos.dtype)
                    eij = torch.full((m,), float(self.lj_epsilon), device=pos.device, dtype=pos.dtype)
                cutoff_mask = (r_ij < self.lj_cutoff).to(pos.dtype)
                invr = 1.0 / r_ij
                sr = sij.view(1,-1) * invr
                sr2 = sr*sr
                sr6 = sr2*sr2*sr2
                sr12 = sr6*sr6
                v_lj = 4.0 * eij.view(1,-1) * (sr12 - sr6)
                v_lj = v_lj * scale_u * cutoff_mask * (~excl_u).to(pos.dtype)
                total += v_lj.sum(dim=1)
            if self.use_coulomb:
                q = _ensure_tensor(self.charges, device=pos.device, dtype=pos.dtype)
                qq = q[iu] * q[ju]
                if self.coulomb_cutoff > 0.0:
                    c_mask = (r_ij < self.coulomb_cutoff).to(pos.dtype)
                else:
                    c_mask = torch.ones_like(r_ij, dtype=pos.dtype)
                invr = 1.0 / r_ij
                v_c = (self.ke.to(pos.dtype) * qq.view(1,-1) * invr) / self.epsilon_r
                v_c = v_c * scale_u * c_mask * (~excl_u).to(pos.dtype)
                total += v_c.sum(dim=1)
        return total

    def _parse_input(self, x: torch.Tensor) -> torch.Tensor:
        N = self.n_atoms; D = self.ndim; R = self.n_replicas
        if x.dim()==1:
            x = x.view(1, -1)
        if x.dim()==2:
            if x.shape[1] == N*D*R:
                pos = x.view(x.shape[0], R, N, D)
            elif x.shape[1] == N*D and R==1:
                pos = x.view(x.shape[0], 1, N, D)
            else:
                raise ValueError(f"Unexpected flattened shape {tuple(x.shape)} for n_replicas={R}")
        elif x.dim()==3:
            # New case: (B, R*N, D)
            if x.shape[1] == R*N and x.shape[2] == D:
                pos = x.view(x.shape[0], R, N, D)
            # Existing case: (B, R, N*D)
            elif x.shape[1]==R and x.shape[2]==N*D:
                pos = x.view(x.shape[0], R, N, D)
            else:
                raise ValueError("3D input must be (B, R, N*D) or (B, R*N, D)")
        elif x.dim()==4:
            if x.shape[1]==R and x.shape[2]==N and x.shape[3]==D:
                pos = x
            else:
                raise ValueError("4D input must be (B, R, N, D)")
        else:
            raise ValueError("Input must have dim 1..4")
        return pos

    def _total_energy(self, x: torch.Tensor):
        pos_all = self._parse_input(x)
        B, R, N, D = pos_all.shape
        pos_flat = pos_all.view(B*R, N, D)
        energies = self._compute_total_energy_batch(pos_flat)
        energies = energies.view(B, R)
        return energies

    def forward(self, x: torch.Tensor):
        e = self._total_energy(x)
        if self.n_replicas == 1:
            return e.view(-1,1)
        return e

    def components(self, x: torch.Tensor):
        pos_all = self._parse_input(x)
        B, R, N, D = pos_all.shape
        pos_flat = pos_all.view(B*R, N, D)
        device = pos_flat.device
        dtype = pos_flat.dtype
        box = None if (self.box is None or not self.pbc) else self.box.to(device=device, dtype=dtype)
        if self.force_mask is not None:
            fm = self.force_mask.to(device=device, dtype=dtype).view(1, N, 1)
            pos_flat = pos_flat * fm + pos_flat.detach() * (1 - fm)
        out = {"bonds": torch.zeros(pos_flat.shape[0], dtype=dtype, device=device),
               "angles": torch.zeros(pos_flat.shape[0], dtype=dtype, device=device),
               "dihedrals": torch.zeros(pos_flat.shape[0], dtype=dtype, device=device),
               "lj": torch.zeros(pos_flat.shape[0], dtype=dtype, device=device),
               "coul": torch.zeros(pos_flat.shape[0], dtype=dtype, device=device)}
        if len(self.bonds)>0:
            idx_i = torch.tensor([b[0] for b in self.bonds], device=device)
            idx_j = torch.tensor([b[1] for b in self.bonds], device=device)
            k = torch.tensor([b[2] for b in self.bonds], device=device, dtype=dtype).view(1,-1)
            r0 = torch.tensor([b[3] for b in self.bonds], device=device, dtype=dtype).view(1,-1)
            ri = pos_flat[:, idx_i, :]; rj = pos_flat[:, idx_j, :]
            rij = ri - rj
            if box is not None: rij = _min_image(rij, box)
            r = torch.linalg.norm(rij, dim=-1).clamp_min(1e-12)
            Eb = 0.5 * k * (r - r0)**2
            out["bonds"] += Eb.sum(dim=1)
        if len(self.angles)>0:
            idx_i = torch.tensor([a[0] for a in self.angles], device=device)
            idx_j = torch.tensor([a[1] for a in self.angles], device=device)
            idx_k = torch.tensor([a[2] for a in self.angles], device=device)
            kk = torch.tensor([a[3] for a in self.angles], device=device, dtype=dtype).view(1,-1)
            theta0 = torch.tensor([a[4] for a in self.angles], device=device, dtype=dtype) * (torch.pi/180.0)
            theta0 = theta0.view(1,-1)
            ri = pos_flat[:, idx_i, :]; rj = pos_flat[:, idx_j, :]; rk = pos_flat[:, idx_k, :]
            vji = ri - rj; vjk = rk - rj
            if box is not None:
                vji = _min_image(vji, box); vjk = _min_image(vjk, box)
            th = _angle(vji, vjk)
            Ea = 0.5 * kk * (th - theta0)**2
            out["angles"] += Ea.sum(dim=1)
        if len(self.dihedrals)>0:
            idx_i = torch.tensor([d[0] for d in self.dihedrals], device=device)
            idx_j = torch.tensor([d[1] for d in self.dihedrals], device=device)
            idx_k = torch.tensor([d[2] for d in self.dihedrals], device=device)
            idx_l = torch.tensor([d[3] for d in self.dihedrals], device=device)
            k_t = torch.tensor([d[4] for d in self.dihedrals], device=device, dtype=dtype).view(1,-1)
            n   = torch.tensor([d[5] for d in self.dihedrals], device=device, dtype=dtype).view(1,-1)
            phi0 = torch.tensor([d[6] for d in self.dihedrals], device=device, dtype=dtype) * (torch.pi/180.0)
            phi0 = phi0.view(1,-1)
            pi = pos_flat[:, idx_i, :]; pj = pos_flat[:, idx_j, :]; pk = pos_flat[:, idx_k, :]; pl = pos_flat[:, idx_l, :]
            phi = _dihedral_pbc(pi, pj, pk, pl, box)
            Ed = k_t * (1.0 - torch.cos(n * (phi - phi0)))
            out["dihedrals"] += Ed.sum(dim=1)
        if self.use_lj or self.use_coulomb:
            iu, ju, r_ij, excl_u, scale_u = self._pairwise_terms(pos_flat, box)
            if self.use_lj:
                if self.lj_peratom:
                    sig_i = _ensure_tensor(self.lj_sigma_i, device=pos_flat.device, dtype=pos_flat.dtype)
                    eps_i = _ensure_tensor(self.lj_epsilon_i, device=pos_flat.device, dtype=pos_flat.dtype)
                    if self.lj_comb == "LB":
                        sij = 0.5 * (sig_i[iu] + sig_i[ju])
                        eij = torch.sqrt((eps_i[iu] * eps_i[ju]).clamp_min(0))
                    else:
                        sij = torch.sqrt((sig_i[iu] * sig_i[ju]).clamp_min(0))
                        eij = torch.sqrt((eps_i[iu] * eps_i[ju]).clamp_min(0))
                else:
                    m = iu.numel()
                    sij = torch.full((m,), float(self.lj_sigma), device=pos_flat.device, dtype=pos_flat.dtype)
                    eij = torch.full((m,), float(self.lj_epsilon), device=pos_flat.device, dtype=pos_flat.dtype)
                cutoff_mask = (r_ij < self.lj_cutoff).to(pos_flat.dtype)
                invr = 1.0 / r_ij
                sr = sij.view(1,-1) * invr
                sr2 = sr*sr
                sr6 = sr2*sr2*sr2
                sr12 = sr6*sr6
                v_lj = 4.0 * eij.view(1,-1) * (sr12 - sr6)
                v_lj = v_lj * scale_u * cutoff_mask * (~excl_u).to(pos_flat.dtype)
                out["lj"] += v_lj.sum(dim=1)
            if self.use_coulomb:
                q = _ensure_tensor(self.charges, device=pos_flat.device, dtype=pos_flat.dtype)
                qq = q[iu] * q[ju]
                if self.coulomb_cutoff > 0.0:
                    c_mask = (r_ij < self.coulomb_cutoff).to(pos_flat.dtype)
                else:
                    c_mask = torch.ones_like(r_ij, dtype=pos_flat.dtype)
                invr = 1.0 / r_ij
                v_c = (self.ke.to(pos_flat.dtype) * qq.view(1,-1) * invr) / self.epsilon_r
                v_c = v_c * scale_u * c_mask * (~excl_u).to(pos_flat.dtype)
                out["coul"] += v_c.sum(dim=1)
        for k in list(out.keys()):
            out[k] = out[k].view(B, R)
        out["total"] = out["bonds"] + out["angles"] + out["dihedrals"] + out["lj"] + out["coul"]


import MDAnalysis as mda

n_rep = 20
n_at = 272
pos_unfolded = np.zeros((5000, n_rep, n_at*3))

for i in range(n_rep):

    u_unfolded = mda.Universe("sims/1/%d/%d.gro" % (i,i), "sims/1/%d/traj_comp.xtc" % i)
    pos_unfo = []

    for t in u_unfolded.trajectory:
        ats = u_unfolded.atoms.select_atoms("protein")
        pos = ats.atoms.positions / 10.0
        pos_tensor = torch.tensor(pos.flatten(), dtype=torch.get_default_dtype()).view(1, -1)
        pos_unfo.append(pos_tensor)


    pos_unfolded[:,i,:] = torch.cat(pos_unfo[:250000:50], dim=0).numpy()

labels_folded = torch.zeros(pos_unfolded.shape[0])
pos_all = torch.cat([torch.tensor(pos_unfolded).reshape(5000, n_rep*n_at*3)])
labels_all = torch.cat([labels_folded])

ds_unbiased = DictDataset({"data": pos_all, "labels": labels_all})

from mlcolvar.data import DictModule, DictDataset
#from mlcolvar.core.loss.committor_loss import compute_descriptors_derivatives, SmartDerivatives
from mlcolvar.cvs.committor.utils import compute_committor_weights

# compute weights
ds = compute_committor_weights(dataset=ds_unbiased,
                                    bias=torch.zeros(len(pos_all)),
                                    data_groups=[0],
                                    beta=beta)


ats = u_unfolded.atoms.select_atoms("protein")
heavy = ats.select_atoms("protein and not name H*")
heavy_indices = heavy.indices

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


# Descriptors
from mlcolvar.core.transform.tools import ContinuousHistogram

mm_energy = MMEnergy.from_files(
    n_atoms=n_at,
    bonds_file="bonds_mm.dat",
    angles_file="angles_mm.dat",
    dihedrals_file="dihedrals_mm.dat",
    lj_peratom_file="lj_peratom_mm.dat",
    use_lj=True,
    lj_cutoff=1.0,
    lj_comb="LB",
    use_coulomb=True,
    charges_file="charges_mm.dat",
    epsilon_r=80.0,
    exclude12=True,
    exclude13=True,
    scale14=0.0,
    box = cell,
    force_atoms = heavy_indices,
    n_replicas=n_rep,
)

hist_ene = LogHistogram(in_features=20, min=-500, max=-200, bins=10)
preprocessing_ene = torch.nn.Sequential(mm_energy, hist_ene)


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
        batch_pos = batch_pos[batch_mask_var, :, :]         # batch_positions for variational dataset only
        
        if len(batch_pos) > 0:
            batch_desc = descriptor_function(batch_pos)

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

print("preprocessing")
pos, desc, d_desc_d_pos = compute_descriptors_derivatives(ds, preprocessing_ene, n_rep*n_at, separate_boundary_dataset = False, batch_size=128)
dataset = DictDataset({"data":desc.clone().detach().to(device), "weights":ds["weights"].to(device),"derivatives":d_desc_d_pos.clone().detach().to(device)})#30 2500 epochs

print("done")
ats = u_unfolded.atoms.select_atoms("protein")
atomic_masses = torch.repeat_interleave(torch.tensor(ats.masses),n_rep)

gamma = 1/0.05
friction = np.zeros(n_at*n_rep*3)
print(friction.shape)
for i_atom in range(n_at*n_rep):
    friction[3*i_atom:3*i_atom+3] = np.array([kT / (gamma*atomic_masses[i_atom])]*3) 
#cell = torch.Tensor([3.0233, 3.0233, 3.0233]).to(device)
#cell = torch.ones(91*3).to(device)*3.961
friction = torch.tensor(friction, device=device,dtype=torch.float32)


from mlcolvar.data import DictModule
from mlcolvar.utils.io import create_dataset_from_files
from mlcolvar.cvs.committor.utils import compute_committor_weights
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from mlcolvar.utils.trainer import MetricsCallback

#import wandb
#from lightning.pytorch.loggers import WandbLogger
from mlcolvar.utils.plot import plot_metrics, paletteFessa, paletteCortina
from mlcolvar.core.loss.generator_loss import compute_eigenfunctions
import os
import gc
import matplotlib.pyplot as plt
import MDAnalysis as mda
print("Initiating Training")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seeds = range(0,2,1)
options = { 'nn':{'activation':'tanh'},
         'optimizer' : {'lr': 5e-4, 'weight_decay': 1e-5}, 
           }  
for i in seeds:
    torch.manual_seed(i)
    
    # Detach dataset tensors to ensure they are leaf variables for each new model
    dataset_iter = DictDataset({
        "data": dataset["data"].detach().clone(),
        "weights": dataset["weights"].detach().clone(),
        "derivatives": dataset["derivatives"].detach().clone()
    })
    
    model = Generator_activation(layers=[10,32,32,1],eta=0.05,r=1,alpha=20,friction=friction, options=options)

    datamodule = DictModule(dataset_iter, lengths=[0.8,0.2], batch_size=256, shuffle=True)
    #wandb_logger = WandbLogger(name=f"seed_{i}_activation",project='dipeptide_clean')
    metrics = MetricsCallback()

    early_stop_callback = EarlyStopping(monitor="train_loss", min_delta=1e-4, patience=500, verbose=False)

    trainer = lightning.Trainer(callbacks=[metrics,early_stop_callback], 
                            max_epochs=10000, 
                            enable_checkpointing=False,
                            logger=False,
                            limit_val_batches=0,    # this to skip validation
                            num_sanity_val_steps=0  # this to skip validation
                            )
    # fit model
    trainer.fit(model, datamodule)

    torch.save(model,f"models_heavy_it2/model_iter_{i}_all.pt")
    fig, ax = plt.subplots(1,1,figsize=(4,3))
    ax = plot_metrics(metrics.metrics,
                  keys=['train_loss', 'train_loss_var', 'train_loss_ortho'],
                  colors=['fessa1', 'fessa3', 'fessa4', 'fessa5'],
                  ax = ax, yscale="log")
    plt.show()
    model = model.to(device)
    g, evals, evecs = model.compute_eigenfunctions(dataset_iter)

    coeffs = evecs.cpu().detach().real
    new_model_all, new_model = create_and_save_ghost(model, [10,32,32,1], 1, friction, coeffs, i )


    bias = BiasModel(model.to("cpu"),l=1,e=1e-7).to(torch.float64)
    lambda_value = 40/(bias(dataset_iter["data"].cpu().detach()).max() - bias(dataset_iter["data"].cpu().detach()).min())#30/(bias(dataset["data"].cpu().detach()).max() - bias(dataset["data"].cpu().detach()).min())
    #lambda_value = 40/(bias(dataset["data"].cpu().detach()).max() - bias(dataset["data"].cpu().detach()).min())
    value = lambda_value * bias(dataset_iter["data"].cpu().detach()).max()
    print("lambda_value", lambda_value)
    print("value", value)
    #os.system(f"./run.sh {i} {-value} {lambda_value} ")

    #os.chdir("../")
    
    # === CLEANUP CUDA MEMORY ===
    # Move model to CPU before deleting
    model = model.to("cpu")
    
    # Delete references to objects holding GPU memory
    del model, trainer, datamodule, dataset_iter, metrics
    del g, evals, evecs, coeffs
    del new_model_all, new_model, bias
    
    # Run garbage collection
    gc.collect()
    
    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Close the figure to free matplotlib memory
    plt.close(fig)
