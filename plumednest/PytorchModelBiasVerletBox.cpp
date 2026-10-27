/* +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Copyright (c) 2022 of Luigi Bonati and Enrico Trizio.
Modified 2026 for NPT (variable box) support with Verlet neighbor list caching.

The pytorch module is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

The pytorch module is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with plumed.  If not, see <http://www.gnu.org/licenses/>.
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ */

#ifdef __PLUMED_HAS_LIBTORCH

#include "core/PlumedMain.h"
#include "function/Function.h"
#include "core/ActionRegister.h"
#include "tools/Pbc.h"

#include <torch/torch.h>
#include <torch/script.h>

#include <fstream>
#include <cmath>

// Backward compatibility hack for <1.10
#if (TORCH_VERSION_MAJOR == 1 && TORCH_VERSION_MINOR <= 10)
  #define DO_TORCH_FREEZE_HACK
  #include <torch/csrc/jit/passes/freeze_module.h>
  #include <torch/csrc/jit/passes/frozen_graph_optimizations.h>
#endif

using namespace std;

namespace PLMD {
namespace function {
namespace pytorch {

//+PLUMEDOC PYTORCH_FUNCTION PYTORCH_MODEL_BIAS_VERLET_BOX
/*
Load a PyTorch model with NPT (variable box) support and Verlet neighbor list caching.

This action is designed for GNN-based collective variables in NPT simulations where 
the box size varies. The simulation box is passed as additional arguments after the 
positions (9 values for the flattened 3x3 cell matrix).

The model expects a forward signature of:
  forward(positions: Tensor[1, N*3], box: Tensor[1, 9]) -> Tensor[1, 1]

where the box is the flattened 3x3 cell matrix (row-major: [[a_x, a_y, a_z], [b_x, b_y, b_z], [c_x, c_y, c_z]]).

The box must be passed as the last 9 arguments (after the N*3 position arguments).
For orthorhombic boxes, only the diagonal elements (xx, yy, zz) are significant.

The Verlet neighbor list is automatically rebuilt when:
- Any atom moves more than skin/2 from its reference position, OR
- The box strain exceeds a threshold (default: 1%)

This ensures correct neighbor lists even as the box fluctuates in NPT.

\par Examples

\plumedfile
# NPT simulation with GNN collective variable
# Pass positions (N*3 values) followed by box (9 values: xx,xy,xz,yx,yy,yz,zx,zy,zz)
pos: POSITION ATOMS=1-559 NOPBC
cell: CELL
model: PYTORCH_MODEL_BIAS_VERLET_BOX FILE=model.pt ARG=pos.x,pos.y,pos.z,cell.ax,cell.ay,cell.az,cell.bx,cell.by,cell.bz,cell.cx,cell.cy,cell.cz NATOMS=559 CUTOFF=5.0 SKIN=1.0 LAMBDA=1.0
# Note: For orthorhombic, pass: cell.ax,0,0,0,cell.by,0,0,0,cell.cz (or use actual cell components)
METAD ARG=model.node-0 ...
\endplumedfile

*/
//+ENDPLUMEDOC

class PytorchModelBiasVerletBox :
  public Function
{
  // Model configuration
  unsigned _n_in;        // Total number of arguments
  unsigned _n_pos_args;  // Number of position arguments (n_atoms * 3)
  unsigned _n_out;
  unsigned _n_atoms;
  double lambda;
  double epsilon;
  double grad_scale;
  torch::jit::Module _model;
  torch::Device device = torch::kCPU;
  
  // Box input configuration
  unsigned _box_start_idx;  // Index where box arguments start
  bool _has_box_args;       // Whether box is passed as arguments
  
  // Verlet neighbor list parameters
  double _cutoff;
  double _skin;
  double _cutoff_with_skin;
  double _cutoff_with_skin_sq;
  double _skin_half_sq;
  
  // Box strain threshold for rebuild (fraction)
  double _box_strain_threshold;
  
  // Cached neighbor list state
  torch::Tensor _cached_edge_index;
  torch::Tensor _ref_positions;  // Positions at last rebuild
  torch::Tensor _ref_box;        // Box at last rebuild (3x3)
  bool _initialized;
  unsigned long _rebuild_count;
  unsigned long _total_steps;
  
public:
  explicit PytorchModelBiasVerletBox(const ActionOptions&);
  void calculate();
  static void registerKeywords(Keywords& keys);

  std::vector<float> tensor_to_vector(const torch::Tensor& x);
  
  // Verlet neighbor list methods with dynamic box
  torch::Tensor min_image(const torch::Tensor& dr, const torch::Tensor& box);
  bool check_rebuild(const torch::Tensor& positions, const torch::Tensor& box);
  torch::Tensor build_neighbor_list(const torch::Tensor& positions, const torch::Tensor& box);
  torch::Tensor filter_to_cutoff(const torch::Tensor& positions, const torch::Tensor& edge_index, const torch::Tensor& box);
  
  // Get box tensor from PLUMED
  torch::Tensor get_box_tensor();
  
  // Check box strain
  double compute_box_strain(const torch::Tensor& box_new, const torch::Tensor& box_ref);
};

PLUMED_REGISTER_ACTION(PytorchModelBiasVerletBox,"PYTORCH_MODEL_BIAS_VERLET_BOX")

void PytorchModelBiasVerletBox::registerKeywords(Keywords& keys) {
  Function::registerKeywords(keys);
  keys.use("ARG");
  keys.add("optional","FILE","Filename of the PyTorch compiled model");
  keys.add("optional","LAMBDA","Prefactor of the bias, default 1");
  keys.add("optional","EPSILON","Numerical regularization term in the logarithm, default 1e-6");
  keys.add("optional","GRAD_SCALE","Gradient scaling for unit compatibility. Set to 10 when using LENGTH=A, default 1");
  // Verlet list parameters
  keys.add("compulsory","NATOMS","Number of atoms in the system");
  keys.add("compulsory","CUTOFF","GNN cutoff distance in same units as positions");
  keys.add("optional","SKIN","Verlet skin distance, default 1.0");
  keys.add("optional","BOX_STRAIN_THRESHOLD","Fractional box strain threshold for neighbor list rebuild, default 0.01 (1%)");
  // GPU support
  keys.addFlag("GPU",false,"Use GPU for model inference and gradient computation");
  keys.addOutputComponent("node", "default", "Model outputs");
  keys.addOutputComponent("bias", "default", "Bias outputs");
}

std::vector<float> PytorchModelBiasVerletBox::tensor_to_vector(const torch::Tensor& x) {
  torch::Tensor cpu_tensor = x.to(torch::kCPU).contiguous();
  return std::vector<float>(cpu_tensor.data_ptr<float>(), cpu_tensor.data_ptr<float>() + cpu_tensor.numel());
}

torch::Tensor PytorchModelBiasVerletBox::get_box_tensor() {
  // Get current box from input arguments (last 9 arguments after positions)
  std::vector<float> box_data(9);
  for (int i = 0; i < 9; i++) {
    box_data[i] = static_cast<float>(getArgument(_box_start_idx + i));
  }
  
  return torch::tensor(box_data, torch::TensorOptions().dtype(torch::kFloat32).device(device)).view({3, 3});
}

double PytorchModelBiasVerletBox::compute_box_strain(const torch::Tensor& box_new, const torch::Tensor& box_ref) {
  // Compute relative strain as ||B_new - B_ref|| / ||B_ref||
  torch::Tensor diff = box_new - box_ref;
  float diff_norm = diff.norm().item<float>();
  float ref_norm = box_ref.norm().item<float>();
  if (ref_norm < 1e-10) return 0.0;
  return static_cast<double>(diff_norm / ref_norm);
}

// Minimum image convention with dynamic box (orthorhombic fast path)
torch::Tensor PytorchModelBiasVerletBox::min_image(const torch::Tensor& dr, const torch::Tensor& box) {
  // For orthorhombic boxes, use diagonal elements
  // box is (3, 3), diagonal is [L_x, L_y, L_z]
  torch::Tensor L = torch::diagonal(box);  // (3,)
  return dr - torch::round(dr / L) * L;
}

// Check if neighbor list needs rebuilding (positions or box changed)
bool PytorchModelBiasVerletBox::check_rebuild(const torch::Tensor& positions, const torch::Tensor& box) {
  if (!_initialized) {
    return true;
  }
  
  // Check box strain
  double strain = compute_box_strain(box, _ref_box);
  if (strain > _box_strain_threshold) {
    return true;
  }
  
  // Check atom displacement
  torch::Tensor dr = min_image(positions - _ref_positions, box);
  torch::Tensor disp_sq = (dr * dr).sum(-1);  // Sum over xyz
  float max_disp_sq = disp_sq.max().item<float>();
  
  return max_disp_sq > _skin_half_sq;
}

// Build neighbor list with cutoff + skin
torch::Tensor PytorchModelBiasVerletBox::build_neighbor_list(const torch::Tensor& positions, const torch::Tensor& box) {
  // Compute pairwise displacements with minimum image
  torch::Tensor dr = positions.unsqueeze(0) - positions.unsqueeze(1);  // (N, N, 3)
  dr = min_image(dr, box);
  
  // Squared distances
  torch::Tensor dist_sq = (dr * dr).sum(-1);  // (N, N)
  
  // Mask: within cutoff+skin and not self
  torch::Tensor mask = (dist_sq < _cutoff_with_skin_sq) & (dist_sq > 0);
  
  // Get edge indices
  torch::Tensor edge_index = mask.nonzero();  // (E, 2)
  edge_index = edge_index.t().contiguous();   // (2, E)
  
  // Update reference state
  _ref_positions = positions.clone();
  _ref_box = box.clone();
  _initialized = true;
  _rebuild_count++;
  
  return edge_index;
}

// Filter cached edges to actual cutoff
torch::Tensor PytorchModelBiasVerletBox::filter_to_cutoff(const torch::Tensor& positions, 
                                                          const torch::Tensor& edge_index,
                                                          const torch::Tensor& box) {
  if (edge_index.size(1) == 0) {
    return edge_index;
  }
  
  torch::Tensor src = edge_index[0];
  torch::Tensor dst = edge_index[1];
  
  // Current distances for cached edges
  torch::Tensor dr = positions.index({dst}) - positions.index({src});
  dr = min_image(dr, box);
  torch::Tensor dist_sq = (dr * dr).sum(-1);
  
  // Filter to actual cutoff
  torch::Tensor mask = dist_sq < (_cutoff * _cutoff);
  
  return edge_index.index({torch::indexing::Slice(), mask});
}

PytorchModelBiasVerletBox::PytorchModelBiasVerletBox(const ActionOptions&ao):
  Action(ao),
  Function(ao),
  _initialized(false),
  _rebuild_count(0),
  _total_steps(0),
  _has_box_args(true)
{
  // Total number of inputs (positions + box)
  _n_in = getNumberOfArguments();
  
  // Parse number of atoms
  parse("NATOMS", _n_atoms);
  
  // Expected: N*3 position arguments + 9 box arguments
  _n_pos_args = _n_atoms * 3;
  _box_start_idx = _n_pos_args;
  
  unsigned expected_args = _n_pos_args + 9;
  if (_n_in != expected_args) {
    plumed_merror("Number of arguments (" + std::to_string(_n_in) + 
                  ") must equal NATOMS*3 + 9 (" + std::to_string(expected_args) + 
                  "): N*3 positions followed by 9 box components (xx,xy,xz,yx,yy,yz,zx,zy,zz)");
  }
  
  // Parse model filename
  std::string fname = "model.ptc";
  parse("FILE", fname);
  
  // Parse bias parameters
  lambda = 1.0;
  parse("LAMBDA", lambda);
  epsilon = 1e-6;
  parse("EPSILON", epsilon);
  grad_scale = 1.0;
  parse("GRAD_SCALE", grad_scale);
  
  // Parse GPU flag
  bool use_gpu = false;
  parseFlag("GPU", use_gpu);
  if (use_gpu) {
    if (torch::cuda::is_available()) {
      device = torch::Device(torch::kCUDA);
      log.printf("  GPU acceleration ENABLED (CUDA available)\n");
    } else {
      log.printf("  WARNING: GPU requested but CUDA not available, using CPU\n");
    }
  }
  
  // Parse Verlet list parameters
  parse("CUTOFF", _cutoff);
  _skin = 1.0;
  parse("SKIN", _skin);
  
  // Box strain threshold
  _box_strain_threshold = 0.01;  // 1% default
  parse("BOX_STRAIN_THRESHOLD", _box_strain_threshold);
  
  // Derived quantities
  _cutoff_with_skin = _cutoff + _skin;
  _cutoff_with_skin_sq = _cutoff_with_skin * _cutoff_with_skin;
  _skin_half_sq = (_skin / 2.0) * (_skin / 2.0);
  
  // Initialize reference tensors (will be set on first calculate())
  _ref_positions = torch::zeros({(long)_n_atoms, 3}, 
                                torch::TensorOptions().dtype(torch::kFloat32).device(device));
  _ref_box = torch::zeros({3, 3}, 
                          torch::TensorOptions().dtype(torch::kFloat32).device(device));
  
  // Initialize empty edge index
  _cached_edge_index = torch::zeros({2, 0}, 
                                    torch::TensorOptions().dtype(torch::kLong).device(device));
  
  // Metadata for model loading
  std::unordered_map<std::string, std::string> metadata = {
    {"_jit_bailout_depth", ""},
    {"_jit_fusion_strategy", ""}
  };

  // Load model
  try {
    _model = torch::jit::load(fname, device, metadata);
  } catch (const c10::Error& e) {
    std::ifstream infile(fname);
    bool exist = infile.good();
    infile.close();
    if (exist) {
      std::stringstream ss;
      ss << TORCH_VERSION_MAJOR << "." << TORCH_VERSION_MINOR << "." << TORCH_VERSION_PATCH;
      std::string version;
      ss >> version;
      plumed_merror("Cannot load FILE: '" + fname + "'. LibTorch version: " + version);
    } else {
      plumed_merror("The FILE: '" + fname + "' does not exist.");
    }
  }
  checkRead();

  // Optimize model (limited, to preserve gradient computation)
  _model.eval();
  
  // Note: We apply limited optimizations to preserve gradient flow
  // Freezing is done carefully to maintain 2nd derivative support
  #ifdef DO_TORCH_FREEZE_HACK
    // Don't freeze for older versions - it can break gradients
  #else
    // For newer versions, freeze is safer but we skip it to ensure grad support
    // _model = torch::jit::freeze(_model);
  #endif
  
  #if (TORCH_VERSION_MAJOR == 1 && TORCH_VERSION_MINOR <= 10)
    size_t jit_bailout_depth = 1;
    if (!metadata["_jit_bailout_depth"].empty()) {
      jit_bailout_depth = std::stoi(metadata["_jit_bailout_depth"]);
    }
    torch::jit::getBailoutDepth() = jit_bailout_depth;
  #else
    torch::jit::FusionStrategy strategy;
    if (metadata["_jit_fusion_strategy"].empty()) {
      strategy = {{torch::jit::FusionBehavior::DYNAMIC, 0}};
    } else {
      std::stringstream strat_stream(metadata["_jit_fusion_strategy"]);
      std::string fusion_type, fusion_depth;
      while(std::getline(strat_stream, fusion_type, ',')) {
        std::getline(strat_stream, fusion_depth, ';');
        strategy.push_back({
          fusion_type == "STATIC" ? torch::jit::FusionBehavior::STATIC : torch::jit::FusionBehavior::DYNAMIC, 
          std::stoi(fusion_depth)
        });
      }
    }
    torch::jit::setFusionStrategy(strategy);
  #endif

  // Test model with dummy input (positions and box)
  log.printf("Checking output dimension (NPT model with box input):\n");
  std::vector<float> input_test(_n_pos_args);  // Positions only
  torch::Tensor single_input = torch::tensor(input_test).view({1, (long)_n_pos_args});
  single_input = single_input.to(device);
  
  // Create dummy box tensor (3x3 flattened to 9)
  torch::Tensor dummy_box = torch::eye(3, torch::TensorOptions().dtype(torch::kFloat32).device(device)) * 10.0;
  torch::Tensor dummy_box_flat = dummy_box.view({1, 9});
  
  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(single_input);
  inputs.push_back(dummy_box_flat);
  torch::Tensor output = _model.forward(inputs).toTensor();
  vector<float> cvs = this->tensor_to_vector(output);
  _n_out = cvs.size();

  log.printf("  Model outputs %d values\n", _n_out);
  if (_n_out >= 2) {
    log.printf("  Detected model with grad_norm_sq output (fast bias computation)\n");
  }

  // Create output components
  addComponentWithDerivatives("node-0");
  componentIsNotPeriodic("node-0");
  
  addComponentWithDerivatives("bias-0");
  componentIsNotPeriodic("bias-0");

  // Print configuration
  log.printf("Pytorch Model Loaded with NPT (Variable Box) Support\n");
  log.printf("  Device: %s\n", device.is_cuda() ? "CUDA (GPU)" : "CPU");
  log.printf("  Number of atoms: %d\n", _n_atoms);
  log.printf("  Number of position args: %d\n", _n_pos_args);
  log.printf("  Number of box args: 9\n");
  log.printf("  Total arguments: %d\n", _n_in);
  log.printf("  Number of outputs: %d\n", _n_out);
  log.printf("  GNN cutoff: %f\n", _cutoff);
  log.printf("  Verlet skin: %f\n", _skin);
  log.printf("  Total neighbor radius: %f\n", _cutoff_with_skin);
  log.printf("  Box strain threshold: %f (%0.1f%%)\n", _box_strain_threshold, _box_strain_threshold * 100);
  log.printf("  Gradient scale: %f\n", grad_scale);
  
  log.printf("  Model signature: forward(positions[1,N*3], box[1,9]) -> CV\n");
  log.printf("  Box is read from input arguments (last 9 args after positions)\n");
  
  log.printf("  Bibliography: ");
  log << plumed.cite("Bonati, Rizzi and Parrinello, J. Phys. Chem. Lett. 11, 2998-3004 (2020)");
  log << plumed.cite("Trizio and Parrinello, J. Phys. Chem. Lett. 12, 8621-8626 (2021)");
  log.printf("\n");
}


void PytorchModelBiasVerletBox::calculate() {
  _total_steps++;
  
  // Retrieve positions from arguments (first N*3 arguments)
  vector<float> current_pos(_n_pos_args);
  for (unsigned i = 0; i < _n_pos_args; i++) {
    current_pos[i] = getArgument(i);
  }
  
  // Convert to tensor: (1, N*3) for model input, (N, 3) for neighbor list
  torch::Tensor input_flat = torch::tensor(current_pos).view({1, (long)_n_pos_args}).to(device);
  torch::Tensor positions = torch::tensor(current_pos).view({(long)_n_atoms, 3}).to(device);
  
  // Get current box from arguments (last 9 arguments)
  torch::Tensor box = get_box_tensor();  // (3, 3)
  torch::Tensor box_flat = box.view({1, 9});  // (1, 9) for model input
  
  // =====================================================================
  // VERLET NEIGHBOR LIST: Check and rebuild if necessary (box-aware)
  // =====================================================================
  if (check_rebuild(positions, box)) {
    _cached_edge_index = build_neighbor_list(positions, box);
  }
  // Filter cached edges to actual cutoff with current box
  torch::Tensor edge_index = filter_to_cutoff(positions, _cached_edge_index, box);
  
  // =====================================================================
  // MODEL FORWARD PASS (with box as second input)
  // =====================================================================
  input_flat.set_requires_grad(true);
  
  // Pass both positions AND box to model
  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(input_flat);
  inputs.push_back(box_flat);
  
  torch::Tensor output = _model.forward(inputs).toTensor();
  
  // =====================================================================
  // Extract CV value (first output)
  // =====================================================================
  torch::Tensor cv_output;
  if (_n_out >= 2) {
    cv_output = output.index({torch::indexing::Slice(), 0}).unsqueeze(1);  // Shape: (1, 1)
  } else {
    cv_output = output;
  }
  
  // =====================================================================
  // GRADIENT COMPUTATION (of CV only)
  // =====================================================================
  torch::Tensor grad_output = torch::ones({1}).expand({1, 1}).to(device);
  torch::Tensor gradient = torch::autograd::grad(
    {cv_output},
    {input_flat},
    {grad_output},
    /*retain_graph=*/true,
    /*create_graph=*/true
  )[0];
  
  // =====================================================================
  // BIAS COMPUTATION
  // =====================================================================
  torch::Tensor Epsilon = torch::tensor(epsilon).to(device);
  torch::Tensor log_grad_sq;
  torch::Tensor grad_norm_sq;
  
  if (_n_out >= 2) {
    // Fast path: grad_norm_sq is provided by model
    grad_norm_sq = output[0][1];
  } else {
    // Slow path: compute grad_norm_sq from gradient
    torch::Tensor GradScale = torch::tensor(grad_scale).to(device);
    torch::Tensor gradient_scaled = gradient * GradScale;
    grad_norm_sq = torch::sum(torch::pow(gradient_scaled, 2));
  }
  
  // Compute bias from grad_norm_sq
  if (lambda == 0.0) {
    log_grad_sq = torch::zeros({1}).to(device);
  } else {
    log_grad_sq = lambda * (torch::log(grad_norm_sq + Epsilon) - torch::log(Epsilon));
  }
  
  // =====================================================================
  // BIAS DERIVATIVES
  // =====================================================================
  torch::Tensor gradient2;
  if (lambda == 0.0) {
    gradient2 = torch::zeros_like(gradient);
  } else {
    torch::Tensor grad_output_bias = torch::ones({1}).to(device);
    
    torch::Tensor d_grad_norm_sq = torch::autograd::grad({grad_norm_sq},
                            {input_flat},
          /*grad_outputs=*/ {grad_output_bias},
          /*retain_graph=*/true,
          /*create_graph=*/false)[0];
    
    gradient2 = lambda * d_grad_norm_sq / (grad_norm_sq + Epsilon);
  }

  // =====================================================================
  // SET PLUMED OUTPUTS
  // =====================================================================
  
  // CV derivatives (only for position arguments, box args get zero)
  vector<float> der = this->tensor_to_vector(gradient);  
  string name_comp = "node-0";
  for (unsigned i = 0; i < _n_pos_args; i++) {
    setDerivative(getPntrToComponent(name_comp), i, der[i]);
  }
  // Set zero derivatives for box arguments
  for (unsigned i = _n_pos_args; i < _n_in; i++) {
    setDerivative(getPntrToComponent(name_comp), i, 0.0);
  }
  
  // Bias derivatives (only for position arguments, box args get zero)
  vector<float> der2 = this->tensor_to_vector(gradient2);  
  string name_comp_bias = "bias-0";
  for (unsigned i = 0; i < _n_pos_args; i++) {
    setDerivative(getPntrToComponent(name_comp_bias), i, der2[i]);
  }
  // Set zero derivatives for box arguments
  for (unsigned i = _n_pos_args; i < _n_in; i++) {
    setDerivative(getPntrToComponent(name_comp_bias), i, 0.0);
  }

  // CV values
  vector<float> cvs = this->tensor_to_vector(output);
  getPntrToComponent("node-0")->set(cvs[0]);

  // Bias values
  vector<float> bias = this->tensor_to_vector(log_grad_sq);
  getPntrToComponent("bias-0")->set(bias[0]);
  
  // Periodic logging of rebuild statistics
  if (_total_steps % 10000 == 0 && _total_steps > 0) {
    double rebuild_fraction = (double)_rebuild_count / (double)_total_steps * 100.0;
    log.printf("Verlet stats (NPT): %lu rebuilds / %lu steps (%.1f%%)\n", 
               _rebuild_count, _total_steps, rebuild_fraction);
  }
}


} // pytorch
} // function
} // PLMD

#endif // __PLUMED_HAS_LIBTORCH
