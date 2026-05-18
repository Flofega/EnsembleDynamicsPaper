/* +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Copyright (c) 2022 of Luigi Bonati and Enrico Trizio.
Optimized version with BIAS_STRIDE support.

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
#include "function/ActionRegister.h"

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

class PytorchModelBias :
  public Function
{
  unsigned _n_in;
  unsigned _n_out;
  double lambda;
  double epsilon;
  double log_epsilon;  // Precomputed log(epsilon)
  torch::jit::Module _model;
  torch::Device device = torch::kCPU;
  
  // BIAS_STRIDE optimization: compute expensive second derivative less frequently
  int bias_stride_ = 1;
  long last_bias_step_ = -1;
  std::vector<std::vector<double>> cached_bias_derivs_; // cached derivatives for each output
  std::vector<double> cached_bias_values_;              // cached bias values
  
  // Option to use finite-difference HVP (experimental)
  bool use_fd_hvp_ = false;
  double fd_eps_ = 1e-4;
  
public:
  explicit PytorchModelBias(const ActionOptions&);
  void calculate();
  static void registerKeywords(Keywords& keys);

  std::vector<float> tensor_to_vector(const torch::Tensor& x);
};

PLUMED_REGISTER_ACTION(PytorchModelBias,"PYTORCH_MODEL_BIAS")

void PytorchModelBias::registerKeywords(Keywords& keys) {
  Function::registerKeywords(keys);
  keys.use("ARG");
  keys.add("optional","FILE","Filename of the PyTorch compiled model");
  keys.add("optional","LAMBDA","Prefactor of the bias, default 1");
  keys.add("optional","EPSILON","Numerical regularization term in the logarithm, default 1e-6");
  keys.add("optional","BIAS_STRIDE","Frequency (in steps) to recompute bias derivatives; between updates, cached derivatives are used (default: 1 = every step)");
  keys.addFlag("FD_HVP",false,"Use finite-difference Hessian-vector product instead of create_graph (experimental, may be faster for large models)");
  keys.add("optional","FD_EPS","Finite difference step size for FD_HVP (default: 1e-4)");
  // GPU support
  keys.addFlag("GPU",false,"Use GPU for model inference and gradient computation");
  keys.add("optional","GPU_DEVICE","CUDA device index to use (default: 0). Use this to assign PyTorch to a different GPU than GROMACS.");
  keys.addOutputComponent("node", "default", "Model outputs");
  keys.addOutputComponent("bias", "default", "Bias value based on gradient magnitude");
}

std::vector<float> PytorchModelBias::tensor_to_vector(const torch::Tensor& x) {
  // Must move tensor to CPU before accessing data pointer
  torch::Tensor cpu_tensor = x.to(torch::kCPU).contiguous();
  return std::vector<float>(cpu_tensor.data_ptr<float>(), cpu_tensor.data_ptr<float>() + cpu_tensor.numel());
}

PytorchModelBias::PytorchModelBias(const ActionOptions&ao):
  Action(ao),
  Function(ao)
{
  // Number of inputs of the model
  _n_in = getNumberOfArguments();

  // Parse model name
  std::string fname = "model.ptc";
  parse("FILE", fname);

  // Parse params
  lambda = 1.0;
  parse("LAMBDA", lambda);

  epsilon = 1e-6;
  parse("EPSILON", epsilon);
  log_epsilon = std::log(epsilon);  // Precompute

  // Parse BIAS_STRIDE
  bias_stride_ = 1;
  parse("BIAS_STRIDE", bias_stride_);
  if(bias_stride_ < 1) bias_stride_ = 1;
  
  // Parse FD_HVP option
  parseFlag("FD_HVP", use_fd_hvp_);
  parse("FD_EPS", fd_eps_);

  // Parse GPU flag and device index
  bool use_gpu = false;
  parseFlag("GPU", use_gpu);
  int gpu_device_index = 0;
  parse("GPU_DEVICE", gpu_device_index);
  
  if (use_gpu) {
    if (torch::cuda::is_available()) {
      int num_gpus = torch::cuda::device_count();
      if (gpu_device_index >= num_gpus) {
        plumed_merror("GPU_DEVICE=" + std::to_string(gpu_device_index) + 
                      " but only " + std::to_string(num_gpus) + " GPU(s) available (indices 0-" + 
                      std::to_string(num_gpus-1) + ")");
      }
      device = torch::Device(torch::kCUDA, gpu_device_index);
      log.printf("  GPU acceleration ENABLED on CUDA device %d\n", gpu_device_index);
      if (num_gpus > 1) {
        log.printf("  Available GPUs: %d (use GPU_DEVICE to select a different one)\n", num_gpus);
      }
    } else {
      log.printf("  WARNING: GPU requested but CUDA not available, using CPU\n");
    }
  }

  // Create metadata dict
  std::unordered_map<std::string, std::string> metadata = {
    {"_jit_bailout_depth", ""},
    {"_jit_fusion_strategy", ""}
  };

  // Deserialize the model from file
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
      plumed_merror("Cannot load FILE: '"+fname+"'. Please check that it is a Pytorch compiled model (exported with 'torch.jit.trace' or 'torch.jit.script') and that the Pytorch version matches the LibTorch one ("+version+").");
    } else {
      plumed_merror("The FILE: '"+fname+"' does not exist.");
    }
  }
  checkRead();

  // Move model to target device BEFORE any optimizations
  // This is needed for models with constants/buffers that may not be moved by torch::jit::load
  _model.to(device);

  // Optimize model
  _model.eval();
  
  #ifdef DO_TORCH_FREEZE_HACK
    bool optimize_numerics = true;
    auto out_mod = torch::jit::freeze_module(_model, {});
    auto graph = out_mod.get_method("forward").graph();
    OptimizeFrozenGraph(graph, optimize_numerics);
    _model = out_mod;
  #else
    // Only freeze if we're not using FD_HVP (which needs gradients to work properly)
    if(!use_fd_hvp_) {
      _model = torch::jit::freeze(_model);
    }
  #endif

  // Move model to device AGAIN after freezing/optimization
  // Freezing can embed constants that need to be moved
  _model.to(device);

  #if (TORCH_VERSION_MAJOR == 1 && TORCH_VERSION_MINOR <= 10)
    size_t jit_bailout_depth;
    if (metadata["_jit_bailout_depth"].empty()) {
      jit_bailout_depth = 1;
    } else {
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
        strategy.push_back({fusion_type == "STATIC" ? torch::jit::FusionBehavior::STATIC : torch::jit::FusionBehavior::DYNAMIC, std::stoi(fusion_depth)});
      }
    }
    torch::jit::setFusionStrategy(strategy);
  #endif

  // Don't use optimize_for_inference when we need gradients
  #if (TORCH_VERSION_MAJOR == 1 && TORCH_VERSION_MINOR >= 10)
    if(!use_fd_hvp_) {
      // Note: optimize_for_inference can interfere with gradient computation
      // Commenting out for safety when derivatives are needed
      // _model = torch::jit::optimize_for_inference(_model);
    }
  #endif

  // Check output dimension
  log.printf("Checking output dimension:\n");
  std::vector<float> input_test(_n_in);
  torch::Tensor single_input = torch::tensor(input_test).view({1, _n_in});
  single_input = single_input.to(device);
  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(single_input);
  torch::Tensor output = _model.forward(inputs).toTensor();
  vector<float> cvs = this->tensor_to_vector(output);
  _n_out = cvs.size();

  // Create components of output
  for(unsigned j = 0; j < _n_out; j++) {
    string name_comp = "node-" + std::to_string(j);
    addComponentWithDerivatives(name_comp);
    componentIsNotPeriodic(name_comp);
  }
  
  for(unsigned j = 0; j < _n_out; j++) {
    string name_comp = "bias-" + std::to_string(j);
    addComponentWithDerivatives(name_comp);
    componentIsNotPeriodic(name_comp);
  }

  // Initialize cache
  cached_bias_derivs_.resize(_n_out, std::vector<double>(_n_in, 0.0));
  cached_bias_values_.resize(_n_out, 0.0);

  // Print log
  log.printf("  Device: %s\n", device.is_cuda() ? "CUDA (GPU)" : "CPU");
  log.printf("Number of input: %d \n", _n_in);
  log.printf("Number of outputs: %d \n", _n_out);
  if(bias_stride_ > 1) {
    log.printf("BIAS_STRIDE: %d (bias derivatives updated every %d steps)\n", bias_stride_, bias_stride_);
  }
  if(use_fd_hvp_) {
    log.printf("Using finite-difference HVP with eps=%g\n", fd_eps_);
  }
  log.printf("  Bibliography: ");
  log << plumed.cite("Bonati, Rizzi and Parrinello, J. Phys. Chem. Lett. 11, 2998-3004 (2020)");
  log << plumed.cite("Trizio and Parrinello, J. Phys. Chem. Lett. 12, 8621-8626 (2021)");
  log.printf("\n");
}


void PytorchModelBias::calculate() {
  const long current_step = getStep();
  const bool update_bias_derivs = (bias_stride_ <= 1) || 
                                   (last_bias_step_ < 0) || 
                                   ((current_step - last_bias_step_) >= bias_stride_);

  // Retrieve arguments
  vector<float> current_S(_n_in);
  for(unsigned i = 0; i < _n_in; i++)
    current_S[i] = getArgument(i);

  // Convert to tensor
  torch::Tensor input_S = torch::tensor(current_S).view({1, _n_in}).to(device);
  input_S.set_requires_grad(true);

  // Convert to IValue
  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(input_S);

  // Calculate output (forward pass - always needed)
  torch::Tensor output = _model.forward(inputs).toTensor();

  // Compute gradient of CV (always needed for node derivatives)
  torch::Tensor grad_output = torch::ones({1}).expand({1, 1}).to(device);
  
  // For the first gradient, we need create_graph only if we're updating bias derivatives
  // and not using finite-difference HVP
  const bool need_create_graph = update_bias_derivs && !use_fd_hvp_;
  
  torch::Tensor gradient = torch::autograd::grad(
    {output}, {input_S},
    /*grad_outputs=*/ {grad_output},
    /*retain_graph=*/ true,
    /*create_graph=*/ need_create_graph
  )[0];

  // Set node derivatives (always computed)
  vector<float> der = this->tensor_to_vector(gradient);
  string name_comp = "node-" + std::to_string(0);
  for(unsigned i = 0; i < _n_in; i++)
    setDerivative(getPntrToComponent(name_comp), i, der[i]);

  // Set CV values
  vector<float> cvs = this->tensor_to_vector(output);
  for(unsigned j = 0; j < _n_out; j++) {
    string name_comp = "node-" + std::to_string(j);
    getPntrToComponent(name_comp)->set(cvs[j]);
  }

  // Bias computation - only update derivatives if needed
  if(update_bias_derivs) {
    // Compute bias: lambda * (log(||gradient||^2 + epsilon) - log(epsilon))
    torch::Tensor grad_sq_sum = torch::sum(torch::pow(gradient, 2));
    torch::Tensor log_grad_sq = lambda * (torch::log(grad_sq_sum + epsilon) - log_epsilon);

    // Compute bias derivatives
    torch::Tensor grad_output2 = torch::ones({1}).to(device);
    
    torch::Tensor gradient2;
    if(use_fd_hvp_) {
      // Finite-difference Hessian-vector product
      // d(bias)/dx_k = 2*lambda * sum_j(g_j * H_jk) / (||g||^2 + eps)
      // where H_jk = d^2f/dx_j dx_k
      // We approximate Hv using: Hv ≈ (grad(f, x+eps*v) - grad(f, x-eps*v)) / (2*eps)
      // where v = gradient (the vector we're multiplying with)
      
      // Get gradient as vector for perturbation
      torch::Tensor v = gradient.detach().clone();
      const double grad_sq = grad_sq_sum.item<double>();
      
      // Perturb input: x + eps*v
      torch::Tensor input_plus = (input_S + fd_eps_ * v).detach();
      input_plus.set_requires_grad(true);
      std::vector<torch::jit::IValue> inputs_plus;
      inputs_plus.push_back(input_plus);
      torch::Tensor output_plus = _model.forward(inputs_plus).toTensor();
      torch::Tensor grad_plus = torch::autograd::grad(
        {output_plus}, {input_plus},
        /*grad_outputs=*/ {grad_output},
        /*retain_graph=*/ false,
        /*create_graph=*/ false
      )[0];
      
      // Perturb input: x - eps*v
      torch::Tensor input_minus = (input_S - fd_eps_ * v).detach();
      input_minus.set_requires_grad(true);
      std::vector<torch::jit::IValue> inputs_minus;
      inputs_minus.push_back(input_minus);
      torch::Tensor output_minus = _model.forward(inputs_minus).toTensor();
      torch::Tensor grad_minus = torch::autograd::grad(
        {output_minus}, {input_minus},
        /*grad_outputs=*/ {grad_output},
        /*retain_graph=*/ false,
        /*create_graph=*/ false
      )[0];
      
      // H @ v ≈ (grad_plus - grad_minus) / (2 * eps)
      torch::Tensor Hv = (grad_plus - grad_minus) / (2.0 * fd_eps_);
      
      // d(bias)/dx = 2 * lambda * (H @ g) / (||g||^2 + eps)
      gradient2 = (2.0 * lambda / (grad_sq + epsilon)) * Hv;
      
    } else {
      // Original method: backprop through the gradient using create_graph
      gradient2 = torch::autograd::grad(
        {log_grad_sq}, {input_S},
        /*grad_outputs=*/ {grad_output2},
        /*retain_graph=*/ true,
        /*create_graph=*/ false
      )[0];
    }

    // Cache the results
    vector<float> der2 = this->tensor_to_vector(gradient2);
    for(unsigned i = 0; i < _n_in; i++) {
      cached_bias_derivs_[0][i] = der2[i];
    }
    cached_bias_values_[0] = log_grad_sq.item<double>();
    last_bias_step_ = current_step;
  }

  // Set bias derivatives (from cache or freshly computed)
  string name_comp_bias = "bias-" + std::to_string(0);
  for(unsigned i = 0; i < _n_in; i++) {
    setDerivative(getPntrToComponent(name_comp_bias), i, cached_bias_derivs_[0][i]);
  }

  // Set bias value
  for(unsigned j = 0; j < _n_out; j++) {
    string name_comp = "bias-" + std::to_string(j);
    getPntrToComponent(name_comp)->set(cached_bias_values_[j]);
  }
}

} // pytorch
} // function
} // PLMD

#endif // PLUMED_HAS_LIBTORCH
