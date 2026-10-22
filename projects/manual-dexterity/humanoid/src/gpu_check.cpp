#include <torch/torch.h>
#include <iostream>
#include <stdexcept>

int main() {
  try {
    if (!torch::cuda::is_available()) throw std::runtime_error("CUDA unavailable in LibTorch");
    auto x = torch::ones({16}, torch::TensorOptions().device(torch::kCUDA));
    const float result = torch::dot(x, x).cpu().item<float>();
    if (result != 16.0f) throw std::runtime_error("CUDA computation failed");
    std::cout << "{\"cuda_available\":true,\"cuda_devices\":" << static_cast<int>(torch::cuda::device_count())
              << ",\"gpu_dot_product\":" << result << ",\"runtime\":\"C++ LibTorch\"}\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
