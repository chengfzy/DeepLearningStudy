#include <iostream>
#include <memory>
#include <vector>
#include "torch/script.h"
#include "torch/torch.h"

using namespace std;
using namespace torch::jit;

int main(int argc, char* argv[]) {
    // model files
    string modelFile01{"../../../temp/model01.pt"};

    // CUDA is available
    torch::Device device = torch::kCPU;
    if (torch::cuda::is_available()) {
        device = torch::kCUDA;
    }

    // deserialize the script module from file
    shared_ptr<script::Module> module01 = load(modelFile01);
    if (!module01) {
        cout << "load model from file \"" << modelFile01 << "\" failed" << endl;
    }
    module01->to(device);

    // create inputs
    vector<IValue> inputs;
    inputs.push_back(torch::ones({1, 3, 224, 224}).to(device));

    // execute the model and turn its output to a tensor
    at::Tensor output = module01->forward(inputs).toTensor().cpu();
    cout << "output: " << output.slice(1, 0, 5) << endl;

    return 0;
}