#include "droid/pretrained_policy.hpp"

#include <cmath>
#include <exception>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
using Json = nlohmann::json;

double number(const Json& value) {
    if (!value.is_number()) throw std::invalid_argument("Expected a finite numeric value");
    const double result = value.get<double>();
    if (!std::isfinite(result)) throw std::invalid_argument("Expected a finite numeric value");
    return result;
}

template <std::size_t Size>
std::array<float, Size> vector(const Json& value) {
    if (!value.is_array() || value.size() != Size)
        throw std::invalid_argument("Unexpected input vector dimension");
    std::array<float, Size> result{};
    for (std::size_t index = 0; index < Size; ++index) {
        result[index] = static_cast<float>(number(value[index]));
        if (!std::isfinite(result[index])) throw std::invalid_argument("Input exceeds finite float32 range");
    }
    return result;
}

Json infer(const droid::PretrainedPolicy& policy, const droid::PretrainedInput& input,
           const Json& uniform) {
    const auto logits = policy.logits(input);
    const auto probabilities = policy.probabilities(input);
    return Json{{"logits", logits}, {"probabilities", probabilities},
                {"action", droid::PretrainedPolicy::categorical(probabilities, number(uniform))}};
}

Json call(const droid::PretrainedPolicy& policy, const Json& request) {
    if (!request.is_object() || request.at("schema") != "pretrained_actor_parity_request_v1")
        throw std::invalid_argument("Expected pretrained_actor_parity_request_v1");
    const std::string op = request.value("op", std::string("actor"));
    if (op == "actor")
        return infer(policy, vector<droid::kPretrainedInputSize>(request.at("input")), request.at("uniform"));
    if (op == "categorical")
        return Json{{"action", droid::PretrainedPolicy::categorical(
            vector<droid::kPretrainedActionCount>(request.at("probabilities")), number(request.at("uniform")))}};
    if (op == "frame")
        return Json{{"frame", droid::PretrainedHistory::frame(request.at("observation"),
            number(request.at("executed_effort")), number(request.at("previous_request")), number(request.at("reward")))}};
    if (op == "history") {
        droid::PretrainedHistory history(request.at("initial"));
        const Json& transitions = request.at("transitions");
        if (!transitions.is_array() || transitions.size() > 640)
            throw std::invalid_argument("History requires at most640 completed decision entries");
        for (const auto& entry : transitions)
            history.append(entry.at("observation"), number(entry.at("executed_effort")),
                           number(entry.at("previous_request")), number(entry.at("reward")));
        Json result{{"input", history.values()}, {"count", history.count()}, {"diagnostics", history.diagnostics()}};
        if (request.contains("uniform")) result.update(infer(policy, history.values(), request.at("uniform")));
        return result;
    }
    throw std::invalid_argument("Unknown pretrained parity operation");
}
} // namespace

int main(int argc, char** argv) {
    try {
        std::filesystem::path artifact{"artifacts/pretrained/pooled-seed101-v1.json"};
        for (int index = 1; index < argc; ++index) {
            const std::string argument(argv[index]);
            if (argument == "--artifact") {
                if (++index == argc) throw std::invalid_argument("--artifact requires a path");
                artifact = argv[index];
            } else if (argument == "--parity") {
                // All operations are inference/local-history only. No world exists.
            } else if (argument == "--help" || argument == "-h") {
                std::cout << "Usage: droid-pretrained-actor --artifact PATH --parity\n"
                             "Read pretrained_actor_parity_request_v1 JSON lines; emit logits/probabilities/action,\n"
                             "or op frame/history/categorical diagnostics. No physics or learning is performed.\n";
                return 0;
            } else throw std::invalid_argument("Unknown argument: " + argument);
        }
        const droid::PretrainedPolicy policy(artifact);
        std::string line;
        while (std::getline(std::cin, line)) {
            try {
                if (line.size() > 8 * 1024 * 1024) throw std::invalid_argument("Parity request is too large");
                std::cout << call(policy, Json::parse(line)).dump() << std::endl;
            } catch (const std::exception& error) {
                std::cout << Json{{"error", error.what()}, {"ok", false}}.dump() << std::endl;
            }
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "pretrained actor: " << error.what() << '\n';
        return 2;
    }
}
