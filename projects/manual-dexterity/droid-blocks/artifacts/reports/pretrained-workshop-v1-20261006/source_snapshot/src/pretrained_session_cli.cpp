#include "droid/pretrained_session.hpp"
#include <iostream>
#include <stdexcept>
#include <string>
using Json = nlohmann::json;
int main() {
    try {
        droid::PretrainedSession session;
        std::string line;
        while (std::getline(std::cin,line)) {
            try {
                if (line.size() > 8192) throw std::invalid_argument("Request too large");
                const Json request = Json::parse(line);
                const std::string op = request.at("op");
                Json result;
                if (op == "state") result = session.state();
                else if (op == "trace") result = session.trace();
                else if (op == "spec") result = session.specification();
                else if (op == "control") result = session.control(request.at("request"));
                else if (op == "advance") {
                    if (!request.at("steps").is_number_integer() || request.at("steps") < 0)
                        throw std::invalid_argument("Steps must be a nonnegative integer");
                    session.advance_steps(request.at("steps").get<std::size_t>());
                    result = session.state();
                } else if (op == "close") {
                    std::cout << Json{{"ok",true}}.dump() << std::endl;
                    return 0;
                } else throw std::invalid_argument("Unknown operation");
                std::cout << Json{{"ok",true},{"result",result}}.dump() << std::endl;
            } catch (const std::exception& error) {
                std::cout << Json{{"ok",false},{"error",error.what()}}.dump() << std::endl;
            }
        }
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
