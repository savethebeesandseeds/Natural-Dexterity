#include "droid/construction_session.hpp"
#include <iostream>
#include <stdexcept>
#include <string_view>

namespace {
using Json = nlohmann::json;
void check(bool condition, std::string_view message) {
    if (!condition) throw std::runtime_error(std::string(message));
}
template<class Function> void rejects(Function function) {
    try { function(); } catch (const std::invalid_argument&) { return; }
    catch (const std::logic_error&) { return; }
    throw std::runtime_error("Invalid operation was accepted");
}
}
int main() {
    try {
        droid::ConstructionSession session;
        const Json initial = session.state();
        check(initial.at("status") == "ready", "initial ready");
        for (const Json& bad : {Json::array(), Json{{"action", 2}},
                Json{{"action", "unknown"}}, Json{{"action", "pause"}, {"assembly", nullptr}},
                Json{{"action", "build"}, {"assembly", Json::object()}},
                Json{{"action", "replay"}, {"pattern_id", "unseen"}}}) {
            rejects([&] { (void)session.control(bad); });
            check(session.state() == initial, "Rejected request changed state");
        }
        (void)session.control(Json{{"action", "explore"}});
        session.advance_steps(31);
        const Json paused = session.control(Json{{"action", "pause"}});
        session.advance_steps(50);
        check(session.state() == paused, "pause changed motion or memory");
        (void)session.control(Json{{"action", "play"}});
        const Json before_bad = session.state();
        rejects([&] { (void)session.control(Json{{"action", "build"}, {"assembly", initial.at("assembly")}}); });
        check(session.state() == before_bad, "running build changed state");
        session.advance_steps(2000);
        const Json complete = session.state();
        check(complete.at("status") == "completed", complete.at("error").get<std::string>());
        check(complete.at("elapsed_s") == 30.0, "wrong equal trial horizon");
        check(complete.at("repertoire").size() == 5, "missing motion cards");
        const Json card = complete.at("repertoire").at(0);
        (void)session.control(Json{{"action", "replay"}, {"pattern_id", card.at("pattern_id")}});
        session.advance_steps(400);
        const Json replay = session.state();
        check(replay.at("status") == "completed", replay.at("error").get<std::string>());
        check(replay.at("elapsed_s") == 6.0, "replay horizon");
        check(replay.at("trail") == card.at("tip_trail"), "replay geometry differs");
        check(replay.at("repertoire") == complete.at("repertoire"), "replay erased memory");
        (void)session.control(Json{{"action", "reset"}});
        check(session.state().at("repertoire") == complete.at("repertoire"), "rest erased repertoire");
        Json changed = initial.at("assembly");
        changed["segments"] = Json::array({3, 5});
        const Json built = session.control(Json{{"action", "build"}, {"assembly", changed}});
        check(built.at("assembly") == changed, "build/save assembly changed");
        check(built.at("repertoire").empty(), "rebuild retained old-body evidence");
        check(built.at("assembly_revision").get<int>() == 2, "assembly revision");
        droid::ConstructionSession restored;
        const Json restored_state = restored.control(Json{{"action", "build"}, {"assembly", built.at("assembly")}});
        check(restored_state.at("physics") == built.at("physics"), "saved assembly does not reproduce physics");
        std::cout << "construction session tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n'; return 1;
    }
}
