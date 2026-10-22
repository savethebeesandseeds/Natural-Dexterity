#include "droid/pretrained_session.hpp"
#include <iostream>
#include <stdexcept>
#include <string>
using Json = nlohmann::json;
void check(bool value,const char* message) { if (!value) throw std::runtime_error(message); }
template<class F> void rejected(F&& call) {
    bool failed = false;
    try { call(); } catch (const std::exception&) { failed = true; }
    check(failed,"Invalid control accepted");
}
Json command(const Json& state,const std::string& action) {
    return Json{{"action",action},{"expected_session_id",state.at("session_id")}};
}
int main() {
    try {
        droid::PretrainedSession session, paired;
        const Json initial = session.state();
        check(initial.at("status") == "ready" && initial.at("elapsed_s") == 0,"Not initially paused");
        check(initial.at("history_frames") == 1 && initial.at("history").empty(),"Initial local stack incorrect");
        const Json policy = initial.at("policy");
        rejected([&] { session.control(Json{{"action","wake"}}); });
        rejected([&] { session.control(Json{{"action","wake"},{"expected_session_id",2}}); });
        rejected([&] { session.control(Json{{"action","wake"},{"expected_session_id",1.0}}); });
        auto invalid = command(initial,"rebuild");
        invalid.update(Json{{"body","arbitrary"},{"motor","base"},{"lamp","left"}});
        rejected([&] { session.control(invalid); });
        check(session.state() == initial,"Rejected control mutated initial state");
        session.control(command(initial,"wake"));
        paired.control(command(paired.state(),"wake"));
        session.advance_steps(37);
        check(session.state().at("pending_native_steps") == 2,"Pending macro was not retained");
        const Json paused = session.control(command(session.state(),"pause"));
        session.advance_steps(100);
        check(session.state() == paused,"Paused body advanced or brain changed");
        session.control(command(paused,"wake"));
        session.advance_steps(1263);
        paired.advance_steps(1300);
        check(session.state() == paired.state(),"Pause/resume differs from uninterrupted native evolution");
        check(session.trace() == paired.trace(),"Pause/resume changed action stream or trace");
        check(session.state().at("policy") == policy,"Wake changed pretrained weights/provenance");
        session.control(command(session.state(),"pause"));
        for (const std::string body : {"long_upper","long_lower","weighted"})
            for (const std::string motor : {"base","elbow"})
                for (const std::string lamp : {"left","right"}) {
                    const Json before = session.state();
                    auto rebuild = command(before,"rebuild");
                    rebuild.update(Json{{"body",body},{"motor",motor},{"lamp",lamp}});
                    const Json fresh = session.control(rebuild);
                    check(fresh.at("session_id").get<int>() == before.at("session_id").get<int>()+1,"Rebuild did not invalidate stale controls");
                    check(fresh.at("policy") == policy,"Rebuild changed pretrained policy");
                    check(fresh.at("elapsed_s") == 0 && fresh.at("history").empty() &&
                        fresh.at("history_frames") == 1 && fresh.at("pending_native_steps") == 0,"Rebuild kept old-body observations");
                    rejected([&] { session.control(command(before,"wake")); });
                    check(session.state() == fresh,"Stale control mutated rebuilt body");
                    session.control(command(fresh,"wake"));
                    const Json running = session.state();
                    auto running_rebuild = rebuild;
                    running_rebuild["expected_session_id"] = running.at("session_id");
                    rejected([&] { session.control(running_rebuild); });
                    check(session.state() == running,"Rejected running rebuild changed state");
                    session.advance_steps(3200);
                    const Json finished = session.state();
                    check(finished.at("status") == "completed","Canonical demo run failed safely completing");
                    check(finished.at("native_steps") == 3200 && finished.at("elapsed_s") == 64 &&
                        finished.at("macro_decisions") == 640 && finished.at("history").size() == 640,"Episode budget incorrect");
                    check(finished.at("policy") == policy,"Complete run changed weights");
                    session.advance_steps(5);
                    rejected([&] { session.control(command(finished,"wake")); });
                    check(session.state() == finished,"Completed body restarted or advanced automatically");
                }
        std::cout << "Frozen policy, 12 supported cases, atomic rebuild, stale controls, pending pause continuity and exact budget passed\n";
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
