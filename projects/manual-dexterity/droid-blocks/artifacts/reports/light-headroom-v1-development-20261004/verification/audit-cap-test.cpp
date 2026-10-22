#define main droid_headroom_auditor_entry
#include "audit-light-headroom.cpp"
#undef main

int main() {
    Audit audit;
    audit.check(true, "valid");
    require(audit.failed_checks == 0 && audit.issues.empty(), "valid check changed error accounting");
    for (std::size_t i = 0; i < 1000; ++i) audit.check(false, "retained error");
    require(audit.failed_checks == 1000 && audit.issues.size() == 1000, "display cap accounting differs");
    const auto before = audit.failed_checks;
    audit.check(false, "error after display cap");
    require(audit.failed_checks == before + 1 && audit.issues.size() == 1000, "capped diagnostics hid a failed check");
    audit.check(true, "valid after cap");
    require(audit.failed_checks == 1001, "valid check incremented error count");
    std::cout << "Audit issue cap keeps monotonically increasing failed-check accounting.\n";
}
