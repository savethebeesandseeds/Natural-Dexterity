#include "hinf_controller.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
using Json=nlohmann::json;
using Array=std::vector<std::vector<double>>;
void require(bool value,const char* message){if(!value)throw std::runtime_error(message);}
Array multiply(const Array& a,const Array& b){Array c(a.size(),std::vector<double>(b[0].size()));for(std::size_t i=0;i<a.size();++i)for(std::size_t k=0;k<b.size();++k)for(std::size_t j=0;j<b[0].size();++j)c[i][j]+=a[i][k]*b[k][j];return c;}
Array transpose(const Array& a){Array b(a[0].size(),std::vector<double>(a.size()));for(std::size_t i=0;i<a.size();++i)for(std::size_t j=0;j<a[0].size();++j)b[j][i]=a[i][j];return b;}
double maxabs(const Array& a){double result=0;for(const auto& row:a)for(double value:row)result=std::max(result,std::abs(value));return result;}
double independent_are(const Array& a,const Array& x,const Array& q,const Array& r){const auto atx=multiply(transpose(a),x),xa=multiply(x,a),xrx=multiply(multiply(x,r),x);double residual=0,scale=1;for(std::size_t i=0;i<a.size();++i)for(std::size_t j=0;j<a.size();++j){residual=std::max(residual,std::abs(atx[i][j]+xa[i][j]+q[i][j]+xrx[i][j]));scale=std::max(scale,std::abs(atx[i][j])+std::abs(xa[i][j])+std::abs(q[i][j])+std::abs(xrx[i][j]));}return residual/scale;}
bool finite(const Json& j){if(j.is_number())return std::isfinite(j.get<double>());if(j.is_structured())for(const auto& child:j)if(!finite(child))return false;return true;}
}

int main() {
  try {
    using helicopter::HinfController;namespace rotor=helicopter::rotor;
    const auto design=HinfController::design();require(design.at("passed").get<bool>(),"Synthesis did not pass");require(finite(design),"Nonfinite design artifact");
    require(design.at("dgkf_assumptions").at("passed").get<bool>(),"Generalized-plant DGKF assumptions failed");
    const auto& selected=design.at("selected_design");const auto& matrices=design.at("matrices");const double gamma=selected.at("gamma");
    const Array a=matrices.at("A"),b1=matrices.at("B1"),b2=matrices.at("B2"),c1=matrices.at("C1"),c2=matrices.at("C2"),x=matrices.at("X"),y=matrices.at("Y");
    auto q=multiply(transpose(c1),c1),b=multiply(b1,transpose(b1)),rx=multiply(b2,transpose(b2)),ry=multiply(transpose(c2),c2);
    for(std::size_t i=0;i<a.size();++i)for(std::size_t j=0;j<a.size();++j){rx[i][j]=b[i][j]/(gamma*gamma)-rx[i][j];ry[i][j]=q[i][j]/(gamma*gamma)-ry[i][j];}
    const double rx_residual=independent_are(a,x,q,rx),ry_residual=independent_are(transpose(a),y,b,ry);
    require(rx_residual<1e-7&&ry_residual<1e-7,"Independent ARE reconstruction failed");
    require(selected.at("coupling_spectral_radius").get<double>()<gamma*gamma,"DGKF coupling condition failed");
    require(selected.at("closed_loop_spectral_abscissa").get<double>()<0&&selected.at("sampled_closed_loop_spectral_radius").get<double>()<1,"Continuous or sampled internal stability failed");
    require(selected.at("hinf_norm_upper_bound").get<double>()<gamma,"Bounded-real gain certificate failed");
    require(selected.at("hinf_norm_sampled").get<double>()<=selected.at("hinf_norm_upper_bound").get<double>()*(1+1e-6),"Frequency peak contradicts gain certificate");
    const double entropy=selected.at("entropy_value"),h2=selected.at("h2_norm");
    require(entropy>=h2*h2*(1-1e-7),"Minimum-entropy functional fell below squared H2 norm");
    require(std::abs(selected.at("entropy_frequency_integral").get<double>()-entropy)/entropy<.003,"Logdet frequency integral disagrees with state-space entropy");
    require(std::abs(selected.at("h2_squared_frequency_integral").get<double>()-h2*h2)/(h2*h2)<.003,"Frequency H2 integral disagrees with Lyapunov result");
    require(std::abs(selected.at("large_gamma_entropy").get<double>()-h2*h2)/(h2*h2)<.001,"Fixed closed-map entropy does not approach squared H2 norm at large gamma");
    const Array ac=matrices.at("Acl"),bc=matrices.at("Bcl"),cc=matrices.at("Ccl"),pg=matrices.at("Pgamma"),gram=matrices.at("H2_observability_gramian");
    auto closed_r=multiply(bc,transpose(bc));for(auto& row:closed_r)for(auto& value:row)value/=gamma*gamma;
    require(independent_are(ac,pg,multiply(transpose(cc),cc),closed_r)<1e-7,"Independent bounded-real ARE reconstruction failed");
    require(independent_are(ac,gram,multiply(transpose(cc),cc),Array(ac.size(),std::vector<double>(ac.size())))<1e-7,"Independent H2 Gramian reconstruction failed");
    require(maxabs(matrices.at("D11").get<Array>())==0&&maxabs(matrices.at("Dk").get<Array>())==0,"The entropy map is not strictly proper");
    rotor::Params parameters;auto trim=rotor::initial_state(parameters);trim[2]=2.5;
    const rotor::Reference reference{{trim[0],trim[1],trim[2]},{},{}};
    HinfController controller;auto at_trim=controller.update(trim,reference);
    const auto trim_input=design.at("trim_command").get<rotor::Input<double>>();
    for(int j=0;j<4;++j)require(std::abs(at_trim.command[j]-trim_input[j])<1e-10,"Zero-error trim produced a feedback command");
    auto auxiliary=trim;for(int j=13;j<23;++j)auxiliary[j]+=j<21?.001:5;
    HinfController other;const auto irrelevant=other.update(auxiliary,reference);
    require(at_trim.diagnostics.at("measurement")==irrelevant.diagnostics.at("measurement"),"H-infinity measurements read hidden rotor or thermal states");
    auto displaced=trim;displaced[0]+=.1;displaced[4]=-.05;const auto before=controller.update(displaced,reference);
    const auto after=controller.update(displaced,reference);require(finite(after.diagnostics),"Runtime diagnostics contain nonfinite values");
    double difference=0;for(int j=0;j<4;++j)difference+=std::pow(after.command[j]-trim_input[j],2);
    require(difference>1e-12,"Dynamic output feedback did not react to measured displacement");
    for(int j=0;j<4;++j)require(std::abs(after.command[j]-before.command[j])<=parameters.input_rate_max[j]*HinfController::sample_period+1e-12,"Runtime slew limit failed");
    controller.reset();const auto reset=controller.update(trim,reference);require(reset.command==at_trim.command,"Reset did not restore the deterministic trim realization");
    // Replay the published digital realization independently. With zero
    // acceleration reference and no clipping, the runtime must be exactly the
    // exported central controller, including output-before-state-update timing.
    const Array ad=matrices.at("Ad"),bd=matrices.at("Bd"),ck=matrices.at("Ck");
    const auto scales=design.at("sensor_scales").get<std::vector<double>>();
    const auto input_scales=design.at("control_scales_rad").get<rotor::Input<double>>();
    Array expected(23,std::vector<double>(1));HinfController replay;
    for(int step=0;step<30;++step) {
      auto observation=trim;observation[0]+=.001*std::sin(.2*step);observation[4]+=.001*std::cos(.3*step);
      const auto action=replay.update(observation,reference);require(!action.diagnostics.at("saturated").get<bool>(),"Small-signal digital realization unexpectedly clipped");
      const auto state=action.diagnostics.at("controller_state").get<std::vector<double>>();
      for(int j=0;j<23;++j)require(std::abs(state[j]-expected[j][0])<1e-11,"Runtime controller state differs from published exact-ZOH realization");
      const auto output=multiply(ck,expected);
      for(int j=0;j<4;++j)require(std::abs(action.command[j]-trim_input[j]-input_scales[j]*output[j][0])<1e-11,"Runtime output differs from the central controller Ck realization");
      const auto measurement=action.diagnostics.at("measurement").get<std::vector<double>>();Array measured(12,std::vector<double>(1));
      for(int j=0;j<12;++j)measured[j][0]=measurement[j]/scales[j];
      const auto next=multiply(ad,expected),injection=multiply(bd,measured);
      for(int j=0;j<23;++j)expected[j][0]=next[j][0]+injection[j][0];
    }
    std::cout<<Json{{"passed",true},{"test","DGKF central minimum-entropy synthesis and digital realization"},
      {"independent_are_residual_x",rx_residual},{"independent_are_residual_y",ry_residual},
      {"selected_design",selected},{"design",design},{"runtime_displacement_action",after.diagnostics}}.dump(2)<<'\n';
    return 0;
  }catch(const std::exception& error){std::cout<<Json{{"passed",false},{"error",error.what()}}.dump(2)<<'\n';return 1;}
}
