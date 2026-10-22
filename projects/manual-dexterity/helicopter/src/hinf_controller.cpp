#include "hinf_controller.hpp"

#include <algorithm>
#include <chrono>
#include <complex>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" {
void dgees_(char*,char*,int(*)(double*,double*),int*,double*,int*,int*,double*,double*,double*,int*,double*,int*,int*,int*);
void dgeev_(char*,char*,int*,double*,int*,double*,double*,double*,int*,double*,int*,double*,int*,int*);
void dgesv_(int*,int*,double*,int*,int*,double*,int*,int*);
void dsyev_(char*,char*,int*,double*,int*,double*,double*,int*,int*);
void dtrsyl_(char*,char*,int*,int*,int*,double*,int*,double*,int*,double*,int*,double*,int*);
void zgesv_(int*,int*,std::complex<double>*,int*,int*,std::complex<double>*,int*,int*);
void zgesvd_(char*,char*,int*,int*,std::complex<double>*,int*,double*,std::complex<double>*,int*,std::complex<double>*,int*,std::complex<double>*,int*,double*,int*);
}

namespace helicopter {
namespace {
using Json=nlohmann::json;
using Complex=std::complex<double>;
constexpr double pi=3.14159265358979323846;
constexpr int n=23, ny=12, nu=4, nw=18, nz=27;
constexpr double filter_leak=.04;
const std::array<double,n> state_scale{.5,.5,.5,1,1,1,.1,.1,.1,1,1,1,.05,.05,.05,.1,3,3,.1,.1,.5,.5,.5};
const std::array<double,n> output_weight{3,3,5,1.5,1.5,2,3,3,2,.35,.4,.3,1,1,1,1,.03,.03,.3,.3,1.2,1.2,2};
const rotor::Input<double> control_scale{.05,.08,.08,.10};
const std::array<double,6> process_scale{.8,.8,1.2,1.5,1.5,1.0};

struct Matrix {
  int rows=0,cols=0;std::vector<double> a;
  Matrix()=default;
  Matrix(int r,int c):rows(r),cols(c),a(static_cast<std::size_t>(r*c),0){}
  double& operator()(int r,int c){return a[r+rows*c];}
  double operator()(int r,int c)const{return a[r+rows*c];}
};
Matrix identity(int size){Matrix x(size,size);for(int j=0;j<size;++j)x(j,j)=1;return x;}
Matrix transpose(const Matrix& a){Matrix b(a.cols,a.rows);for(int j=0;j<a.cols;++j)for(int i=0;i<a.rows;++i)b(j,i)=a(i,j);return b;}
Matrix operator+(const Matrix& a,const Matrix& b){if(a.rows!=b.rows||a.cols!=b.cols)throw std::runtime_error("Matrix addition shape");auto c=a;for(std::size_t k=0;k<c.a.size();++k)c.a[k]+=b.a[k];return c;}
Matrix operator-(const Matrix& a,const Matrix& b){if(a.rows!=b.rows||a.cols!=b.cols)throw std::runtime_error("Matrix subtraction shape");auto c=a;for(std::size_t k=0;k<c.a.size();++k)c.a[k]-=b.a[k];return c;}
Matrix operator*(const Matrix& a,double b){auto c=a;for(auto& v:c.a)v*=b;return c;}
Matrix operator*(double b,const Matrix& a){return a*b;}
Matrix operator*(const Matrix& a,const Matrix& b){if(a.cols!=b.rows)throw std::runtime_error("Matrix product shape");Matrix c(a.rows,b.cols);for(int j=0;j<b.cols;++j)for(int k=0;k<a.cols;++k)for(int i=0;i<a.rows;++i)c(i,j)+=a(i,k)*b(k,j);return c;}
double norm(const Matrix& a){double sum=0;for(double v:a.a)sum+=v*v;return std::sqrt(sum);}
double maxabs(const Matrix& a){double value=0;for(double v:a.a)value=std::max(value,std::abs(v));return value;}
bool finite(const Matrix& a){return std::all_of(a.a.begin(),a.a.end(),[](double v){return std::isfinite(v);});}
Matrix block(const Matrix& a,int r,int c,int nr,int nc){Matrix b(nr,nc);for(int j=0;j<nc;++j)for(int i=0;i<nr;++i)b(i,j)=a(r+i,c+j);return b;}
void put(Matrix& a,int r,int c,const Matrix& b){for(int j=0;j<b.cols;++j)for(int i=0;i<b.rows;++i)a(r+i,c+j)=b(i,j);}
Matrix solve(Matrix a,Matrix b){if(a.rows!=a.cols||a.rows!=b.rows)throw std::runtime_error("Linear solve shape");int size=a.rows,rhs=b.cols,info=0;std::vector<int> piv(size);dgesv_(&size,&rhs,a.a.data(),&size,piv.data(),b.a.data(),&size,&info);if(info!=0||!finite(b))throw std::runtime_error("LAPACK linear solve failed");return b;}
Json json_matrix(const Matrix& a){Json rows=Json::array();for(int i=0;i<a.rows;++i){Json row=Json::array();for(int j=0;j<a.cols;++j)row.push_back(a(i,j));rows.push_back(row);}return rows;}
std::vector<Complex> eigenvalues(Matrix a){int size=a.rows,info=0,one=1,lwork=-1;char no='N';double query=0,dummy=0;std::vector<double> real(size),imag(size);dgeev_(&no,&no,&size,a.a.data(),&size,real.data(),imag.data(),&dummy,&one,&dummy,&one,&query,&lwork,&info);lwork=std::max(1,int(query));std::vector<double> work(lwork);dgeev_(&no,&no,&size,a.a.data(),&size,real.data(),imag.data(),&dummy,&one,&dummy,&one,work.data(),&lwork,&info);if(info)throw std::runtime_error("LAPACK eigenvalues failed");std::vector<Complex> result;for(int i=0;i<size;++i)result.emplace_back(real[i],imag[i]);return result;}
double spectral_abscissa(const Matrix& a){double worst=-1e300;for(auto z:eigenvalues(a))worst=std::max(worst,z.real());return worst;}
double spectral_radius(const Matrix& a){double worst=0;for(auto z:eigenvalues(a))worst=std::max(worst,std::abs(z));return worst;}
double minimum_eigenvalue(Matrix a){int size=a.rows,info=0,lwork=-1;char vectors='N',upper='U';double query=0;std::vector<double> values(size);dsyev_(&vectors,&upper,&size,a.a.data(),&size,values.data(),&query,&lwork,&info);lwork=std::max(1,int(query));std::vector<double> work(lwork);dsyev_(&vectors,&upper,&size,a.a.data(),&size,values.data(),work.data(),&lwork,&info);if(info)throw std::runtime_error("LAPACK symmetric eigenvalues failed");return values.front();}
int select_stable(double* real,double*){return *real<0;}
struct Schur {Matrix t,u;std::vector<double> real,imag;int stable=0;};
Schur schur(const Matrix& a,bool sort) {
  int size=a.rows,info=0,lwork=-1,stable=0;char vectors='V',ordered=sort?'S':'N';double query=0;
  Matrix t=a,u(size,size);std::vector<double> real(size),imag(size);std::vector<int> bwork(size);
  dgees_(&vectors,&ordered,select_stable,&size,t.a.data(),&size,&stable,real.data(),imag.data(),u.a.data(),&size,&query,&lwork,bwork.data(),&info);
  lwork=std::max(1,int(query));std::vector<double> work(lwork);t=a;
  dgees_(&vectors,&ordered,select_stable,&size,t.a.data(),&size,&stable,real.data(),imag.data(),u.a.data(),&size,work.data(),&lwork,bwork.data(),&info);
  if(info)throw std::runtime_error("LAPACK ordered Schur decomposition failed");return {t,u,real,imag,stable};
}
struct Riccati {Matrix x;double residual=0,stable_abscissa=0,min_eigenvalue=0,condition=0;};
Riccati are(const Matrix& a,const Matrix& r,const Matrix& q) {
  const int size=a.rows;Matrix h(2*size,2*size);
  put(h,0,0,a);put(h,0,size,r);put(h,size,0,q*-1);put(h,size,size,transpose(a)*-1);
  const auto s=schur(h,true);
  if(s.stable!=size)throw std::runtime_error("Hamiltonian has wrong stable subspace dimension");
  for(double real:s.real)if(std::abs(real)<1e-8)throw std::runtime_error("Hamiltonian approaches imaginary axis");
  const Matrix u1=block(s.u,0,0,size,size),u2=block(s.u,size,0,size,size);
  const Matrix inverse=solve(u1,identity(size));Matrix x=u2*inverse;
  const double condition=norm(u1)*norm(inverse);
  if(condition>1e12||norm(x-transpose(x))>1e-6*(1+norm(x)))throw std::runtime_error("Riccati invariant subspace is ill-conditioned");
  x=(x+transpose(x))*.5;
  const Matrix residual=transpose(a)*x+x*a+q+x*r*x;
  const double relative=norm(residual)/(1+2*norm(a)*norm(x)+norm(q)+norm(x)*norm(r)*norm(x));
  const double minimum=minimum_eigenvalue(x),abscissa=spectral_abscissa(a+r*x);
  if(!finite(x)||relative>1e-8||minimum < -1e-8*(1+norm(x))||abscissa>=-1e-8)
    throw std::runtime_error("Riccati residual, positivity or stabilizing branch check failed");
  return {x,relative,abscissa,minimum,condition};
}
Matrix lyapunov(const Matrix& a,const Matrix& q) {
  auto s=schur(a,false);Matrix c=transpose(s.u)*q*s.u*-1;
  int size=a.rows,sign=1,info=0;double scale=1;char trans='T',normal='N';
  dtrsyl_(&trans,&normal,&sign,&size,&size,s.t.a.data(),&size,s.t.a.data(),&size,c.a.data(),&size,&scale,&info);
  if(info||scale<=0)throw std::runtime_error("Lyapunov solve failed");
  Matrix result=s.u*(c*(1/scale))*transpose(s.u);return (result+transpose(result))*.5;
}
double trace(const Matrix& a){double value=0;for(int j=0;j<std::min(a.rows,a.cols);++j)value+=a(j,j);return value;}
Matrix exponential(Matrix a) {
  const int scaling=std::max(0,int(std::ceil(std::log2(std::max(1.0,norm(a)/.25)))));
  a=a*std::ldexp(1.0,-scaling);Matrix value=identity(a.rows),term=value;
  for(int k=1;k<=80;++k){term=(term*a)*(1.0/k);value=value+term;if(norm(term)<1e-16*(1+norm(value)))break;if(k==80)throw std::runtime_error("Matrix exponential did not converge");}
  for(int k=0;k<scaling;++k)value=value*value;return value;
}
std::vector<double> singular_values(std::vector<Complex> a,int rows,int cols) {
  int info=0,lwork=-1,one=1;char none='N';Complex query,dummy;std::vector<double> values(std::min(rows,cols)),rwork(5*std::min(rows,cols));
  zgesvd_(&none,&none,&rows,&cols,a.data(),&rows,values.data(),&dummy,&one,&dummy,&one,&query,&lwork,rwork.data(),&info);
  lwork=std::max(1,int(query.real()));std::vector<Complex> work(lwork);
  zgesvd_(&none,&none,&rows,&cols,a.data(),&rows,values.data(),&dummy,&one,&dummy,&one,work.data(),&lwork,rwork.data(),&info);
  if(info)throw std::runtime_error("Complex SVD failed");return values;
}
std::vector<double> response(const Matrix& a,const Matrix& b,const Matrix& c,double omega) {
  int size=a.rows,rhs=b.cols,info=0;std::vector<Complex> dynamic(size*size),solution(size*rhs);std::vector<int> piv(size);
  for(int j=0;j<size;++j)for(int i=0;i<size;++i)dynamic[i+size*j]=-a(i,j)+(i==j?Complex(0,omega):Complex(0));
  for(std::size_t k=0;k<b.a.size();++k)solution[k]=b.a[k];
  zgesv_(&size,&rhs,dynamic.data(),&size,piv.data(),solution.data(),&size,&info);
  if(info)throw std::runtime_error("Frequency response solve failed");std::vector<Complex> transfer(c.rows*b.cols);
  for(int j=0;j<b.cols;++j)for(int k=0;k<size;++k)for(int i=0;i<c.rows;++i)transfer[i+c.rows*j]+=c(i,k)*solution[k+size*j];
  return singular_values(std::move(transfer),c.rows,b.cols);
}
double unstable_pbh_margin(const Matrix& a,const Matrix& input) {
  // PBH checks include imaginary-axis modes. Stable unobservable/uncontrollable
  // modes are permitted by the stabilizability/detectability assumptions.
  double margin=1;
  for(const auto eigenvalue:eigenvalues(a))if(eigenvalue.real()>=-1e-8) {
    const int rows=a.rows,cols=a.cols+input.cols;std::vector<Complex> pencil(rows*cols);
    for(int j=0;j<a.cols;++j)for(int i=0;i<rows;++i)pencil[i+rows*j]=(i==j?eigenvalue:Complex(0))-a(i,j);
    for(int j=0;j<input.cols;++j)for(int i=0;i<rows;++i)pencil[i+rows*(a.cols+j)]=input(i,j);
    const auto singular=singular_values(std::move(pencil),rows,cols);
    margin=std::min(margin,singular.back()/std::max(1.0,singular.front()));
  }
  return margin;
}
std::array<double,4> quaternion_multiply(const std::array<double,4>& a,const std::array<double,4>& b) {
  return {a[0]*b[0]-a[1]*b[1]-a[2]*b[2]-a[3]*b[3],a[0]*b[1]+a[1]*b[0]+a[2]*b[3]-a[3]*b[2],
    a[0]*b[2]-a[1]*b[3]+a[2]*b[0]+a[3]*b[1],a[0]*b[3]+a[1]*b[2]-a[2]*b[1]+a[3]*b[0]};
}
rotor::Vec3<double> attitude_error(const rotor::State<double>& x,const rotor::State<double>& trim) {
  auto q=quaternion_multiply({trim[6],-trim[7],-trim[8],-trim[9]},{x[6],x[7],x[8],x[9]});
  double length=0;for(double v:q)length+=v*v;length=std::sqrt(length);if(length<1e-12)throw std::invalid_argument("Invalid measured quaternion");
  for(auto& v:q)v/=length;if(q[0]<0)for(auto& v:q)v=-v;
  const double sine=std::hypot(q[1],q[2],q[3]),factor=sine<1e-9?2:2*std::atan2(sine,q[0])/sine;
  return {factor*q[1],factor*q[2],factor*q[3]};
}
rotor::State<double> displaced(const rotor::State<double>& trim,const std::array<double,20>& delta) {
  auto x=trim;for(int i=0;i<6;++i)x[i]+=delta[i];
  const double angle=std::hypot(delta[6],delta[7],delta[8]);const double f=angle<1e-9?.5:std::sin(angle*.5)/angle;
  const auto q=quaternion_multiply({trim[6],trim[7],trim[8],trim[9]},{std::cos(angle*.5),f*delta[6],f*delta[7],f*delta[8]});
  for(int i=0;i<4;++i)x[6+i]=q[i];for(int i=9;i<20;++i)x[i+1]+=delta[i];return x;
}
std::array<double,20> reduced_derivative(const rotor::State<double>& x,const rotor::Input<double>& input,const rotor::Params& p) {
  const auto d=rotor::derivative(x,input,rotor::Vec3<double>{},p);std::array<double,20> result{};
  for(int i=0;i<6;++i)result[i]=d[i];for(int i=0;i<3;++i)result[6+i]=x[10+i];
  for(int i=9;i<20;++i)result[i]=d[i+1];return result;
}
struct Plant {Matrix a,b1,b2,c1,c2,d12,d21;rotor::State<double> trim;rotor::Input<double> input,vertical_ff;double linearization_check=0;};
Plant make_plant(const rotor::Params& p) {
  Plant g{Matrix(n,n),Matrix(n,nw),Matrix(n,nu),Matrix(nz,n),Matrix(ny,n),Matrix(nz,nu),Matrix(ny,nw),rotor::initial_state(p),{}, {},0};
  for(int i=0;i<nu;++i)g.input[i]=g.trim[13+i];
  const auto at_trim=reduced_derivative(g.trim,g.input,p);double trim_error=0;for(double d:at_trim)trim_error=std::max(trim_error,std::abs(d));
  if(trim_error>1e-7)throw std::runtime_error("Hover linearization point is not mechanical equilibrium");
  for(int column=0;column<20+nu;++column) {
    std::array<double,20> delta{};auto up=g.input,um=g.input;
    const double h=1e-5*(column<20?state_scale[column]:control_scale[column-20]);
    if(column<20)delta[column]=h;else {up[column-20]+=h;um[column-20]-=h;}
    auto plus=reduced_derivative(displaced(g.trim,delta),up,p);for(auto& d:delta)d=-d;
    auto minus=reduced_derivative(displaced(g.trim,delta),um,p);
    for(int row=0;row<20;++row) {
      const double derivative=(plus[row]-minus[row])/(2*h);
      if(column<20)g.a(row,column)=derivative*state_scale[column]/state_scale[row];
      else g.b2(row,column-20)=derivative*control_scale[column-20]/state_scale[row];
    }
  }
  for(int j=0;j<3;++j){g.a(20+j,j)=state_scale[j]/state_scale[20+j];g.a(20+j,20+j)=-filter_leak;}
  for(int j=0;j<6;++j){const int row=j<3?3+j:9+j-3;g.b1(row,j)=process_scale[j]/state_scale[row];}
  for(int j=0;j<n;++j)g.c1(j,j)=output_weight[j]*state_scale[j];
  for(int j=0;j<nu;++j)g.d12(n+j,j)=1;
  for(int j=0;j<ny;++j){g.c2(j,j)=state_scale[j]/HinfController::sensor_scales[j];g.d21(j,6+j)=1;}
  auto pg=p,mg=p;pg.gravity+=.001;mg.gravity-=.001;
  const auto tp=rotor::initial_state(pg),tm=rotor::initial_state(mg);for(int j=0;j<nu;++j)g.vertical_ff[j]=(tp[13+j]-tm[13+j])/.002;
  // Independent directional first-order consistency check in reduced physical coordinates.
  std::array<double,20> deviation{};Matrix normalized(n,1);auto perturbed_input=g.input;Matrix normalized_input(nu,1);
  for(int j=0;j<20;++j){deviation[j]=state_scale[j]*1e-5*std::sin(j+1.0);normalized(j,0)=deviation[j]/state_scale[j];}
  for(int j=0;j<nu;++j){normalized_input(j,0)=1e-5*std::cos(j+1.0);perturbed_input[j]+=control_scale[j]*normalized_input(j,0);}
  const auto truth=reduced_derivative(displaced(g.trim,deviation),perturbed_input,p);const auto linear=g.a*normalized+g.b2*normalized_input;
  for(int j=0;j<20;++j)g.linearization_check=std::max(g.linearization_check,std::abs(truth[j]/state_scale[j]-linear(j,0)));
  return g;
}
struct Central {double gamma=0,rho=0;Riccati x,y;Matrix ak,bk,ck,ac,bc,cc;double closed_abscissa=0;};
Central central(const Plant& g,double gamma) {
  const Matrix q=transpose(g.c1)*g.c1,b=g.b1*transpose(g.b1),input=g.b2*transpose(g.b2),measurement=transpose(g.c2)*g.c2;
  const auto x=are(g.a,b*(1/(gamma*gamma))-input,q),y=are(transpose(g.a),q*(1/(gamma*gamma))-measurement,b);
  const double rho=spectral_radius(x.x*y.x);if(rho>=gamma*gamma*(1-1e-8))throw std::runtime_error("DGKF coupling spectral radius violates gamma squared");
  const Matrix f=transpose(g.b2)*x.x*-1,l=y.x*transpose(g.c2)*-1,z=solve(identity(n)-y.x*x.x*(1/(gamma*gamma)),identity(n));
  const Matrix ak=g.a+g.b2*f+z*l*g.c2+b*x.x*(1/(gamma*gamma)),bk=z*l*-1;
  Matrix ac(2*n,2*n),bc(2*n,nw),cc(nz,2*n);
  put(ac,0,0,g.a);put(ac,0,n,g.b2*f);put(ac,n,0,bk*g.c2);put(ac,n,n,ak);
  put(bc,0,0,g.b1);put(bc,n,0,bk*g.d21);put(cc,0,0,g.c1);put(cc,0,n,g.d12*f);
  const double abscissa=spectral_abscissa(ac);if(abscissa>=-1e-8)throw std::runtime_error("Central closed-loop realization is unstable");
  return {gamma,rho,x,y,ak,bk,f,ac,bc,cc,abscissa};
}
struct Certificate {
  bool passed=false;double upper=0,lower=0,entropy=0,h2=0,br_residual=0,br_abscissa=0,grid_peak=0,integral_entropy=0,integral_h2=0;
  double coarse_entropy=0,coarse_h2=0,tail_entropy_bound=0,tail_h2_bound=0,frequency_maximum=0,large_gamma_entropy=0;
  Matrix entropy_riccati,h2_gramian;Json frequencies=Json::array();
};
Certificate certify(const Central& k,bool frequency_grid) {
  const Matrix b=k.bc*transpose(k.bc),q=transpose(k.cc)*k.cc;
  const auto br=are(k.ac,b*(1/(k.gamma*k.gamma)),q);
  Certificate c;c.entropy=trace(transpose(k.bc)*br.x*k.bc);c.br_residual=br.residual;c.br_abscissa=br.stable_abscissa;c.entropy_riccati=br.x;
  const Matrix gram=lyapunov(k.ac,q);const double h2_squared=trace(transpose(k.bc)*gram*k.bc);c.h2=std::sqrt(std::max(0.0,h2_squared));
  c.h2_gramian=gram;
  double lower=0,upper=k.gamma;
  for(int iteration=0;iteration<30;++iteration) {
    const double trial=.5*(lower+upper);
    try{are(k.ac,b*(1/(trial*trial)),q);upper=trial;}catch(const std::exception&){lower=trial;}
  }
  c.upper=upper;c.lower=lower;c.passed=upper<k.gamma&&c.entropy>=0&&std::isfinite(c.entropy);
  if(frequency_grid) {
    const int count=2001;const double minimum=1e-5,maximum=std::max(1e6,2*norm(k.ac)),delta=std::log(maximum/minimum)/(count-1);
    c.frequency_maximum=maximum;
    double prior_omega=0,prior_entropy=0,prior_h2=0;
    double coarse_omega=0,coarse_entropy=0,coarse_h2=0;
    for(int point=0;point<count;++point) {
      const double omega=minimum*std::exp(delta*point);const auto s=response(k.ac,k.bc,k.cc,omega);
      double entropy=0,h2=0;for(double value:s){const double ratio=value*value/(k.gamma*k.gamma);if(ratio>=1)throw std::runtime_error("Frequency diagnostic contradicts bounded-real certificate");entropy-=k.gamma*k.gamma*std::log1p(-ratio);h2+=value*value;}
      c.grid_peak=std::max(c.grid_peak,s.front());
      c.integral_entropy+=(omega-prior_omega)*.5*(entropy+prior_entropy)/pi;
      c.integral_h2+=(omega-prior_omega)*.5*(h2+prior_h2)/pi;
      prior_omega=omega;prior_entropy=entropy;prior_h2=h2;
      if(point%2==0) {
        c.coarse_entropy+=(omega-coarse_omega)*.5*(entropy+coarse_entropy)/pi;
        c.coarse_h2+=(omega-coarse_omega)*.5*(h2+coarse_h2)/pi;
        coarse_omega=omega;coarse_entropy=entropy;coarse_h2=h2;
      }
      if(point%10==0)c.frequencies.push_back({{"omega_rad_s",omega},{"sigma_max",s.front()}});
    }
    // Resolvent expansion T(s)=CB/s+CA(sI-A)^-1 B/s and Frobenius
    // submultiplicativity give a conservative analytic high-frequency tail.
    const double tail_coefficient=norm(k.cc*k.bc)+norm(k.cc*k.ac)*norm(k.bc)/(maximum-norm(k.ac));
    const double tail_ratio=std::pow(tail_coefficient/(maximum*k.gamma),2);
    c.tail_h2_bound=tail_coefficient*tail_coefficient/(pi*maximum);
    c.tail_entropy_bound=c.tail_h2_bound/(1-tail_ratio);
    const auto large_gamma=are(k.ac,b*(1/(1e4*k.gamma*k.gamma)),q);
    c.large_gamma_entropy=trace(transpose(k.bc)*large_gamma.x*k.bc);
    c.passed=c.passed&&tail_ratio<1&&c.entropy>=h2_squared*(1-1e-7)
      &&std::abs(c.integral_entropy-c.entropy)<.003*c.entropy
      &&std::abs(c.integral_h2-h2_squared)<.003*h2_squared
      &&std::abs(c.coarse_entropy-c.integral_entropy)<.003*c.entropy
      &&std::abs(c.large_gamma_entropy-h2_squared)<.001*h2_squared;
  }
  return c;
}
struct Design {Plant plant;Central central;Certificate certificate;Matrix ad,bd,baw;Json report;};
Design synthesize(const rotor::Params& p) {
  const Plant g=make_plant(p);
  const double stabilizable_b1=unstable_pbh_margin(g.a,g.b1),stabilizable_b2=unstable_pbh_margin(g.a,g.b2),
    detectable_c1=unstable_pbh_margin(transpose(g.a),transpose(g.c1)),detectable_c2=unstable_pbh_margin(transpose(g.a),transpose(g.c2));
  if(std::min({stabilizable_b1,stabilizable_b2,detectable_c1,detectable_c2})<=1e-8)
    throw std::runtime_error("Generalized-plant stabilizability or detectability PBH check failed");
  double lower=0,upper=1;Central candidate;bool found=false;
  for(int i=0;i<24;++i){try{candidate=central(g,upper);found=true;break;}catch(const std::exception&){lower=upper;upper*=2;}}
  if(!found)throw std::runtime_error("No DGKF feasible gamma bracket found");
  for(int i=0;i<30;++i){const double gamma=.5*(lower+upper);try{central(g,gamma);upper=gamma;}catch(const std::exception&){lower=gamma;}}
  const double selected_gamma=1.5*upper;Central chosen=central(g,selected_gamma);Certificate certificate=certify(chosen,true);
  Matrix augmented(n+ny+nu,n+ny+nu);put(augmented,0,0,chosen.ak);put(augmented,0,n,chosen.bk);put(augmented,0,n+ny,g.b2);
  const auto sampled=exponential(augmented*HinfController::sample_period);
  const Matrix ad=block(sampled,0,0,n,n),bd=block(sampled,0,n,n,ny),baw=block(sampled,0,n+ny,n,nu);
  Matrix plant_augmented(n+nu,n+nu);put(plant_augmented,0,0,g.a);put(plant_augmented,0,n,g.b2);
  const auto plant_sampled=exponential(plant_augmented*HinfController::sample_period);
  Matrix sampled_closed(2*n,2*n);put(sampled_closed,0,0,block(plant_sampled,0,0,n,n));put(sampled_closed,0,n,block(plant_sampled,0,n,n,nu)*chosen.ck);
  put(sampled_closed,n,0,bd*g.c2);put(sampled_closed,n,n,ad);
  const double sampled_radius=spectral_radius(sampled_closed);
  const double normalization_error=std::max({norm(transpose(g.d12)*g.d12-identity(nu)),norm(g.d21*transpose(g.d21)-identity(ny)),norm(transpose(g.d12)*g.c1),norm(g.b1*transpose(g.d21))});
  Json selected={{"gamma",selected_gamma},{"hinf_norm_upper_bound",certificate.upper},{"hinf_norm_bisection_lower_endpoint",certificate.lower},
    {"hinf_norm_sampled",certificate.grid_peak},{"hinf_norm_method","Continuous-time bounded-real stabilizing Riccati bisection; frequency grid is diagnostic only"},
    {"h2_norm",certificate.h2},{"entropy_value",certificate.entropy},
    {"entropy_definition","I_gamma(T) = -gamma^2/(2*pi) integral_R log det(I - T(iw)* T(iw)/gamma^2) dw; equals trace(Bcl^T Pgamma Bcl) for the published strictly proper stable map"},
    {"stable",chosen.closed_abscissa<0},{"closed_loop_spectral_abscissa",chosen.closed_abscissa},
    {"sampled_closed_loop_spectral_radius",sampled_radius},{"riccati_residual_x",chosen.x.residual},{"riccati_residual_y",chosen.y.residual},
    {"riccati_min_eigenvalue_x",chosen.x.min_eigenvalue},{"riccati_min_eigenvalue_y",chosen.y.min_eigenvalue},
    {"riccati_stabilizing_abscissa_x",chosen.x.stable_abscissa},{"riccati_stabilizing_abscissa_y",chosen.y.stable_abscissa},
    {"coupling_spectral_radius",chosen.rho},{"coupling_ratio",chosen.rho/(selected_gamma*selected_gamma)},
    {"bounded_real_residual",certificate.br_residual},{"bounded_real_stabilizing_abscissa",certificate.br_abscissa},
    {"entropy_frequency_integral",certificate.integral_entropy},{"h2_squared_frequency_integral",certificate.integral_h2},
    {"entropy_frequency_coarse_integral",certificate.coarse_entropy},{"h2_squared_frequency_coarse_integral",certificate.coarse_h2},
    {"high_frequency_entropy_tail_bound",certificate.tail_entropy_bound},{"high_frequency_h2_squared_tail_bound",certificate.tail_h2_bound},
    {"frequency_integral_maximum_rad_s",certificate.frequency_maximum},{"large_gamma_entropy",certificate.large_gamma_entropy},{"large_gamma_multiplier",100},
    {"frequency_integral_scope","2001-point log-spaced trapezoidal diagnostic, compared with its 1001-point subset. Conservative resolvent high-frequency tail bound; low-frequency truncation remains a diagnostic. This quadrature is not a norm certificate."}};
  Json sweep=Json::array();
  for(double multiplier:{1.05,1.2,1.5,2.0,4.0}) {
    const double gamma=multiplier*upper;
    try{auto k=central(g,gamma);auto c=certify(k,false);sweep.push_back({{"gamma",gamma},{"feasible",c.passed},{"hinf_norm_upper_bound",c.upper},{"hinf_norm_sampled",nullptr},{"h2_norm",c.h2},{"entropy_value",c.entropy},{"reason","DGKF and bounded-real numerical checks passed"}});}
    catch(const std::exception& e){sweep.push_back({{"gamma",gamma},{"feasible",false},{"reason",e.what()}});}
  }
  const bool passed=certificate.passed&&sampled_radius<1&&normalization_error<1e-12&&g.linearization_check<1e-6;
  Json matrices={{"A",json_matrix(g.a)},{"B1",json_matrix(g.b1)},{"B2",json_matrix(g.b2)},{"C1",json_matrix(g.c1)},{"C2",json_matrix(g.c2)},
    {"D11",json_matrix(Matrix(nz,nw))},{"D12",json_matrix(g.d12)},{"D21",json_matrix(g.d21)},{"D22",json_matrix(Matrix(ny,nu))},
    {"Ak",json_matrix(chosen.ak)},{"Bk",json_matrix(chosen.bk)},{"Ck",json_matrix(chosen.ck)},{"Dk",json_matrix(Matrix(nu,ny))},
    {"X",json_matrix(chosen.x.x)},{"Y",json_matrix(chosen.y.x)},
    {"Pgamma",json_matrix(certificate.entropy_riccati)},{"H2_observability_gramian",json_matrix(certificate.h2_gramian)},
    {"Acl",json_matrix(chosen.ac)},{"Bcl",json_matrix(chosen.bc)},{"Ccl",json_matrix(chosen.cc)},
    {"Ad",json_matrix(ad)},{"Bd",json_matrix(bd)}};
  const std::vector<std::string> state_labels{"position_error_x","position_error_y","position_error_z","velocity_error_x","velocity_error_y","velocity_error_z",
    "attitude_log_error_x","attitude_log_error_y","attitude_log_error_z","body_rate_x","body_rate_y","body_rate_z",
    "actual_collective_deviation","actual_longitudinal_cyclic_deviation","actual_lateral_cyclic_deviation","actual_tail_pitch_deviation",
    "main_inflow_deviation","tail_inflow_deviation","longitudinal_flap_deviation","lateral_flap_deviation","position_filter_x","position_filter_y","position_filter_z"};
  auto controller_labels=state_labels,weighted_labels=state_labels;
  for(auto& label:controller_labels)label="risk_sensitive_coordinate_"+label;
  for(auto& label:weighted_labels)label="weighted_"+label;
  for(const char* label:{"normalized_collective_feedback","normalized_longitudinal_feedback","normalized_lateral_feedback","normalized_tail_feedback"})weighted_labels.push_back(label);
  const std::vector<std::string> measurement_labels(state_labels.begin(),state_labels.begin()+ny);
  std::vector<std::string> disturbance_labels{"linear_acceleration_x","linear_acceleration_y","linear_acceleration_z","angular_acceleration_x","angular_acceleration_y","angular_acceleration_z"};
  for(const auto& label:measurement_labels)disturbance_labels.push_back("sensor_noise_"+label);
  Json report={{"passed",passed},{"objective_kind","minimum_entropy_hinf"},{"controller_family","central_dgkf"},
    {"method","Central DGKF minimum-entropy H-infinity output-feedback"},{"free_parameter_Q","0 (central controller)"},
    {"selected_design",selected},{"gamma_sweep",sweep},{"frequency_response",certificate.frequencies},{"matrices",matrices},
    {"gamma_feasibility_bracket",{lower,upper}},{"gamma_selection","1.5 times the feasible upper bracket from30 DGKF bisection steps; gamma is dimensionless for the declared weighted channels"},
    {"state_scales",state_scale},{"state_output_weights",output_weight},{"control_scales_rad",control_scale},
    {"sensor_scales",HinfController::sensor_scales},{"process_acceleration_scales",process_scale},{"position_filter_leak_per_s",filter_leak},
    {"state_labels",state_labels},{"controller_state_labels",controller_labels},{"measurement_labels",measurement_labels},
    {"weighted_output_labels",weighted_labels},{"disturbance_labels",disturbance_labels},
    {"state_coordinate_definition","Physical deviations and position filters equal diag(state_scales) times the normalized plant coordinates; controller coordinates have the central risk-sensitive realization, not an unbiased state-estimate interpretation"},
    {"channel_coordinate_definition","Matrices use normalized y and w. Runtime measurement and measurement_reconstruction use physical error units before division by sensor_scales; weighted_output is dimensionless reconstructed z."},
    {"dgkf_assumptions",{{"passed",true},{"pbh_relative_singular_value_tolerance",1e-8},{"stabilizable_A_B1_margin",stabilizable_b1},
      {"stabilizable_A_B2_margin",stabilizable_b2},{"detectable_C1_A_margin",detectable_c1},{"detectable_C2_A_margin",detectable_c2}}},
    {"normalized_dgkf_assumption_residual",normalization_error},{"linearization_directional_error",g.linearization_check},
    {"trim_state",g.trim},{"trim_command",g.input},{"vertical_reference_feedforward_rad_per_m_s2",g.vertical_ff},
    {"certificate_scope","Numerical continuous-time LTI hover generalized-plant certificate. No nonlinear, saturation, reference-feedforward, gain-scheduling, contact, or sampled H-infinity guarantee."},
    {"sources",{"https://www.doyle.caltech.edu/images/doyle/2/20/TAC1989.pdf","https://arxiv.org/pdf/1403.5020"}}};
  if(!passed)throw std::runtime_error("H-infinity design failed numerical or sampled stability checks: "+selected.dump());
  return {g,chosen,certificate,ad,bd,baw,report};
}
const Design& nominal_design(){static const Design design=synthesize(rotor::Params{});return design;}
}

struct HinfController::Impl {
  rotor::Params p;const Design* design;Matrix state{n,1};rotor::Input<double> previous{};unsigned updates=0,saturated_updates=0;
  explicit Impl(const rotor::Params& parameters):p(parameters),design(&nominal_design()) {
    if(rotor::model_metadata(p)!=rotor::model_metadata(rotor::Params{}))throw std::invalid_argument("H-infinity runtime uses the published fixed nominal design only");
    previous=design->plant.input;
  }
};
HinfController::HinfController(const rotor::Params& p):impl_(std::make_unique<Impl>(p)){}
HinfController::~HinfController()=default;
void HinfController::reset(){reset(impl_->design->plant.input);}
void HinfController::reset(const rotor::Input<double>& held){impl_->state=Matrix(n,1);impl_->previous=held;impl_->updates=impl_->saturated_updates=0;}
HinfResult HinfController::update(const rotor::State<double>& measured,const rotor::Reference& reference,double dt) {
  const auto start=std::chrono::steady_clock::now();auto& s=*impl_;const auto& d=*s.design;
  if(!std::isfinite(dt)||std::abs(dt-sample_period)>1e-10)throw std::invalid_argument("H-infinity controller requires the declared0.01s sample interval");
  for(double value:measured)if(!std::isfinite(value))throw std::invalid_argument("Nonfinite measured state");
  Matrix measurement(ny,1);const auto attitude=attitude_error(measured,d.plant.trim);
  const auto r=rotor::detail::rotation(d.plant.trim);
  const rotor::Vec3<double> requested_acceleration{
    reference.acceleration[0]+s.p.linear_drag*reference.velocity[0]/s.p.mass,
    reference.acceleration[1]+s.p.linear_drag*reference.velocity[1]/s.p.mass,
    reference.acceleration[2]+s.p.linear_drag*reference.velocity[2]/s.p.mass};
  const rotor::Vec3<double> world_rotation{-requested_acceleration[1]/s.p.gravity,requested_acceleration[0]/s.p.gravity,0};
  rotor::Vec3<double> attitude_ff{};for(int i=0;i<3;++i)for(int j=0;j<3;++j)attitude_ff[i]+=r[3*j+i]*world_rotation[j];
  std::array<double,ny> measured_error{};
  for(int i=0;i<3;++i){measured_error[i]=measured[i]-reference.position[i];measured_error[3+i]=measured[3+i]-reference.velocity[i];measured_error[6+i]=attitude[i]-attitude_ff[i];measured_error[9+i]=measured[10+i];}
  for(int i=0;i<ny;++i)measurement(i,0)=measured_error[i]/sensor_scales[i];
  const Matrix normalized_command=d.central.ck*s.state;rotor::Input<double> raw{},applied{};bool saturated=false;Matrix correction(nu,1);
  for(int i=0;i<nu;++i) {
    const double feedforward=d.plant.input[i]+d.plant.vertical_ff[i]*requested_acceleration[2];
    raw[i]=feedforward+control_scale[i]*normalized_command(i,0);
    applied[i]=std::clamp(raw[i],std::max(s.p.input_min[i],s.previous[i]-s.p.input_rate_max[i]*dt),
                              std::min(s.p.input_max[i],s.previous[i]+s.p.input_rate_max[i]*dt));
    saturated=saturated||std::abs(applied[i]-raw[i])>1e-12;
    correction(i,0)=(applied[i]-raw[i])/control_scale[i];
  }
  const Matrix weighted=d.plant.c1*s.state+d.plant.d12*normalized_command;
  const Matrix before=s.state;
  s.state=d.ad*s.state+d.bd*measurement+d.baw*correction;
  if(!finite(s.state))throw std::runtime_error("Nonfinite H-infinity controller state");
  s.previous=applied;++s.updates;s.saturated_updates+=saturated;
  std::vector<double> coordinate_scaled(n);for(int j=0;j<n;++j)coordinate_scaled[j]=before(j,0)*state_scale[j];
  Matrix reconstruction=d.plant.c2*before;for(int j=0;j<ny;++j)reconstruction(j,0)*=sensor_scales[j];
  Json reasons=Json::array();double position_error=0,velocity_error=0,attitude_norm=0,rate_norm=0;
  for(int j=0;j<3;++j){position_error+=measured_error[j]*measured_error[j];velocity_error+=measured_error[3+j]*measured_error[3+j];attitude_norm+=attitude[j]*attitude[j];rate_norm+=measured_error[9+j]*measured_error[9+j];}
  if(std::sqrt(position_error)>.6)reasons.push_back("position_error_exceeds0.6m");
  if(std::sqrt(velocity_error)>1.5)reasons.push_back("velocity_error_exceeds1.5m_per_s");
  if(std::sqrt(attitude_norm)>.35)reasons.push_back("trim_attitude_deviation_exceeds0.35rad");
  if(std::sqrt(rate_norm)>1.0)reasons.push_back("body_rate_exceeds1rad_per_s");
  if(saturated)reasons.push_back("actuator_limit_active");
  const double elapsed=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
  HinfResult result;result.command=applied;
  result.diagnostics={{"available",true},{"applied_controller","minimum_entropy_hinf"},{"status","central_controller_update"},
    {"update_ms",elapsed},{"control_interval_ms",1000*dt},{"controller_state",before.a},{"controller_coordinate_scaled",coordinate_scaled},
    {"measurement",measured_error},{"measurement_reconstruction",reconstruction.a},
    {"measurement_reconstruction_kind","C2 times central controller coordinates, converted to physical error units; not an unbiased state estimate"},
    {"weighted_output",weighted.a},{"weighted_output_kind","Reconstruction C1*xi+D12*Ck*xi from central controller coordinates; not actual plant z"},{"disturbance",nullptr},
    {"raw_command",raw},{"applied_command",applied},{"saturated",saturated},{"updates",s.updates},{"saturated_updates",s.saturated_updates},
    {"local_scope",{{"within_declared_region",reasons.empty()},{"reasons",reasons},{"guarantee","Descriptive local-region check only; not a nonlinear robustness certificate"}}}};
  return result;
}
Json HinfController::design(){return nominal_design().report;}
Json HinfController::description() {
  return {{"name","Central minimum-entropy H-infinity output-feedback controller"},
    {"objective_kind","minimum_entropy_hinf"},{"controller_family","central_dgkf"},
    {"formulation","Mustafa/Glover minimum entropy of the weighted closed-loop transfer map at fixed gamma; DGKF two-Riccati central solution"},
    {"entropy_is_not","Thermodynamic entropy production, a PID tuning cost, or physical-energy NMPC"},
    {"state_order",{"position_error_xyz","velocity_error_xyz","body_attitude_log_error_xyz","body_rate_xyz","actual_pitch_deviation4","inflow_deviation2","flap_deviation2","leaky_position_filter3"}},
    {"measurement_channels","12 noisy position, velocity, trim-relative attitude-log and body-rate channels; auxiliary rotor/thermal states are not read by the feedback law"},
    {"position_weight_filter","eta_dot=position_error-0.04 eta; stable low-frequency weight, not an exact integral-action guarantee"},
    {"runtime","Fixed nominal gains; exact zero-order hold of the continuous controller at100Hz. Sampled closed-loop stability checked separately."},
    {"reference_handling","Subtract position/velocity reference; subtract small-angle acceleration and nominal-drag attitude feedforward about trim; add vertical acceleration trim-sensitivity feedforward"},
    {"limits","Physical pitch and sampled slew limits; observer input correction uses known applied-minus-requested input. Limits/antiwindup are outside the linear certificate."},
    {"scope","Strictly proper frozen LTI hover generalized plant. No claim that its H-infinity bound certifies nonlinear flight, contact, changing mass, noisy sampled implementation, or saturation."}};
}
} // namespace helicopter
