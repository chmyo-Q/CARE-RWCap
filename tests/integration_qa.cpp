#include "dnn_solver.h"
#include <dlfcn.h>
#include <fstream>
#include <cstdlib>
struct QA:DNNSolverGrad{using DNNSolver::setCurrentStream;};
int main(){
 torch::InferenceMode guard;QA model;model.loadModels();model.setCurrentStream();
 using Fn=at::Tensor(*)(void*,at::Tensor&,at::Tensor&,at::Tensor&);
 auto sample=(Fn)dlsym(RTLD_DEFAULT,"_ZN13DNNSolverGrad14sampleGradientERN2at6TensorES2_S2_");TORCH_CHECK(sample,"sample missing");
 // With CPGR disabled, compare the interposed entry against the upstream entry.
 auto upstream=dlopen("libdnnsolver.so",RTLD_NOW);
 auto original=(Fn)dlsym(upstream,"_ZN13DNNSolverGrad14sampleGradientERN2at6TensorES2_S2_");
 TORCH_CHECK(original && original!=sample,"upstream/interposed entries not distinct");
 setenv("S29_GRADIENT_PARITY_ENABLE","0",1);
 for(int ax:{0,1,2}){
  auto opt=at::TensorOptions().dtype(at::kFloat).device(at::kCUDA);
  at::manual_seed(2029);auto x=.2+.8*at::rand({8,1,1,23,23,23},opt);
  auto axis=at::full({8},ax,opt.dtype(at::kByte)),mv=at::ones({8},opt);
  auto xx=x.clone();sample(&model,xx,mv,axis);
  at::manual_seed(2029);xx=x.clone();auto a=sample(&model,xx,mv,axis).to(at::kCPU).clone();
  at::manual_seed(2029);xx=x.clone();auto b=original(&model,xx,mv,axis).to(at::kCPU).clone();
  TORCH_CHECK(at::equal(a,b),"disabled CPGR differs from upstream");
 }
 int checked=0,changed=0;setenv("S29_GRADIENT_PARITY_ENABLE","1",1);
 for(int B:{1,8,128,2048})for(int sym:{0,1})for(int ax:{0,1,2}){
  at::manual_seed(914+B);auto opt=at::TensorOptions().dtype(at::kFloat).device(at::kCUDA);
  auto x=.3+.7*at::rand({B,1,1,23,23,23},opt);
  if(sym){auto t=at::arange(-11,12,opt).square()/121;x=(.2+.1*t.view({1,1,1,23,1,1})+.2*t.view({1,1,1,1,23,1})+.3*t.view({1,1,1,1,1,23})).expand({B,1,1,23,23,23}).contiguous();}
  auto axis=at::full({B},ax,opt.dtype(at::kByte)),mv=at::ones({B},opt);auto xx=x.clone();setenv("S29_GRADIENT_JOINT_ENABLE","0",1);sample(&model,xx,mv,axis);
  at::manual_seed(1914);xx=x.clone();auto a=sample(&model,xx,mv,axis).to(at::kCPU).clone();
  at::manual_seed(1914);xx=x.clone();setenv("S29_GRADIENT_JOINT_ENABLE","1",1);auto b=sample(&model,xx,mv,axis).to(at::kCPU).clone();
  if(!sym)TORCH_CHECK(at::equal(a,b),"asymmetric identity changed");
  TORCH_CHECK(at::isfinite(b).all().item<bool>(),"nonfinite");
  TORCH_CHECK(at::equal(a.select(1,4),b.select(1,4)),"center dielectric changed");
  changed+=(!at::equal(a,b));checked+=B;
 }
 TORCH_CHECK(changed>0,"joint selector not affecting sampling");
 std::ofstream f("INTEGRATION_QA.json");f<<"{\"pass\":true,\"cpgr_off_matches_upstream\":true,\"checked_rows\":"<<checked<<",\"asymmetric_bitwise_identity\":true,\"changed_symmetric_batches\":"<<changed<<"}\n";
}
