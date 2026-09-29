#include <cuda_runtime.h>
#include <mutex>
static unsigned long long* counter_ptr=nullptr;
static std::once_flag initialize_counts;
static int allocation_error=0;
__global__ void project(const float* x,const float* q,const unsigned char* faces,float* out,float* ratio,int B,unsigned long long* counts){
 int b=blockIdx.x;if(b>=B)return;bool side=faces[b]>=2;
 __shared__ unsigned mask;
 __shared__ float p[529],r[529],s0[256],s1[256];
 if(threadIdx.x==0)mask=3;
 for(int k=threadIdx.x;k<529;k+=blockDim.x)p[k]=q[b*529+k];
 __syncthreads();
 unsigned local=3;
 for(int k=threadIdx.x;k<12167 && local;k+=blockDim.x){
  int z=k/529,y=k/23%23,t=k%23;float v=x[b*12167+k];
  if((local&1) && v!=x[b*12167+z*529+y*23+22-t])local&=~1u;
  int reflected=side ? (22-z)*529+y*23+t : z*529+(22-y)*23+t;
  if((local&2) && v!=x[b*12167+reflected])local&=~2u;
 }
 atomicAnd(&mask,local);__syncthreads();
 if(threadIdx.x==0)atomicAdd(&counts[mask],1ULL);
 float a=0,c=0;
 for(int k=threadIdx.x;k<529;k+=blockDim.x){
  int y=k/23,t=k%23;float v=p[k],sign=side?-1.f:1.f;
  if(mask==1)v=(p[k]+p[y*23+22-t])*.5f;
  else if(mask==2)v=(p[k]+sign*p[(22-y)*23+t])*.5f;
  else if(mask==3)v=((p[k]+p[y*23+22-t])+sign*(p[(22-y)*23+t]+p[(22-y)*23+22-t]))*.25f;
  r[k]=v;a+=fabsf(p[k]);c+=fabsf(v);
 }
 s0[threadIdx.x]=a;s1[threadIdx.x]=c;__syncthreads();
 for(int d=128;d;d/=2){if(threadIdx.x<d){s0[threadIdx.x]+=s0[threadIdx.x+d];s1[threadIdx.x]+=s1[threadIdx.x+d];}__syncthreads();}
 if(threadIdx.x==0)ratio[b]=mask ? (s0[0]>0 ? s1[0]/s0[0] : 0.f) : 1.f;
 for(int k=threadIdx.x;k<529;k+=blockDim.x)out[b*529+k]=s1[0]>0?r[k]:p[k];
}
extern "C" int parity_launch(const float* x,const float* q,const unsigned char* faces,float* out,float* ratio,int B,void* stream){
 std::call_once(initialize_counts,[]{allocation_error=(int)cudaMallocManaged(&counter_ptr,4*sizeof(unsigned long long));if(!allocation_error)for(int k=0;k<4;++k)counter_ptr[k]=0;});
 if(allocation_error)return allocation_error;
 if(B)project<<<B,256,0,(cudaStream_t)stream>>>(x,q,faces,out,ratio,B,counter_ptr);return (int)cudaGetLastError();
}
extern "C" int projection_counts(unsigned long long* out){
 auto e=cudaDeviceSynchronize();if(e!=cudaSuccess)return (int)e;
 if(!counter_ptr){for(int k=0;k<4;++k)out[k]=0;return 0;}
 return (int)cudaMemcpy(out,counter_ptr,4*sizeof(unsigned long long),cudaMemcpyDeviceToHost);
}

#include "selector.cuh"
