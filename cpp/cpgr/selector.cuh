static unsigned long long* joint_counter=nullptr;
static std::once_flag joint_init;
static int joint_error=0;
__global__ void joint_selector(const float* x,const float* fpw,float* out,int B,unsigned long long* counts){
 int b=blockIdx.x;if(b>=B)return;__shared__ unsigned mask;
 if(threadIdx.x==0)mask=7;__syncthreads();unsigned local=7;
 for(int k=threadIdx.x;k<12167 && local;k+=blockDim.x){
  int z=k/529,y=k/23%23,t=k%23;float v=x[b*12167+k];
  if((local&1) && v!=x[b*12167+z*529+y*23+22-t])local&=~1u;
  if((local&2) && v!=x[b*12167+z*529+(22-y)*23+t])local&=~2u;
  if((local&4) && v!=x[b*12167+(22-z)*529+y*23+t])local&=~4u;
 }
 atomicAnd(&mask,local);__syncthreads();
 if(threadIdx.x==0){
  for(int k=0;k<7;++k)out[b*7+k]=fpw[b*7+k];
  if(mask&1)out[b*7+4]=out[b*7+5]=(fpw[b*7+4]+fpw[b*7+5])*.5f;
  if(mask&2)out[b*7+2]=out[b*7+3]=(fpw[b*7+2]+fpw[b*7+3])*.5f;
  if(mask&4)out[b*7+0]=out[b*7+1]=(fpw[b*7+0]+fpw[b*7+1])*.5f;
  atomicAdd(&counts[mask],1ULL);
 }
}
extern "C" int joint_selector_launch(const float* x,const float* fpw,float* out,int B,void* stream){
 std::call_once(joint_init,[]{joint_error=(int)cudaMallocManaged(&joint_counter,8*sizeof(unsigned long long));if(!joint_error)for(int k=0;k<8;++k)joint_counter[k]=0;});
 if(joint_error)return joint_error;
 if(B)joint_selector<<<B,256,0,(cudaStream_t)stream>>>(x,fpw,out,B,joint_counter);
 return (int)cudaGetLastError();
}
extern "C" int joint_counts(unsigned long long* out){
 auto e=cudaDeviceSynchronize();if(e!=cudaSuccess)return (int)e;
 if(!joint_counter){for(int k=0;k<8;++k)out[k]=0;return 0;}
 return (int)cudaMemcpy(out,joint_counter,8*sizeof(unsigned long long),cudaMemcpyDeviceToHost);
}
