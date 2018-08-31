void check_cuda(cudaError_t err, const char *msg) {
 if (err != cudaSuccess) {
  fprintf(stderr, "%s -- %d: %s\n", msg,
   err, cudaGetErrorString(err));
  exit(0);
 }
}