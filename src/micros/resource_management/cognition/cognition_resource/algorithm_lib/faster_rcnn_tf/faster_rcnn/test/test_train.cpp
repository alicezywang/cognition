#include "fast_rcnn/train.h"

int main()
{
  extern tensorflow::SessionOptions session_options;
  tensorflow::GPUOptions *g = new tensorflow::GPUOptions();
  g->set_allow_growth(true);
  session_options.config.set_allocated_gpu_options(g);
  fast_rcnn::train_net(Scope::NewRootScope(), fast_rcnn::cfg.DATA_DIR, 0, 101);
  return 0;
}
