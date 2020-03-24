#include "plastic_net/plastic_net_model.h"

int main(int argc, char *argv[])
{
  const DefaultParams params;
  machine_learning::PlasticNetModel model(tensorflow::Scope::NewRootScope());
  model.train(0, params.nbiter);
  return 0;
}
