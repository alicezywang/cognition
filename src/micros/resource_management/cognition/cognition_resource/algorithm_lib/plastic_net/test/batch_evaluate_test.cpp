#include "plastic_net/plastic_net_model.h"

int main(int argc, char *argv[])
{
  const DefaultParams params;
  machine_learning::PlasticNetModel model(tensorflow::Scope::NewRootScope());
  model.batch_evaluate(params.nbiter_test);
  return 0;
}
