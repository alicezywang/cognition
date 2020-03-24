
#include <unsupported/Eigen/CXX11/Tensor>
#include "utils/nms.h"

int main()
{
    Tensor2f d(4,6);
    d.setValues({{1,1,1,1,1,1},{2,2,2,2,2,2,},{3,3,3,3,3,3,},{4,4,4,4,4,4}});
    std::vector<float> a;
    float thresh = 1;
    Eigen::Tensor<int,1> k;
    k = nms(d,thresh);

	std::cout << k << std::endl;
    return 0;
}
