#include <unsupported/Eigen/CXX11/Tensor>
#include "nms/cpu_nms.h"

int main()
{
    Tensor2f d(4,6);
    d.setValues({{1,1,1,1,1,1},{2,2,2,2,2,2},{3,3,3,3,3,3},{4,4,4,4,4,4}});
    float thresh = 1;
    Tensor1i k;
    k = cpu_nms(d,thresh);
	std::cout << k << std::endl;
    return 0;
}
