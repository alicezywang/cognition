#include <iostream>
#include <cmath>
#include <algorithm>
#include <vector>

#include "generate_anchors.h"

int main()
{
    std::vector<std::vector<float> > result = anchors_gen().generate_anchors();
    for (int i = 0; i < result.size(); i++)
    {
        for (int j = 0; j < result[0].size(); j++)
        {
            std::cout << result[i][j] << "  ";
        }
        std::cout << std::endl;
    }
    return 0;
}
