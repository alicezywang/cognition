#if GOOGLE_CUDA
#include <iostream>
#include <vector>
#include <algorithm>
#include <unsupported/Eigen/CXX11/Tensor>
#include "nms/gpu_nms.h"

static bool cmp(const std::pair<int, float> &x, const std::pair<int, float> &y)
{
    return x.second > y.second;
}

Tensor1i gpu_nms(Tensor2f dets, float thresh, int device_id)
{
    int boxes_num = dets.dimension(0);
    int boxes_dim = dets.dimension(1);
    int num_out, keep[boxes_num];
    std::vector<int> order;
    std::vector<std::pair<int, float> > scores;
    float sorted_dets[boxes_num * boxes_dim + 1];
    for(int i = 0; i < boxes_num; i++)
    {
        scores.push_back(std::make_pair(i, dets(i, 4)));
    }

    std::sort(scores.begin(), scores.end(), cmp);

    for(std::vector<std::pair<int, float> >::size_type i = 0; i < scores.size(); i++)
    {
        order.push_back(scores[i].first);
    }

    for(int i = 0; i < boxes_num; i++)
    {
        for(int j = 0; j < boxes_dim; j++)
        {
            sorted_dets[i * boxes_dim + j] = dets(order[i], j);
        }
    }

    _nms(keep, &num_out, sorted_dets, boxes_num, boxes_dim, thresh, device_id);

    Tensor1i keepv(num_out);
    for(int i = 0; i < num_out; i++)
    {
        keepv(i) = keep[i];
    }

    return keepv;
}
#endif
