#ifndef _GET_BOXES_GRID_H
#define _GET_BOXES_GRID_H

#include <iostream>
#include <string>
#include <vector>
#include <unsupported/Eigen/CXX11/Tensor>

typedef Eigen::Tensor<float, 2, Eigen::RowMajor> Tensor2f;
typedef Eigen::Tensor<float, 1, Eigen::RowMajor> Tensor1f;

class get_boxes_grid {
public:
    get_boxes_grid(float image_height, float image_width);
    ~get_boxes_grid();
    void boxes();

// private:
    int height,width;
    int num;
    int num_aspect;
    int nums;
    
    Tensor2f boxes_grid;

    
    std::string NET_NAME;
    float ASPECTS[5] = {1.25,1,0.75,0.5,0.25};
    int KERNEL_SIZE;
    float SPATIAL_SCALE;

};

#endif