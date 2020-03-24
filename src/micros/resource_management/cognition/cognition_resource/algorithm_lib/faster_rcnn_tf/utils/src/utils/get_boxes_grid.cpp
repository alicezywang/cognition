#include <iostream>
#include <math.h>
#include <unsupported/Eigen/CXX11/Tensor>
#include "utils/get_boxes_grid.h"


get_boxes_grid::get_boxes_grid(float image_height, float image_width)
{
    NET_NAME = "VGGnet";
    KERNEL_SIZE = 5;
    SPATIAL_SCALE = 0.0625;
    if(NET_NAME == "CaffeNet")
    {
        height = floor((image_height * 600 - 1) / 4.0 + 1);
        height = floor((height - 1) / 2.0 + 1 + 0.5);
        height = floor((height - 1) / 2.0 + 1 + 0.5);

        width = floor((image_width * 600- 1) / 4.0 + 1);
        width = floor((width - 1) / 2.0 + 1 + 0.5);
        width = floor((width - 1) / 2.0 + 1 + 0.5);
        std::cout << "height" << height << " " << "width" << width << std::endl;
    }
    else if(NET_NAME == "VGGnet")
    {
        height = floor(image_height * 600 / 2.0 + 0.5);
        height = floor(height / 2.0 + 0.5);
        height = floor(height / 2.0 + 0.5);
        height = floor(height / 2.0 + 0.5);

        width = floor(image_width * 600 / 2.0 + 0.5);
        width = floor(width / 2.0 + 0.5);
        width = floor(width / 2.0 + 0.5);
        width = floor(width / 2.0 + 0.5);
        std::cout << "height" << height << " " << "width" << width << std::endl;
    }
    else
    {
        std::cerr << "The network architecture is not supported in utils.get_boxes_grid!" << std::endl;
    }
    num = height * width;
    num_aspect = sizeof(ASPECTS) /sizeof(ASPECTS[0]);
    //将cfg./*TRAIN.*/ASPECTS复制给aspect
    nums = height * width * num_aspect;
}

get_boxes_grid::~get_boxes_grid()
{

}

void get_boxes_grid::boxes()
{
    Tensor2f center(num * num_aspect,2);
    int key = 0;

    for(int i = 0; i < height; i++)
    {

        for(int j =0; j < width; j++)
        {
            for(int k = 1; k <= num_aspect; k++)
            {    
                center(key,0) = j;
                center(key,1) = i;
                key++;
            }
        }
    }

    int area = KERNEL_SIZE * KERNEL_SIZE;
    Tensor1f widths1(num_aspect);
    Tensor1f heights1(num_aspect);
    Tensor1f widths;
    Tensor1f heights;

    //对应Python中widths和heights的创建
    for(int i = 0; i < num_aspect; i++)
    {
        widths1(i) = sqrt(area / ASPECTS[i]);
        heights1(i) = sqrt(area * ASPECTS[i]);
    }

    //对应Python中的tile()，扩大num倍
    Eigen::array<int,1> tile({num});
    widths = widths1.broadcast(tile);
    heights = heights1.broadcast(tile);

    Tensor2f temp(num * num_aspect, 6);
    for(int i = 0; i < num * num_aspect; i++)
    {
        temp(i,0) = (center(i,0) - widths(i) * 0.5) / SPATIAL_SCALE;
        temp(i,1) = (center(i,0) + widths(i) * 0.5) / SPATIAL_SCALE;
        temp(i,2) = (center(i,1) - heights(i) * 0.5) / SPATIAL_SCALE;
        temp(i,3) = (center(i,1) + heights(i) * 0.5) / SPATIAL_SCALE;
        temp(i,4) = center(i,0);
        temp(i,5) = center(i,1);
    }
    
    boxes_grid = temp;
    
    //前四列对应python版中的boxes_grid
    //第五列对应Python版中的centers[:,0]
    //第六列对应Python版中的centers[:,0]
    //此处的boxes_grid对应的是python版中三个返回值合并
}
