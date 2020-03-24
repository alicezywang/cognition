#include "utils/get_boxes_grid.h"

int main()
{
    get_boxes_grid box_grid(0.6,0.1);
    box_grid.boxes();
    std::cout << box_grid.boxes_grid << std::endl;
    
    return 0;
}
