#ifndef NEARBY_RECONNAISSANCE_MODEL_H
#define NEARBY_RECONNAISSANCE_MODEL_H

#include <vector>

namespace nearby_reconnaissance_model {

/*
*
*/

struct nearRconUnitState{
    double x;
    double y;
    double h;
};


bool cameraRectangle(double position_h,
                     double & long_,double & wide_,
                     int long_pix = 320,int wide_pix = 240,
                     double l_visual_angle= 80,double w_visual_angle = 60,
                     double min_obj_size = 0.125);


double furthestHight(double & real_l_max,double & real_w_max,
                     int long_pix = 320 ,int wide_pix = 240 ,
                     double l_visual_angle = 80,double w_visual_angle = 60,
                     double min_obj_size = 0.125);

void theBestPositions(const std::vector<std::vector<double> > &people_points,int points_size,
                      std::vector<std::vector<double> > &posizions,
                      int aricraft_number = 4);

int theBestPositionsInLine(const std::vector<std::vector<double> > &points,
                            std::vector<std::vector<double> > &posizion_and_frame,
                            double &real_interval,
                            double h = -1,
                            int aricraft_number =4);

bool isInRectange(nearRconUnitState &people,
                  nearRconUnitState & left_top,nearRconUnitState & right_bottom);


// ax+by+c = 0;
void reconPacking(double a,double b,double c,
                     nearRconUnitState & center,
                     double interval,
                     std::vector<nearRconUnitState> &aricraft_positions,
                     int aricraft_number=4);

double reconCoverage(const std::vector<std::vector<double> > &people_points,
                     const std::vector<nearRconUnitState> &aricraft_positions);


}


#endif // NEARBY_RECONNAISSANCE_MODEL_H
