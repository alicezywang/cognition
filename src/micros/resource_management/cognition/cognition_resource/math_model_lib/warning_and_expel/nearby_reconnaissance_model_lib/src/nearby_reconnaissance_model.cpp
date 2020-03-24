#include <nearby_reconnaissance_model_lib/nearby_reconnaissance_model.h>
#include <warning_and_expel/model_util.h>
#include <math.h>
#include <vector>

namespace nearby_reconnaissance_model{

#define  MIN_AIRCRAFT_INTERVAL 2

bool cameraRectangle(double position_h,
                     double & long_,double & wide_,
                     int long_pix ,int wide_pix ,
                     double l_visual_angle,double w_visual_angle,
                     double min_obj_size){
   wide_ = 2 * position_h * tan(M_PI * w_visual_angle / 360);
   long_ = 2 * position_h * tan(M_PI * l_visual_angle / 360);
   return (wide_/wide_pix  > min_obj_size ) ? false : (long_/long_pix  < min_obj_size );
}

double furthestHight(double & real_l_max,double & real_w_max,
                     int long_pix ,int wide_pix ,
                     double l_visual_angle,double w_visual_angle,
                     double min_obj_size){
    real_l_max = min_obj_size * long_pix;
    real_w_max = min_obj_size * wide_pix;
    double h1 = real_l_max /2 / tan(M_PI * l_visual_angle / 360);
    double h2 = real_w_max /2 / tan(M_PI * w_visual_angle / 360);
    if(h1<h2){
        real_w_max = h1 * tan(M_PI * w_visual_angle / 360) * 2;
        return h1;
    }else {
        real_l_max = h2 * tan(M_PI * l_visual_angle / 360) * 2;
        return h2;
    }



}

// ax+by+c = 0;
void reconPacking(double a,double b,double c,
                     nearRconUnitState & center,
                     double interval,
                     std::vector<nearRconUnitState> &aricraft_positions,
                     int aricraft_number){

    aricraft_positions.clear();
    if(aricraft_number<=0){
        return;
    }
    nearRconUnitState temp = {0};
    if(b==0){
        if(a == 0){
            return;
        }
        if(aricraft_number%2 == 0){
            for(int i=- aricraft_number+1 ;i<aricraft_number;i++){
                if(i==0){
                    continue;
                }
                temp.y = (i*interval / 2) + center.y;
                temp.x = -c/a;
                aricraft_positions.push_back(temp);
            }
        }else{
            //.....
        }
    }else{
        if(aricraft_number%2 == 0){
            for(int i=-aricraft_number+1 ;i<aricraft_number;i+=2){
                if(i==0){
                    continue;
                }
                temp.x = (i*interval / 2) + center.x;
                temp.y = -(temp.x*a + c )/b;
                aricraft_positions.push_back(temp);
            }
        }else{
            //.....
        }
    }
}


double reconCoverage(const std::vector<std::vector<double> > &people_points,
                     const std::vector<nearRconUnitState> &aricraft_positions){
    //.....
    return 0;

}



int theBestPositionsInLine(const std::vector<std::vector<double> > &points,
                            std::vector<std::vector<double> > &posizion_and_frame,
                            double &real_interval,
                            double h,
                            int aricraft_number){

    posizion_and_frame.clear();
    real_interval = 0;


    if(points.size()!=2){
        return -1;
    }

    double long_max,wide_max;
    double highest = furthestHight(long_max,wide_max);
    if(h > 0 ){
        if(h<highest){

               highest = h;
               cameraRectangle(h,long_max,wide_max);

        }
    }

    double a,b,c;
    std::vector< std::vector<double> > _points(points);
    std::vector<nearRconUnitState> aricraft_positions;
    aricraft_positions.clear();

    for(int i = 0 ; i<_points.size();i++){
         warning_expel_model::llToNes(_points[i][0],_points[i][1],points[0][0],points[0][1]);
         // std::cout<<"_point: "<<_points[i][0]<<","<<_points[i][1]<<std::endl;
    }
    nearRconUnitState center;
    center.x = (_points[0][0] + _points[1][0])/2;
    center.y = (_points[0][1] + _points[1][1])/2;

    if(aricraft_number==1){
        aricraft_positions.clear();
        aricraft_positions.push_back(center);
        return 0;
    }

    double leftmost_x = 0;
    double rightmost_x = 0;

    if(_points[0][0] < _points[1][0]){
        leftmost_x = _points[0][0];
        rightmost_x = _points[1][0];
    }else if(abs(_points[0][0] - _points[1][0])<0.001){
        //...the y axis...
        return -2;
    }else{
        leftmost_x = _points[1][0];
        rightmost_x = _points[0][0];
    }

    // 两点确定一条直线的方程为：
    // x(y2-y1) + (x1-x2)y -x1y2 + x2y1 = 0;
    a = _points[1][1] - _points[0][1];
    b = _points[0][0] - _points[1][0];
    c = _points[1][0]*_points[0][1] - _points[0][0]*_points[1][1];



    if(((long_max*aricraft_number) > (rightmost_x-leftmost_x))){
        if((rightmost_x-leftmost_x) < aricraft_number*MIN_AIRCRAFT_INTERVAL){
            reconPacking(a,b,c,center,MIN_AIRCRAFT_INTERVAL,aricraft_positions);
        }else{
            double interval = (rightmost_x-leftmost_x)/aricraft_number;
            reconPacking(a,b,c,center,interval,aricraft_positions);
        }

    }else{
        double max_interval = (rightmost_x-leftmost_x - long_max)/(aricraft_number-1);
        reconPacking(a,b,c,center,max_interval,aricraft_positions);

    }
    std::vector<double> temp;
    double lon;
    double lat;
    if(aricraft_positions.size()>=2){
        real_interval = pow((aricraft_positions[1].x-aricraft_positions[0].x),2) +
                               pow((aricraft_positions[1].y-aricraft_positions[0].y),2) ;
        real_interval = sqrt(real_interval);
    }else {
       real_interval = 0;
    }

    for(int i = 0 ; i<aricraft_positions.size() ; i++){
       temp.clear();
       lon = aricraft_positions[i].x;
       lat = aricraft_positions[i].y;
       // std::cout<<"air_point "<<i<<" : "<<aricraft_positions[i].x<<","<<aricraft_positions[i].y<<std::endl;


       warning_expel_model::nesToLl(lon,lat,points[0][0],points[0][1]);

       //self
       temp.push_back(lon);
       temp.push_back(lat);


       //A
       lon = aricraft_positions[i].x - (long_max/2);
       lat = aricraft_positions[i].y + (wide_max/2);
       warning_expel_model::nesToLl(lon,lat,points[0][0],points[0][1]);
       temp.push_back(lon);
       temp.push_back(lat);
       //B
       lon = aricraft_positions[i].x - (long_max/2);
       lat = aricraft_positions[i].y - (wide_max/2);
       warning_expel_model::nesToLl(lon,lat,points[0][0],points[0][1]);
       temp.push_back(lon);
       temp.push_back(lat);
       //C
       lon = aricraft_positions[i].x + (long_max/2);
       lat = aricraft_positions[i].y - (wide_max/2);
       warning_expel_model::nesToLl(lon,lat,points[0][0],points[0][1]);
       temp.push_back(lon);
       temp.push_back(lat);
       //D
       lon = aricraft_positions[i].x + (long_max/2);
       lat = aricraft_positions[i].y + (wide_max/2);
       warning_expel_model::nesToLl(lon,lat,points[0][0],points[0][1]);
       temp.push_back(lon);
       temp.push_back(lat);


       temp.push_back(highest);


       posizion_and_frame.push_back(temp);
    }
    return 0;



}


void theBestPositions(const std::vector<std::vector<double> > &points,int points_size,
                      std::vector<std::vector<double> > &posizions ,int aricraft_number){
    double long_max,wide_max;
    double highest = furthestHight(long_max,wide_max);
    double a,b;
    std::vector< std::vector<double> > _points;
    _points.assign(points.begin(),points.begin()+points_size);
    if(_points.size()==0){
        return;
    }
    for(int i = 0 ; i<_points.size();i++){
         warning_expel_model::llToNes(_points[i][0],_points[i][1],points[0][0],points[0][1]);
    }
    std::vector<nearRconUnitState> aricraft_positions;
    nearRconUnitState center;
    center.x = _points[0][0];
    center.y = _points[0][1];

    if(aricraft_number==1){
        aricraft_positions.clear();
        aricraft_positions.push_back(center);
        return;
    }

    if(_points.size() == 1){
        reconPacking(0,1,-_points[0][1],center,MIN_AIRCRAFT_INTERVAL,aricraft_positions);
    }else if(warning_expel_model::leastSquare(_points,_points.size(),a,b)){

     //@@@@@@@


    }else{
        double leftmost_x = 0;
        double rightmost_x = 0;
        for(int i=0 ;i<_points.size();i++){
            if(leftmost_x > _points[i][0]){
                leftmost_x = _points[i][0];
            }
            if(rightmost_x < _points[i][0]){
                rightmost_x = _points[i][0];
            }
        }
        center.x = (rightmost_x-leftmost_x)/2 + leftmost_x;
        center.y = a * center.x  + b;
        if(((long_max*aricraft_number) > (rightmost_x-leftmost_x))){
            if((rightmost_x-leftmost_x) < aricraft_number*MIN_AIRCRAFT_INTERVAL){
                reconPacking(a,-1,b,center,MIN_AIRCRAFT_INTERVAL,aricraft_positions);
            }else{
                double interval = (rightmost_x-leftmost_x)/aricraft_number;
                reconPacking(a,-1,b,center,interval,aricraft_positions);
            }

        }else{
            double interval = long_max;
            double max_interval = (rightmost_x-leftmost_x - long_max)/(aricraft_number-1);
            aricraft_positions.clear();
            reconPacking(a,-1,b,center,interval,aricraft_positions);
            double temp_score = reconCoverage(_points,aricraft_positions);
            std::vector<nearRconUnitState> aricraft_positions_save(aricraft_positions);
            double max_score = temp_score;
            while(interval<max_interval){
                interval += 3;
                aricraft_positions.clear();
                reconPacking(a,-1,b,center,interval,aricraft_positions);
                temp_score = reconCoverage(_points,aricraft_positions);
                if(temp_score > max_score){
                   max_score = temp_score;
                   aricraft_positions_save.assign(aricraft_positions.begin(),aricraft_positions.end());
                }

            }
            aricraft_positions.assign(aricraft_positions_save.begin(),aricraft_positions_save.end());

        }

    }



}

}
