
#include <warning_and_expel/low_altitude_suppression.h>
#include <warning_and_expel/model_util.h>

#define MIN_MAGIC_NUMBER 0.000001

namespace warning_expel_model
{





/**
* 直线与矩形的交点，矩形的边应该位于坐标系的正轴
*
*@a  　　　　　　直线的斜率
*@p_x   　　　直线上某一点的ｘ坐标
*@p_ｙ   　　直线上某一点的ｙ坐标
*@lx 　　　　　　矩形的宽
*@ly  　　　　　矩形的高
*@forword  直线与矩形相交的方向，－１为从右到左，１为从左到右
*@p_in   入点坐标
*@p_out  出点坐标
*/
int crossPoint(double a,double p_x,double p_y,double lx,double ly,int forword,double*p_in,double *p_out){
  if (abs(a) <= MIN_MAGIC_NUMBER){
    if(forword > 0 ){
      p_in[0] = 0;
      p_in[1] = p_y;
      p_out[0] = lx;
      p_out[1] = ly;
    }else{
      p_in[0] = lx;
      p_in[1] = p_y;
      p_out[0] = 0;
      p_out[1] = p_y;
    }
    return 0;
  }
  // std::cout << "cross "<< a << "  "<< p_x << "  "<<p_y << " "<<lx << " "<<ly<< " "<<forword<<std::endl;
  double b_ci = p_y - a*p_x;
  double l1_y = a*lx + b_ci;
  double l2_x = (ly - b_ci)/a;
  double l3_y = b_ci;
  double l4_x = -b_ci/a;
  // std::cout<< "cross "<<b_ci<<" "<<l1_y<<" "<<l2_x<<" "<<l3_y<<" "<<l4_x<<std::endl;
  int i = 0;
  int j = 0;
  double cross_0[2];
  double cross_1[2];
  while(!j){
    if((l1_y <= ly) && (l1_y >= 0)){
      // std::cout<<"1"<<std::endl;
      i = 1;
      cross_0[0] = lx;
      cross_0[1] = l1_y;
    }
    if((l2_x <= lx) && (l2_x >= 0)){
      // std::cout<<"2"<<std::endl;
      if(i){
        j = 2;
        cross_1[0] = l2_x;
        cross_1[1] = ly;
        break;
      }else{
        i = 2;
        cross_0[0] = l2_x;
        cross_0[1] = ly;
      }
    }
    if((l3_y <=ly) && (l3_y >= 0)){
      // std::cout<<"3"<<std::endl;
      if(!j){
        if(i){
          j = 3;
          cross_1[0] = 0;
          cross_1[1] = l3_y;
          break;
        }else{
          cross_0[0] = 0;
          cross_0[1] = l3_y;
          i = 3;
        }
      }
    }
    if((l4_x <= lx) && (l4_x >=0 )){
      // std::cout<<"4"<<std::endl;
      if(!j){
        cross_1[0] = l4_x;
        cross_1[1] = 0;
        j = 4;
        break;
      }
    }
    if(forword >0){
      if((cross_0[0] - cross_1[0])>0){
        i = 1;
      }else{
        i = 0;
      }
    }else{
      if((cross_0[0] - cross_1[0])>0){
        i = 0;
      }else{
        i = 1;
      }
    }
  }

  // if i = 1;reverse
  if(i){
    p_in[0] = cross_1[0];
    p_in[1] = cross_1[1];
    p_out[0] = cross_0[0];
    p_out[1] = cross_0[1];
  }else{
    p_in[0] = cross_0[0];
    p_in[1] = cross_0[1];
    p_out[0] = cross_1[0];
    p_out[1] = cross_1[1];
  }
  // std::cout<<p_in[0] << "  "<<p_in[1]<<std::endl;
  // std::cout<<p_out[0] << "  "<<p_out[1]<<std::endl;
  return 0;


}


/**
* 点到直线的距离，直线由两点确定
*
*@p_x     点的坐标ｘ
*@p_ｙ   　　　点的坐标ｙ
*@x1    直线上某一点的坐标ｘ
*@y1    直线上某一点的坐标ｙ
*@x2      直线上某一点的坐标ｘ
*@y2    直线上某一点的坐标ｙ
*
*@out     距离
*/
double disPointToLine(double p_x,double p_y , double x1 , double y1 , double x2 , double y2){
  return abs(p_x*(y2-y1) + p_y*(x1-x2) - x1*y2 + x2*y1)/pow((y2-y1)*(y2-y1)+(x1-x2)*(x1-x2),0.5);
}




int  lowAltitudeSuppression(const std::vector< std::vector<double> > & points , 
                 std::vector< std::vector<double> > & in_and_out, double teams_wide , double height ){
  

  if(points.size() < 7){
    return -1;
  }
  std::vector< std::vector<double> > _points(points);

  //经纬度度转换为东北天，原点是矩形倒数第二个顶点
  for(int i = 0 ; i < _points.size() ; i++){
    llToNes(_points[i][0],_points[i][1],points[points.size() - 4][0],points[points.size() - 4][1]);
//   std::cout <<"i＝　"<<i<<" ; "<< _points[i][0]<< ","<<_points[i][1]<< std::endl;
    
  }
//   std::cout << "............" <<std::endl;
    
  
  Eigen::MatrixXd A(2,2);
  double l_x,l_y;
  std::vector<std::vector<double> > rectangle_point(_points.end()-5,_points.end()-2);


  // 东北天转换为以矩形的第四边为正ｘ轴的坐标系
  nesToBody(rectangle_point,A,l_x,l_y);
  for(int i = 0 ; i < _points.size();i++){
    double x = _points[i][0]*A(0,0) + _points[i][1]*A(0,1);
    double y = _points[i][0]*A(1,0) + _points[i][1]*A(1,1);
    _points[i][0] = x;
    _points[i][1] = y;
//   std::cout <<"i＝　"<<i<<" ; "<< x << ","<<y<< std::endl;

  }

  //　入点，出点
  double p_in[2];
  double p_out[2];

  // y = ax+b
  double a = 0;
  double b = 0;


  //　边境线
  double y2 = _points[points.size()-2][1];
  double y1 = _points[points.size()-1][1];
  double x1 = _points[points.size()-1][0];
  double x2 = _points[points.size()-2][0];

  // 只有一个人的情况
  if(7 == _points.size()){
    a = 0;
    if(abs(y1-y2)<=MIN_MAGIC_NUMBER){
      p_in[0] = _points[0][0];
      p_in[1] = l_y;
      p_out[0] = _points[0][0];
      p_out[1] = 0;
    }else{
      a =  (x1-x2) /(y2-y1);
      crossPoint(a,_points[0][0],_points[0][0],l_x,l_y,1,p_in,p_out);
    }
  }else{
    if(leastSquare(_points,a,b)){
      p_in[0]=_points[0][1];
      p_in[1]= 0;
      p_out[0] = _points[0][1];
      p_out[1] = l_y;
    }else{
      crossPoint(a,0,b,l_x,l_y,1,p_in,p_out);
    }

  }



  
//   std::cout <<"a＝　"<<a<<"  b="<<b<< std::endl;
  // 两点确定一条直线的方程为：
  // x(y2-y1) + (x1-x2)y -x1y2 + x2y1 = 0;
  double long_in = disPointToLine(p_in[0],p_in[1],x1,y1,x2,y2);
  double long_out = disPointToLine(p_out[0],p_out[1],x1,y1,x2,y2);
  if(long_in < long_out){
    double temp_x = p_in[0];
    double temp_y = p_in[1];
    p_in[0] = p_out[0];
    p_in[1] = p_out[1];
    p_out[0] = temp_x;
    p_out[1] = temp_y;
  }

  in_and_out.clear();
  std::vector<double> temp_in_out;
  temp_in_out.push_back(p_in[0]);
  temp_in_out.push_back(p_in[1]);
  in_and_out.push_back(temp_in_out);
  temp_in_out[0] = p_out[0];
  temp_in_out[1] = p_out[1];
  in_and_out.push_back(temp_in_out);
  // std::cout <<"in＝　"<<in_and_out[0][0]<<" ; "<<in_and_out[0][1] << std::endl;
  // std::cout <<"out＝　"<<in_and_out[1][0]<<" ; "<<in_and_out[1][1] << std::endl;
  Eigen::MatrixXd inv_A = A.inverse();
  in_and_out[0][0] = inv_A(0,0)*p_in[0] + inv_A(0,1)*p_in[1];
  in_and_out[0][1] = inv_A(1,0)*p_in[0] + inv_A(1,1)*p_in[1];
  in_and_out[1][0] = inv_A(0,0)*p_out[0] + inv_A(0,1)*p_out[1];
  in_and_out[1][1] = inv_A(1,0)*p_out[0] + inv_A(1,1)*p_out[1];

  nesToLl(in_and_out[0][0], in_and_out[0][1] ,points[points.size() - 4][0],points[points.size() - 4][1]);
  nesToLl(in_and_out[1][0], in_and_out[1][1] ,points[points.size() - 4][0],points[points.size() - 4][1]);
  return 0;



/*
  //==================================================================================
  此段代码是实现平行路径的，但是没有写完整，思路是在垂线上按顺序取点，得到直线，得到交点
  //==================================================================================
  double up_max = 0;
  double down_max = 0;
  int up = 0;
  int down = 0;
  for(int i = 0 ; i<_points.size()-4 ; i++){
    double temp = abs(_points[i][0]*a + b - _points[i][1]);
    if((_points[i][0]*a + b - _points[i][1]) < 0){
      if(up_max < temp){
      up = i;
      up_max = temp;
      }
    }else{
      if(down_max < temp){
        down = i;
        down_max = temp;
      }
    }

  }
  
  int up_times = 0;
  if(up_max <= teams_wide/2){
    up_times = 0;
  }else if(up_max <= teams_wide){
    up_times = 1;
  }else if(up_max > teams_wide){
    up_times = (int)((up_max - teams_wide)/teams_wide);
    up_times += 2;
  }
  int down_times = 0;
  if(down_max <= teams_wide/2){
    down_times = 0;
  }else if(down_max <= teams_wide){
    down_times = 1;
  }else if(down_max > teams_wide){
    down_times = (int)((down_max - teams_wide)/teams_wide);
    down_times += 2;
  }
  
  std::vector< std::vector<double> > perpendicular_point;
  int up_times_save = up_times;
  int down_times_save = down_times;
  std::vector<double> temp;
  temp.push_back(0);
  temp.push_back(0);
  if(up_times >= 1){
    perpendicular_point.push_back(_points[up]);
    up_times--;
    double dl = up_max - teams_wide;
    if(up_times){
      dl /= up_times ;
      
    }
    double l = (teams_wide/2) + (dl/2);
    for(up_times ; up_times>0 ; up_times--){
      double dx = l*a / pow(1+a*a,0.5);
      double dy = - l*abs(a) / pow(1+a*a,0.5);      
      temp[0] = _points[up][0] + dx;
      temp[1] = _points[up][1] + dy;
      perpendicular_point.push_back(temp);
      l = l + dl; 
    }
  }
  temp[0] = 0;
  temp[1] = b;
  perpendicular_point.push_back(temp);

  if(down_times > 1){
    double dl = down_max - teams_wide;
    if(down_times){
      dl /= (down_times-1);
      
    }
    double l = (teams_wide/2) + (dl/2);
    for(down_times ; down_times>1 ; down_times--){
      double dx = l*a / pow(1+a*a,0.5);
      double dy = - l*abs(a) / pow(1+a*a,0.5);      
      temp[0] = _points[up][0] + dx;
      temp[1] = _points[up][1] + dy;
      perpendicular_point.push_back(temp);
      l = l + dl; 
    }
  }
  if(down_times == 1){
    perpendicular_point.push_back(_points[down]);
  }
  int forword = 1;
  if(a>0){
    if(up_times_save%2){
      forword = 1;
    }else{
      forword = -1;
    }
  }
*/


}


}
