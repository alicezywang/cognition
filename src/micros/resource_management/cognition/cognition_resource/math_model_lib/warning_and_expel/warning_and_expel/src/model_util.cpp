#include <warning_and_expel/model_util.h>


namespace warning_expel_model
{



void llToNes(double &longitude, double &latitude,double opoint_long,double opoint_lat )
{
	double R = 6371393;

	// (R * 3.1415926 / 180) = 111201.786
	// pi/180 = 0.01745329
	double t = cos(opoint_lat * 0.01745329);
	double h = (longitude - opoint_long);
	double y = 111201.786 * (latitude-opoint_lat);
	double x = 111201.786 * t * h; 
	longitude = x;
	latitude = y;
}


void nesToLl(double &x, double &y,double opoint_long ,double opoint_lat )
{

	double longitude = x / (111201.786 * cos(opoint_lat * 0.01745329))+ opoint_long;
	double latitude = y/111201.786  + opoint_lat;
	x = longitude;
	y = latitude;
}



void nesToBody(std::vector<std::vector<double> > &_points , Eigen::MatrixXd & A,double &l_x ,double &l_y){
	double dlx = _points[_points.size() - 2][0]  -  _points[_points.size() - 1][0];
  double dly = _points[_points.size() - 2][1]  -  _points[_points.size() - 1][1];
  l_x = pow(dlx*dlx + dly*dly ,0.5);
  dlx = _points[_points.size() - 2][0]  -  _points[_points.size() - 3][0];
  dly = _points[_points.size() - 2][1]  -  _points[_points.size() - 3][1];
  l_y = pow(dlx*dlx + dly*dly ,0.5);

	Eigen::MatrixXd coefficient(4, 4);
	Eigen::Vector4d value(0,l_y,l_x,0);
	coefficient <<_points[_points.size()-3][0],_points[_points.size()-3][1],0,0,
		0,0,_points[_points.size()-3][0],_points[_points.size()-3][1],
		_points[_points.size()-1][0],_points[_points.size()-1][1],0,0,
		0,0,_points[_points.size()-1][0],_points[_points.size()-1][1];
	Eigen::MatrixXd k = coefficient.colPivHouseholderQr().solve(value);
	A<<k(0),k(1),
		k(2),k(3);
	return;
}

/**
* 最小二乘拟合　ｙ　＝　ａｘ　＋　ｂ
*
*@_points  　　　存放顺序为：人的经纬度　正方形四个框的经纬度(逆时针)　边境的经纬度
*             每一个点是一个size为２的std::vector<double>,[0]号元素为经度，[1]号元素为纬度
*@a 　　　　　　　 　　输出，直线方程系数　"a"
*@b  　　　　　　　　　输出，直线方程系数　“ｂ”
*/
int leastSquare(const std::vector< std::vector<double> >& points,double &a ,double &b){
  int size = points.size() - 6;
  return leastSquare(points,size,a,b);

}

int leastSquare(const std::vector< std::vector<double> >& points,int size,double &a ,double &b){
    double t1=0, t2=0, t3=0, t4=0;
    for(int i=0; i<size; ++i){
      t1 += points[i][0] * points[i][0];
      t2 += points[i][0];
      t3 += points[i][0] * points[i][1];
      t4 += points[i][1];

    }
    double low = t1*size - t2*t2;
    if(abs(low) < 0.000001 ){
       return 1;
    }

    a = (t3*size - t2*t4) / low;
    b = (t1*t4 - t2*t3) / low;
    return 0;
}





}
