#include <warning_and_expel/warning_and_expel.h>

using namespace std;

namespace warning_expel_model
{



double deterrenceBreadth(double UAV_Breadth){
  //威慑宽度计算
  double Deterrence_Breadth=UAV_Breadth;
  return Deterrence_Breadth;
}

double deterrenceHeight(double UAV_Height){
  //威慑高度计算
  double Deterrence_Height=UAV_Height;
  return Deterrence_Height;
}


std::vector <double> singleDeterrence(double UAV_Breadth,
double UAV_Height,std::vector<std::vector<double> > _in_and_out,
const std::vector< std::vector<double> > & points){
	double Breadth=deterrenceBreadth(UAV_Breadth);
  double Height=deterrenceHeight(UAV_Height);
  std::vector< std::vector<double> > _points(points);
  
  std::vector <double> error;
  error.push_back(0);
  if(points.size()<8){
    //cout<<"No People!!!!"<<endl;
     return error;
  }
  else if(_in_and_out.size()<2){
    //cout<<"No in_and_out!!!!"<<endl;
    return error;
  }
  //经纬度度转换为东北天，原点是矩形倒数第二个顶点
  for(int i = 0 ; i < _points.size() ; i++){
    llToNes(_points[i][0],_points[i][1],points[points.size() - 4][0],points[points.size() - 4][1]);
  }
  
  for (int i=0;i<_in_and_out.size() ; i++){
    llToNes(_in_and_out[i][0],_in_and_out[i][1],points[points.size() - 4][0],points[points.size() - 4][1]);
  }

  double dlx = _points[_points.size() - 4][0]  -  _points[_points.size() - 3][0];
  double dly = _points[_points.size() - 4][1]  -  _points[_points.size() - 3][1];
  double l_x = pow(dlx*dlx + dly*dly ,0.5);
  dlx = _points[_points.size() - 4][0]  -  _points[_points.size() - 5][0];
  dly = _points[_points.size() - 4][1]  -  _points[_points.size() - 5][1];
  double l_y = pow(dlx*dlx + dly*dly ,0.5);
  
  Eigen::MatrixXd m(4, 4);
  Eigen::Vector4d t(0,l_y,l_x,0);
  m <<_points[points.size()-5][0],_points[points.size()-5][1],0,0,
		0,0,_points[points.size()-5][0],_points[points.size()-5][1],
		_points[points.size()-3][0],_points[points.size()-3][1],0,0,
		0,0,_points[points.size()-3][0],_points[points.size()-3][1];
	Eigen::MatrixXd k = m.colPivHouseholderQr().solve(t);
	Eigen::MatrixXd A(2,2);
	A<<k(0),k(1),
		k(2),k(3);

	//东北天转换为以矩形的第四边为正ｘ轴的坐标系
	for(int i = 0 ; i < _points.size() ; i++){
		double x = _points[i][0]*A(0,0) + _points[i][1]*A(0,1);
		double y = _points[i][0]*A(1,0) + _points[i][1]*A(1,1);
		_points[i][0] = x;
		_points[i][1] = y;
	}
  for(int i=0;i<_in_and_out.size() ; i++){
    double x = _in_and_out[i][0]*A(0,0) + _in_and_out[i][1]*A(0,1);
		double y = _in_and_out[i][0]*A(1,0) + _in_and_out[i][1]*A(1,1);
		_in_and_out[i][0] = x;
		_in_and_out[i][1] = y;
  }
  
  //已知两点<出点，入点>求直线方程，AX+BY+C=0中A = Y2 - Y1；B = X1 - X2；C = X2*Y1 - X1*Y2
  std::vector<double> ABC;
  ABC.push_back(_in_and_out[1][1]-_in_and_out[0][1]);
  ABC.push_back(_in_and_out[0][0]-_in_and_out[1][0]);
  ABC.push_back(_in_and_out[1][0]*_in_and_out[0][1]-_in_and_out[0][0]*_in_and_out[1][1]);
  //double x;
  //double y=(-ABC[0]*x-ABC[2])/ABC[1];
  
  std::vector <double> distance;
  //计算每个point与中心点运动直线的距离
  //cout<<"point.size:"<<_points.size()-6<<endl;
  for(int i = 0 ; i < _points.size()-6 ; i++){
    distance.push_back(fabs( ABC[0] * _points[i][0]+ABC[1] * _points[i][1] + ABC[2] ) / sqrt( pow(fabs(ABC[0]) , 2) + pow(fabs(ABC[1]) , 2) ));
  }

  std::vector <double> num_Deterrence;
  
  //判断point是否在威慑范围内，若在范围内该点威慑值+1
  for(int i = 0 ; i < distance.size() ; i++){
    if( distance[i] <= Breadth/2 )
      num_Deterrence.push_back(1);
      else
      num_Deterrence.push_back(0);
  }

   for(int i = 0 ; i < _points.size()-6 ; i++){
     //cout<<"id:"<<i<<"   distance:"<<distance[i]<<endl;
     //cout<<"num_Deterrence　id:"<<i<<"   "<<num_Deterrence[i]<<endl;
  }
  /*cout<<"Breadth:"<<Breadth/2<<endl;*/
  //debug情况下可以输出

   //计算威慑度
  double count=0;
  for(int i = 0 ; i < _points.size() - 6 ; i++){
     if(num_Deterrence[i] != 0)
      count = count + 1;
  }
  double Degree_of_Deterrence = count / (_points.size() - 6) *100;

  num_Deterrence.push_back(Degree_of_Deterrence);
  return num_Deterrence;
}

std::vector <double> groupDeterrence(double UAV_Breadth,double UAV_Height,int UAV_number,const std::vector<std::vector<double> > in_and_out,const std::vector< std::vector<double> > & points){
    
  std::vector <double> error;
  error.push_back(0);
  if(points.size()<8){
    //cout<<"No People!!!!"<<endl;
     return error;
  }
  else if(in_and_out.size()<2){
    //cout<<"No in_and_out!!!!"<<endl;
    return error;
  }
  std::vector< std::vector<double> > _in_and_out(in_and_out);
  std::vector< std::vector<double> > in_out;
  std::vector<double> tmp;
  std:: vector <double> single_de;
  for(int i=0;i<points.size();i++){
    single_de.push_back(0);
  }
  //第i+1架无人机的出入点，并计算其威慑度
  for(int i=0 ; i < UAV_number ;i++){
  std::vector< std::vector<double> > in_out;
    tmp.push_back(_in_and_out[2*i][0]);
    tmp.push_back(_in_and_out[2*i][1]);
    in_out.push_back(tmp);
    tmp[0] = _in_and_out[2*i+1][0];
    tmp[1] = _in_and_out[2*i+1][1];
    in_out.push_back(tmp);
    
    cout<<in_out.size()<<endl;
    std::vector<double> num_Deterrence(singleDeterrence(UAV_Breadth,UAV_Height,in_out,points));
    /*for(int i=0;i<num_Deterrence.size();i++){
  cout<<num_Deterrence[i]<<endl;
  }*/
    for(int i=0 ; i < points.size() - 6 ; i++){
      if(num_Deterrence[i]!=0)
        single_de[i] = single_de[i] + 1;
    }
  }
  std:: vector <double> statistics;
  //计算多架无人机的威慑度和被威慑多次的数量
  for(int i=0;i<UAV_number+1;i++){
    statistics.push_back(0);
  }

   for(int i = 0 ; i < points.size() - 6 ; i++){
     statistics[single_de[i]] = statistics[single_de[i]] + 1;
  }

  double Degree_of_Deterrence = 100 - (statistics[0] / (points.size() - 6)) * 100 ;
  //返回值vector的最后一个值为多无人机的威慑度，前面statistics[0]为被威慑0次的数量，statistics[1]为被威慑1次的数量.....
  statistics.push_back(Degree_of_Deterrence);
  return statistics;
}

// int main(int argc, char** argv){
//   double A=52.5;
//   double B=10.0;

// std::vector<double> info;
// info.push_back(A);
// info.push_back(B);
// info.push_back(2.0);

//  std::vector<std::vector<double> > in;
//   std::vector<double> tmp;


//   tmp.push_back(113.007614);
//   tmp.push_back(28.231528);
//   in.push_back(tmp);
//   tmp[0]=113.008405;
//   tmp[1]=28.230773;
//   in.push_back(tmp);
//   tmp[0]=113.007902;
//   tmp[1]=28.231544;
//   in.push_back(tmp);
//   tmp[0]=113.008391;
//   tmp[1]=28.231166;
//   in.push_back(tmp);

//   std::vector< std::vector<double> > point;
//  tmp[0]=113.007650;
//   tmp[1]=28.231470;
//   point.push_back(tmp);
//  tmp[0]=113.007875;
//   tmp[1]=28.231381;
//   point.push_back(tmp);
//  tmp[0]=113.008144;
//   tmp[1]=28.231023;
//   point.push_back(tmp);
//  tmp[0]=113.008216;
//   tmp[1]=28.23123;
//   point.push_back(tmp);
//  tmp[0]=113.007879;
//   tmp[1]=28.231318;
//   point.push_back(tmp);
// //double xxx=Single_Deterrence(A,B,in,point);
// //cout<<xxx<<endl;
//  vector <double> xx(Group_Deterrence(info,in,point));
//   cout<<xx[0]<<"  "<<xx[1]<<"   "<<xx[2]<<"   deterrence: "<<xx[3]<<endl;
//  return 0;
// }

}
