#include <plastic_net/imgdata.h>
#include <plastic_net/random.h>
#include <plastic_net/config.h>

using namespace tensorflow;

ImgData::ImgData(std::string ParamDir)
{
    Input_param = ParamDir ;
    filesnumber = 0;
    GetClassinfo();
}

ImgData::~ImgData()
{

}

void ImgData::GetImgPath(std::string Input_param_dir)
{
    struct dirent *ptr;
    DIR *dir;
    dir = opendir(Input_param_dir.data());
    while ((ptr=readdir(dir))!=NULL)
    {
        if(ptr->d_name[0] == '.')
        {
            continue;
        }
        if(int(ptr->d_type)==4)
        {
            filesnumber += 1;
            std::string dir_name = Input_param_dir;
            std::string files_name = ptr->d_name;
            std::string search_path = dir_name+files_name+"/";
            GetImgPath(search_path);
            imagedata.push_back(temp_imagePath);
            temp_imagePath.clear();
        }
        else
        {
            std::string full_path = Input_param_dir+ptr->d_name;
            std::string ImgName = ptr->d_name;
            if(ImgName.find("png"))
            {
                temp_imagePath.push_back(full_path);
                ImgPathList.push_back(full_path);
                ImgClassinfo.push_back(SplitString(full_path,"/"));
            }
        }
    }
    closedir(dir);
}

std::multimap<std::string, std::vector<std::string> > ImgData::GetClassinfo()
{
    this->GetImgPath(Input_param);

    for(int it=0;it<ImgPathList.size();it++)
    {
        ImgPathToClassInfo.insert(std::pair<std::string, std::vector<std::string> >(ImgPathList[it],ImgClassinfo[it]));
    }

    return ImgPathToClassInfo;
}

std::vector<std::string> ImgData::SplitString(const std::string &s, const std::string &c)
{
    std::vector<std::string> v;
    std::string::size_type pos1, pos2;
    pos2 = s.find(c);
    pos1 = 0;
    while(std::string::npos != pos2)
    {
        v.push_back(s.substr(pos1, pos2-pos1));

        pos1 = pos2 + c.size();
        pos2 = s.find(c, pos1);
    }
    if(pos1 != s.length())
        v.push_back(s.substr(pos1));


    std::vector<std::string>::iterator start = v.end()-4;
    std::vector<std::string>::iterator end   = v.end();

    std::vector<std::string> Getlast4File(start,end);
    return Getlast4File;
}

tensorflow::Tensor ImgData::OutputImageData(std::pair<int, std::string> imageinfo)
{
    //read image
    cv::Mat Imgdata = cv::imread(imageinfo.second,CV_LOAD_IMAGE_COLOR);
    //norm
    cv::Mat NormImgData;
    Imgdata.convertTo(NormImgData,CV_32FC3,1/255.0);
    //rotate
    cv::Mat RateImgData =rotate_arbitrarily_angle(NormImgData,imageinfo.first);
    //resize
    int height = 46;
    int width = 46;
    cv::Mat dts;
    cv::resize(RateImgData,dts,cv::Size(height,width));
    //toTensor
    float *image_float_data =(float*)dts.data;

    tensorflow::Tensor image_tensor;

    tensorflow::TensorShape image_shape = tensorflow::TensorShape{1, dts.rows, dts.cols, dts.channels()};
    image_tensor = tensorflow::Tensor(tensorflow::DT_FLOAT, image_shape);
    std::copy_n(image_float_data, image_shape.num_elements(), image_tensor.flat<float>().data());

    return image_tensor;
}


/**
 * @brief deprecated code of class ImgData
 */

cv::Mat ImgData::rotate(int rotate ,cv::Mat src)
{
    cv::Mat dts;
    if(rotate==0)
    {
        return src;
    }
    else if (rotate==1) {
        // 矩阵转置
        transpose(src, dts);
        flip(src, dts, 1);
        return dts;
    }
    else if (rotate==2) {
        flip(src, dts, 0);
        flip(src, dts, 1);
        return dts;
    }
    else if (rotate==3) {
        transpose(src, dts);
        flip(src, dts, 0);
        return dts;
    }
}

cv::Mat ImgData::rotate_arbitrarily_angle(Mat &src,int Inputangle)
{
    double angle;
    switch (Inputangle) {
    case 0:
        angle = 0;
        break;
    case 1:

        angle = 90.0;
        break;
    case 2:
        angle = 180.0;
        break;
    case 3:
        angle = 270.0;
        break;
    default:
        angle = 0;
    }
    std::cout<<"angle :"<<angle<<std::endl;

    Mat dst;
    double radian = (double) (angle /180.0 * CV_PI);

    //填充图像
    int maxBorder =(int) (max(src.cols, src.rows)* 1.414 ); //即为sqrt(2)*max
    int dx = (maxBorder - src.cols)/2;
    int dy = (maxBorder - src.rows)/2;
    copyMakeBorder(src, dst, dy, dy, dx, dx, BORDER_REPLICATE,cv::Scalar(255, 255, 255));

    //旋转
    Point2f center( (float)(dst.cols/2) , (float) (dst.rows/2));
    Mat affine_matrix = getRotationMatrix2D( center, angle, 1.0 );//求得旋转矩阵
    warpAffine(dst, dst, affine_matrix, dst.size());

    //计算图像旋转之后包含图像的最大的矩形
    float sinVal = abs(sin(radian));
    float cosVal = abs(cos(radian));
    Size targetSize( (int)(src.cols * cosVal +src.rows * sinVal),
                     (int)(src.cols * sinVal + src.rows * cosVal) );

    //剪掉多余边框
    int x = (dst.cols - targetSize.width) / 2;
    int y = (dst.rows - targetSize.height) / 2;
    Rect rect(x, y, targetSize.width, targetSize.height);
    dst = Mat(dst,rect);

    return dst;
}

std::pair<Mat, std::vector<std::string> > ImgData::RandomSomeoneImgData()
{
    this->GetImgPath(Input_param);

    std::pair<int,int> range;
    range.first = 0;
    range.second = ImgPathToClassInfo.size();
    int random = tools::RandomIntTool(range);
    cv::Mat RandomImgdata = cv::imread(ImgPathList[random],CV_LOAD_IMAGE_COLOR);
    //norm
    RandomImgdata.convertTo(RandomImgdata,CV_32FC3,1/255.0);

    std::pair<Mat, std::vector<std::string> > result;
    result.first  = RandomImgdata;
    result.second = ImgClassinfo[random];

    return result;

}

std::pair<cv::Mat, std::vector<std::string> > ImgData::rotate_image()
{
    //random
    std::pair<double,double> rotate_angle;
    rotate_angle.first=0;
    rotate_angle.second = 360;
    std::pair<cv::Mat,std::vector<std::string> > acceptRandom ;
    acceptRandom = RandomSomeoneImgData();
    Mat dst = rotate_arbitrarily_angle(acceptRandom.first, tools::RandomDoubleTool(rotate_angle));

    std::pair<cv::Mat, std::vector<std::string> > result;
    result.first = dst;
    result.second = acceptRandom.second;

    return result;
}


/**
 * @brief generateInputsLabelsAndTarget
 * @param params
 * @param imagedata
 * @param is_test
 * @return
 */
std::vector<Tensor> generateInputsLabelsAndTarget(const DefaultParams &params, ImgData &imagedata)
{
    const int nbsteps = params.nbshots * ((params.prestime + params.ipd) * params.nbclesses) + params.prestimetest;

    Tensor inputT(DT_FLOAT,tensorflow::TensorShape({nbsteps,1,1,params.ImgSize,params.ImgSize}));
    Tensor labelT(DT_FLOAT,tensorflow::TensorShape({nbsteps,1,params.nbclesses}));
    inputT.flat<float>().setZero(); labelT.flat<float>().setZero();

    std::vector<int> cats;
    std::cout << "image_size: " << imagedata.imagedata.size() << std::endl;
    if(params.isFixed == false)
    {
      cats = tools::vectorNbclasses(tools::vectorShuffleInt(tools::arange(0, imagedata.imagedata.size())), params.nbclesses);
    }
    else
    {
      cats = tools::vectorNbclasses(tools::arange(0, imagedata.imagedata.size()), params.nbclesses);
    }
    //cats = np.random.permutation(cats)
    cats = tools::vectorShuffleInt(cats);
    //std::cout<<"cats_size"<<cats.size()<<std::endl;
    //rots = np.random.randint(4, size=len(imagedata))
    std::vector<int> rotsList;
    //std::srand(params.rngseed);
    std::cout<<"imagedata.imagedata.size:"<<imagedata.imagedata.size()<<std::endl;
    for(int i=0;i<imagedata.imagedata.size();i++)
    {
      rotsList.push_back(tools::RandomSeed(0,4)); //[0,4)   [0,1,2,3]
    }
    //testcat = random.choice(cats) # select the class on which we'll test in this episode
    //int testcat = tools::RandomSeed(0,cats.size());
    int testcat = tools::RandomSeed(0,params.nbshots);
    //cout
    for(int test=0;test<params.nbclesses;test++)
    {
      std::cout<<" : "<<cats[test]<<" ";
    }
    std::cout<<std::endl;
    std::pair<int,std::string> imginfo;//rots,imgPath

    //# Inserting the character images and labels in the input tensor at the proper places
    int location = 0;
    for(int nc=0;nc<params.nbshots;nc++)
    {
      std::vector<int> shuffleCats = tools::vectorShuffleInt(cats);
      //cout
      std::cout<<"shuffleCats:"<<std::endl;
      for(int test=0;test<params.nbclesses;test++)
      {
        std::cout<<" : "<<shuffleCats[test]<<" ";
      }
      std::cout<<std::endl;
      for(int ii=0;ii<shuffleCats.size();ii++)
      {
        std::cout<<"::"<<imagedata.imagedata[shuffleCats[ii]].size()<<std::endl;
        int p = tools::RandomSeed(0,imagedata.imagedata[shuffleCats[ii]].size());
        //imagedata[p]
        std::string imgPath = imagedata.imagedata[shuffleCats[ii]][p];
        //imginfo.first = rotsList[((ii*20)+p+1)];
        imginfo.first = rotsList[shuffleCats[ii]];
        imginfo.second = imgPath;
        tensorflow::Tensor InputImageData = imagedata.OutputImageData(imginfo);
        //InputImageData
        for(int nn=0;nn<params.prestime;nn++)
        {
          //labelT(location,0,);
          int index_=0;
          auto templabelT = labelT.tensor<float,3>();
          for(int xx=0;xx<params.nbclesses;xx++)
          {
            //std::cout<<"xx:"<<shuffleCats[ii]<<std::endl;
            if(cats[xx]==shuffleCats[ii])
            {
              std::cout<<"xx:"<<xx<<std::endl;
              index_ = xx;
            }
          }
          for(int index_x=0;index_x<params.nbclesses;index_x++)
          {
            if(index_ == index_x)
            {
              //std::cout<<"index:"<<index<<std::endl;
              templabelT(location,0,index_x) = 1;
            }
            else
            {
              templabelT(location,0,index_x) = 0;
            }
          }
          //cout
          for(int test=0;test<params.nbclesses;test++)
          {
            std::cout<<" : "<<templabelT(location,0,test);
          }
          std::cout<<std::endl;
          auto temp = InputImageData.tensor<float,4>();
          auto inputT_temp = inputT.tensor<float,5>();
          for(int i=0;i<params.ImgSize;i++)
          {
            for(int j=0;j<params.ImgSize;j++)
            {
              inputT_temp(location,0,0,i,j) = temp(0,i,j,0);
            }
          }
          location+=1;
        }
        location+=params.ipd;
      }
    }

    //std::cout<<"-------for this------ location:"<<location<<std::endl;
    // Inserting the test character
    int pRandom = tools::RandomSeed(0,imagedata.imagedata[cats[testcat]].size());
    std::string p_test = imagedata.imagedata[cats[testcat]][pRandom];
    std::pair<int, std::string> testImg;
    testImg.first = rotsList[cats[testcat]];
    testImg.second = p_test;
    //std::cout<<"pRandom:"<<pRandom<<"p_test:"<<p_test<<endl;
    tensorflow::Tensor testImageData = imagedata.OutputImageData(testImg);
    //std::cout<<"-------reader over------"<<std::endl;
    for(int nn=0;nn<params.prestimetest;nn++)
    {
      auto temp = testImageData.tensor<float,4>();
      auto inputT_temp_2 = inputT.tensor<float,5>();
      for(int m=0;m<params.ImgSize;m++)
      {
        for(int n=0;n<params.ImgSize;n++)
        {
          inputT_temp_2(location,0,0,m,n) = temp(0,m,n,0);
        }
      }
      location+=1;
    }


    //Generating the test label
    //params.nbclesses
    Tensor testlabel(DT_FLOAT,tensorflow::TensorShape({params.nbclesses}));
    auto temp_testlabel = testlabel.vec<float>();
    for(int x=0;x<params.nbclesses;x++)
    {
      if(x==testcat)
      {
        std::cout<<"x:"<<x<<std::endl;
        temp_testlabel(x) = 1.0f;
      }
      else
      {
        temp_testlabel(x) = 0.0f;
      }
    }
    //cout
    for(int test=0;test<params.nbclesses;test++)
    {
      std::cout<<" : "<<temp_testlabel(test);
    }
    std::cout<<std::endl;

    if(location == nbsteps)
    {
      std::cout<<"ok!"<<std::endl;
    }
    else
    {
      std::cout<<"####ERROR#####"<<std::endl;
    }

    //vector<Tensor> labels;
    //labels.push_back(inputT);
    //labels.push_back(labelT);
    //labels.push_back(testlabel);

    return {inputT,labelT,testlabel};
    //return labels;
}
