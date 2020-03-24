#include "mobilenet_ssd/mobilenet_ssd_tensorrt_model.h"

namespace machine_learning
{

MobileNetSSDTensorRTModel::MobileNetSSDTensorRTModel()
{
  string package_name = "pretrained_model";
  string package_dir = ros::package::getPath(package_name);

  string netcfg_path_ = package_dir + "/TensorRT/MobileNetSSD/TensorRT_MobileNetSSD_Car.prototxt";
  string model_path_  = package_dir + "/TensorRT/MobileNetSSD/TensorRT_MobileNetSSD_Car.caffemodel";

  const char* modelConfiguration = netcfg_path_.data();
  const char* modelBinary = model_path_.data();

  INPUT_BLOB_NAME = "data";
  OUTPUT_BLOB_NAME = "detection_out";
  output_vector = {OUTPUT_BLOB_NAME};

  tensorNet.LoadNetwork(modelConfiguration,modelBinary,INPUT_BLOB_NAME, output_vector,1);

  //alloc memory for input image and output results
  DimsCHW dimsData = tensorNet.getTensorDims(INPUT_BLOB_NAME);
  DimsCHW dimsOut  = tensorNet.getTensorDims(OUTPUT_BLOB_NAME);

  //data = _allocateMemory( dimsData, (char*)"input blob");
  output = _allocateMemory( dimsOut, (char*)"output blob");

}

MobileNetSSDTensorRTModel::~MobileNetSSDTensorRTModel()
{

}

void MobileNetSSDTensorRTModel::train(int start, int end)
{
    throw std::runtime_error("No train implementation.");
}

ResultType MobileNetSSDTensorRTModel::evaluate(cv::Mat &cv_img)
{
    Timer time;
    time.tic();

    //output
    ResultType result;
    auto vec_bboxes_  = &result.vec_bboxes;
    auto vec_scores_  = &result.vec_scores;
    auto vec_classes_ = &result.vec_classes;
    //inputs
    cv::Mat cv_img_;
    cv_img_ = cv_img.clone();

    // resize img
    cv::resize(cv_img_, cv_img_, cv::Size(inWidth,inHeight));

    const size_t size = inWidth * inHeight * sizeof(float3);

    if(!imgCUDA) {
      if( CUDA_FAILED( cudaMalloc( &imgCUDA, size))){
          cout <<"Cuda Memory allocation error occured."<<endl;
      }
    }
    if(!imgCPU){
      imgCPU = malloc(size);
      memset(imgCPU,0,size);
    }

    _loadImg(cv_img_,cv_img_.rows,cv_img_.cols,(float*)imgCPU,make_float3(meanVal,meanVal,meanVal),0.007843);

    cudaMemcpyAsync(imgCUDA,imgCPU,size,cudaMemcpyHostToDevice);


    void* buffers[] = { imgCUDA, output };

    //tensorNet.imageInference( buffers, output_vector.size() + 1, BATCH_SIZE);
    tensorNet.imageInference( buffers, output_vector.size() + 1, 1);

    for (int k=0; k<100; k++)
    {
        if(output[7*k+1] == -1)
            break;
        int classIndex = output[7*k+1];
        float confidence = output[7*k+2];

        float left   = output[7*k + 3] * cv_img.cols;
        float top    = output[7*k + 4] * cv_img.rows;
        float right  = output[7*k + 5] * cv_img.cols;
        float bottom = output[7*k + 6] * cv_img.rows;

        if (classNames[classIndex] == "car"){
        vec_classes_->push_back("car");
        vec_scores_->push_back(confidence);
        vector<float> bbox;
        bbox.push_back(left);
        bbox.push_back(top);
        bbox.push_back(right - left);
        bbox.push_back(bottom - top);
        vec_bboxes_->push_back(bbox);
      }
    }

    time.toc();

    //double tt = time.t;
    cout<<"Running Time: "<<time.t<<" ms "<<endl;

    return result;
}

void MobileNetSSDTensorRTModel::batch_evaluate(int size)
{
    throw std::runtime_error("No batch_evaluate implementation.");
}

float* MobileNetSSDTensorRTModel::_allocateMemory(DimsCHW dims, char* info)
{
    float* ptr;
    size_t size;
    std::cout << "Allocate memory: " << info << std::endl;
    //size = BATCH_SIZE * dims.c() * dims.h() * dims.w();
    size = dims.c() * dims.h() * dims.w();
    assert(!cudaMallocManaged( &ptr, size*sizeof(float)));
    return ptr;
}

void MobileNetSSDTensorRTModel::_loadImg( cv::Mat &input, int re_width, int re_height, float *data_unifrom,const float3 mean,const float scale )
{
    int i;
    int j;
    int line_offset;
    int offset_g;
    int offset_r;
    cv::Mat dst;

    unsigned char *line = NULL;
    float *unifrom_data = data_unifrom;

    cv::resize( input, dst, cv::Size( re_width, re_height ), (0.0), (0.0), cv::INTER_LINEAR );
    offset_g = re_width * re_height;//
    offset_r = re_width * re_height * 2;
    for( i = 0; i < re_height; ++i )
    {
        line = dst.ptr< unsigned char >( i );
        line_offset = i * re_width;
        for( j = 0; j < re_width; ++j )
        {
            // b
            unifrom_data[ line_offset + j  ] = (( float )(line[ j * 3 ] - mean.x) * scale);
            // g
            unifrom_data[ offset_g + line_offset + j ] = (( float )(line[ j * 3 + 1 ] - mean.y) * scale);
            // r
            unifrom_data[ offset_r + line_offset + j ] = (( float )(line[ j * 3 + 2 ] - mean.z) * scale);
        }
    }
}

}//namespace
