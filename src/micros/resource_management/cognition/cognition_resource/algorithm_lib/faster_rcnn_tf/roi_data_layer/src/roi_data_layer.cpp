#include "roi_data_layer/roi_data_layer.h"
#include "datasets/roidb.h"

namespace rdl{

RoIDataLayer::RoIDataLayer(std::vector<annotation_prepare> get_roidbs)
{
    roidbs = get_roidbs;
    IMS_PER_BATCH = fast_rcnn::cfg.TRAIN.IMS_PER_BATCH;
    roidb_size = roidbs.size();

    perm = Tensor1i(roidb_size);
    for(int i = 0; i < roidb_size; ++i){
        perm(i) = i ;
    }
    cur = 0;
    //_shuffle_roidb_inds(); //代码调试,第一轮数据集更改为固定顺序
}

RoIDataLayer::~RoIDataLayer()
{
}

void RoIDataLayer::_shuffle_roidb_inds()
{	
    //srand(int(time(0)));
    //std::shuffle();
    //std::random_shuffle();
    for(int i = roidb_size -1; i > 0; --i){
        int a = rand()%(i+1);
        int temp = perm(i);
        perm(i) = perm(a);
        perm(a) = temp;
    }
    cur = 0;
}

Blobs RoIDataLayer::forward()
{
    Blobs blobs_tmp;
    blobs_tmp = _get_next_minibatch();
    return blobs_tmp;
}

Blobs RoIDataLayer::_get_next_minibatch()
{
    Tensor1i db_inds = _get_next_minibatch_inds();

    minibatch_db.clear();
    for(int i =0; i < IMS_PER_BATCH; ++i)
    {
      minibatch_db.push_back(roidbs[db_inds(i)]);//根据db_inds的元素个数读取对应个数的roidb
    }
    return _get_minibatch(minibatch_db);
}

Tensor1i RoIDataLayer::_get_next_minibatch_inds()
{
    assert(cfg.TRAIN.HAS_RPN);
    if(cur + IMS_PER_BATCH>= roidb_size)
        _shuffle_roidb_inds();

    Tensor1i db_inds(IMS_PER_BATCH);
    for(int i = 0; i < IMS_PER_BATCH; ++i){
        db_inds(i) = perm(cur + i);
    }
    cur += IMS_PER_BATCH;
    return db_inds;
}

Blobs RoIDataLayer::_get_minibatch(std::vector<annotation_prepare> batch_roidbs)
{
    Blobs blobs;
    assert(fast_rcnn::cfg.TRAIN.HAS_RPN);
    assert(batch_roidbs.size() == 1);
    annotation_prepare roidb = batch_roidbs.front();
    cout << "get one image named:"<< roidb.image_name << endl;
    //数据处理
    float im_scale;
    string im_path = fast_rcnn::cfg.DATA_DIR + "VOCdevkit/VOC2007/JPEGImages/" + roidb.image_name;
    _get_image_blob(im_path, blobs.data, im_scale);
        int im_h = blobs.data.shape().dim_size(1);
        int im_w = blobs.data.shape().dim_size(2);
    int fg_size = roidb.gt_classes.size();
    Tensor2f gt_boxes(fg_size, 5);    //(n,5):n代表不同的前景框
    roidb.boxes;                      //[(x_min,y_min),(x_max,y_max)]
    for(int i=0; i< fg_size; ++i){
        gt_boxes(i, 0) = roidb.boxes[i][0] * im_scale;
        gt_boxes(i, 1) = roidb.boxes[i][1] * im_scale;
        gt_boxes(i, 2) = roidb.boxes[i][2] * im_scale;
        gt_boxes(i, 3) = roidb.boxes[i][3] * im_scale;
        gt_boxes(i, 4) = roidb.gt_classes[i];
    }

    //数据返回
    blobs.gt_boxes = Tensor(DT_FLOAT, {fg_size, 5});
    blobs.gt_boxes.tensor<float,2>() = gt_boxes;
    blobs.im_info = Tensor(DT_FLOAT, {1, 3});
    blobs.im_info.tensor<float,2>().setValues({{float(im_h), float(im_w), im_scale},}); //(1,3)
    return blobs;
}

// """Builds an input blob from the images in the roidb at the specified scales.
void RoIDataLayer::_get_image_blob(string& im_path, Tensor& image, float& scale){
    //vector<Tensor> resized_tensors;
    //TF_CHECK_OK(RoIDataLayer::_ReadTensorFromImageFile(im_path, scale, &resized_tensors));
    //image = resized_tensors[0];
    cv::Mat cv_img = cv::imread(im_path, CV_LOAD_IMAGE_COLOR);
    cvMat2tfTensor(cv_img, scale, image);
}

//读取图片
void RoIDataLayer::cvMat2tfTensor(cv::Mat& cv_img, float& img_scale, tensorflow::Tensor& image_tensor){
    // minus means  //py-faster-rcnn中demo.py代码与C++版本的代码对比
                    //https://blog.csdn.net/u013553529/article/details/79029270?utm_source=copy
    cv::Mat cv_new(cv_img.rows, cv_img.cols, CV_32FC3, cv::Scalar(0,0,0));
    for (int h = 0; h < cv_img.rows; ++h ){
        for (int w = 0; w < cv_img.cols; ++w){
            cv_new.at<cv::Vec3f>(cv::Point(w, h))[0] = float(cv_img.at<cv::Vec3b>(cv::Point(w, h))[0])-float(102.9801);// Blue
            cv_new.at<cv::Vec3f>(cv::Point(w, h))[1] = float(cv_img.at<cv::Vec3b>(cv::Point(w, h))[1])-float(115.9465);// Green
            cv_new.at<cv::Vec3f>(cv::Point(w, h))[2] = float(cv_img.at<cv::Vec3b>(cv::Point(w, h))[2])-float(122.7717);// Red
        }
    }

    // resize: scale image to (1000, y) or (x, 600)
    const int  MAX_SIZE = 1000;
    const int  SCALE_SIZE = 600;
    int max_side = max(cv_img.rows, cv_img.cols);
    int min_side = min(cv_img.rows, cv_img.cols);
    cout << "(width, height) = (" << cv_img.cols << ", " << cv_img.rows << ")" << endl;
    img_scale = float(SCALE_SIZE) / float(min_side);
    if (round(float(max_side) * img_scale) > MAX_SIZE) {
        img_scale = float(MAX_SIZE) / float(max_side);
    }
    cout << "img_scale: " << img_scale << endl;

    int height = int(cv_img.rows * img_scale);
    int width = int(cv_img.cols * img_scale);
    cout << "re-scaled (width, height) = (" << width << ", " << height << ")" << endl;
    cv::Mat cv_resized;
    cv::resize(cv_new, cv_resized, cv::Size(width, height));

    //cv::Mat to tensorflow::Tensor
        // convert byte to float image
    cv::Mat image_float;
    cv_resized.convertTo(image_float, CV_32FC3);
    float *image_float_data = (float*)image_float.data;
        // create input shape
    tensorflow::TensorShape image_shape = tensorflow::TensorShape{1, cv_resized.rows, cv_resized.cols, cv_resized.channels()};
    image_tensor = tensorflow::Tensor(tensorflow::DT_FLOAT, image_shape);
    std::copy_n((char*) image_float_data, image_shape.num_elements() * sizeof(float),
                    const_cast<char*>(image_tensor.tensor_data().data()));
}

bool RoIDataLayer::EndsWith(const string &a, const string &b)
{
    string::const_reverse_iterator i = a.rbegin();
    string::const_reverse_iterator j = b.rbegin();
    while(j != b.rend())
    {
        if(i == a.rend() || *i != *j) return false;
        i++; j++;
    }
    return true;
}

Status RoIDataLayer::_ReadEntireFile(tensorflow::Env* env, const string& filename, Tensor* output){
    tensorflow::uint64 file_size = 0;
    TF_RETURN_IF_ERROR(env->GetFileSize(filename, &file_size));

    string contents;
    contents.resize(file_size);

    std::unique_ptr<tensorflow::RandomAccessFile> file;
    TF_RETURN_IF_ERROR(env->NewRandomAccessFile(filename, &file));

    tensorflow::StringPiece data;
    TF_RETURN_IF_ERROR(file->Read(0, file_size, &data, &(contents)[0]));
    if (data.size() != file_size) {
      return tensorflow::errors::DataLoss("Truncated read of '", filename,
                                          "' expected ", file_size, " got ",
                                          data.size());
    }
    output->scalar<string>()() = data.ToString();
    return Status::OK();
}

Status RoIDataLayer::_ReadTensorFromImageFile(const string &file_name, float &im_scale, std::vector<Tensor> *out_tensors)
{/*
    //从file name读取一张图片.并resize到特定大小(input_height,input_width)
    //并对图片数据作期望的预处理:(600,长边)/(1000,短边);长边<1000,短边<600;
*/
    auto root = tensorflow::Scope::NewRootScope();
    using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

    string input_name = "file_reader";
    string output_name = "normalized";

    // read file_name into a tensor named input
    Tensor input(tensorflow::DT_STRING, tensorflow::TensorShape());
    TF_RETURN_IF_ERROR(
        _ReadEntireFile(tensorflow::Env::Default(), file_name, &input));

    // use a placeholder to read input data
    auto file_reader =
        Placeholder(root.WithOpName("input"), tensorflow::DataType::DT_STRING);

    std::vector<std::pair<string, tensorflow::Tensor>> inputs = {
        {"input", input}, };

    // Now try to figure out what kind of file it is and decode it.
    const int wanted_channels = 3;
    tensorflow::Output image_reader;
    if (EndsWith(file_name, ".png")){
      image_reader = DecodePng(root.WithOpName("png_reader"), file_reader,
                               DecodePng::Channels(wanted_channels));
    } else if (EndsWith(file_name, ".gif")) {
      // gif decoder returns 4-D tensor, remove the first dim
      image_reader =
          Squeeze(root.WithOpName("squeeze_first_dim"),
                  DecodeGif(root.WithOpName("gif_reader"), file_reader));
    } else if (EndsWith(file_name, ".bmp")) {
      image_reader = DecodeBmp(root.WithOpName("bmp_reader"), file_reader);
    } else {
      // Assume if it's neither a PNG nor a GIF then it must be a JPEG.
      image_reader = DecodeJpeg(root.WithOpName("jpeg_reader"), file_reader,
                                DecodeJpeg::Channels(wanted_channels));
    }
    // Now cast the image data to float so we can do normal math on it.
    auto float_caster =
        Cast(root.WithOpName("float_caster"), image_reader, tensorflow::DT_FLOAT);

    ///此处先获取dims_expander(是个op)大小,再resized,需要写测试程序
    /// 如果不能直接获取op的output的大小,就只能在这里直接session run了
    auto img_shape = Shape(root.WithOpName("img_shape"), float_caster);

    // This runs the GraphDef network definition that we've just constructed, and
    // returns the results in the output tensor.
    tensorflow::GraphDef graph_0;
    TF_RETURN_IF_ERROR(root.ToGraphDef(&graph_0));

    std::unique_ptr<tensorflow::Session> session_0(
        tensorflow::NewSession(tensorflow::SessionOptions()));
    TF_RETURN_IF_ERROR(session_0->Create(graph_0));
    TF_RETURN_IF_ERROR(session_0->Run({inputs}, {string("img_shape")}, {}, out_tensors));
    //cout << "image size(h,w,c): " << out_tensors->back().tensor<int,1>() << endl; //获取了形状的某一维

    //求解图像合适的尺度:(600,长边)/(1000,短边);长边<1000,短边<600;
    auto img_dims = out_tensors->back().tensor<int,1>();
    int h = img_dims(0);
    int w = img_dims(1);
    int im_size_min = (w < h) ? w : h;
    int im_size_max = (w > h) ? w : h;
    for(auto target_size : fast_rcnn::cfg.TEST.SCALES ){ //实际上SCALES只有一个数,并没有构建完整的图像金字塔
        im_scale = float(target_size) / float(im_size_min);
        if (round(im_scale * im_size_max) > fast_rcnn::cfg.TEST.MAX_SIZE){
            im_scale = float(fast_rcnn::cfg.TEST.MAX_SIZE) / float(im_size_max);
            //im_scale_factors.push_back(im_scale); //图像金子塔的scales,暂时不需要
        }
    }

    //输出形状 4D:[batch, height, width, channel]
    int output_h = h*im_scale;
    int output_w = w*im_scale;
    auto dims_expander = Reverse(root, ExpandDims(root, float_caster, 0), {3});
    // Bilinearly resize the image to fit the required dimensions.
    auto resized = ResizeNearestNeighbor(
        root.WithOpName(output_name), Sub(root, dims_expander, {fast_rcnn::cfg.PIXEL_MEANS}),
        Const(root.WithOpName("size"), {output_h, output_w}));

    tensorflow::GraphDef graph;
    TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));
    std::unique_ptr<tensorflow::Session> session(
        tensorflow::NewSession(tensorflow::SessionOptions()));
    TF_RETURN_IF_ERROR(session->Create(graph));
    TF_RETURN_IF_ERROR(session->Run({inputs}, {output_name}, {}, out_tensors));

    return Status::OK();
}

bool _is_valid(vector<float>& max_overlaps)
{
    for(std::vector<float>::iterator itr2 = max_overlaps.begin(); itr2 != max_overlaps.end(); itr2++){
        if(*itr2 >= cfg.TRAIN.FG_THRESH || (*itr2 < cfg.TRAIN.BG_THRESH_HI &
                                            *itr2 >= cfg.TRAIN.BG_THRESH_LO)){
            return true;
        }
    }

    return false;
}

vector<annotation_prepare> _filter_roidb(std::vector<annotation_prepare> &roidb){
    int num = roidb.size();

    for(std::vector<annotation_prepare>::iterator itr1 = roidb.begin(); itr1 != roidb.end();){
        if(!_is_valid((*itr1).max_overlaps)){
            roidb.erase(itr1);
        }
        else{
          itr1++;
        }
    }

    int num_after = roidb.size();
    std::cout << "Filtered " << num -num_after <<" roidb entries:"
    << num << " -> " << num_after << std::endl;

    return roidb;
}

//上层接口
RoIDataLayer get_data_layer(vector<float> &means_ravel, vector<float> &stds_ravel){
    assert(cfg.TRAIN.HAS_RPN);
    assert(fast_rcnn::cfg.TRAIN.IS_MULTISCALE == false);

    pascal_voc imdb("trainval", "2007", fast_rcnn::cfg.DATA_DIR);
        cout << "Loaded dataset "<< imdb.name() << " for training" <<endl;

    //使用水平反转图像（数据增强），防止过拟合
    if (cfg.TRAIN.USE_FLIPPED){
        cout << "Appending horizontally-flipped training examples..." <<endl;
        imdb.append_flipped_images();
        cout << "done" << endl;
    }

    cout << "Preparing training data..." << endl;
    vector<annotation_prepare> roidbs = imdb.prepare_roidb();
    cout << "done" << endl;

    _filter_roidb(roidbs);

    datasets::add_bbox_regression_targets(roidbs, means_ravel, stds_ravel);
    return RoIDataLayer(roidbs);
}

}//namespace
