#ifndef CONFIG_H
#define CONFIG_H
#include <string>
#include <vector>
#include <boost/filesystem.hpp>
#include <boost/optional.hpp>
#include <ros/ros.h>
namespace fast_rcnn {
using namespace std;
using namespace boost;
using namespace boost::filesystem;
//! 调用此函数注意增加返回值非空判断, optional用于包装可能出现的非法值
optional<path> find_dir(const path& dir, const string& dirname );
//获取当前文件路径
size_t get_executable_path( char* processdir, size_t len);
//获取rootPath
string getRootPath();

//#
//# Training options
//#
struct Train
{
    // learning rate
    float LEARNING_RATE = 0.001;
    float MOMENTUM = 0.9;
    float GAMMA = 0.1;
    int STEPSIZE = 50000;
    int DISPLAY = 10;
    bool IS_MULTISCALE = false;


    //# Scales to compute real features
    //vector<float> SCALES_BASE = {0.25, 0.5, 1.0, 2.0, 3.0};
    //vector<float> SCALES_BASE = {1.0}; //(1.0,)

    //# parameters for ROI generating
    //float SPATIAL_SCALE = 0.0625;
    //int KERNEL_SIZE = 5;

    //# Aspect ratio to use during training
    //vector<float> ASPECTS = {1, 0.75, 0.5, 0.25};
    //vector<float> ASPECTS= {1}; //(1,)

    //# Scales to use during training (can list multiple scales)
    //# Each scale is the pixel size of an image's shortest side
    vector<int> SCALES = {600}; //(600,)数据类型存疑

    //# Max pixel size of the longest side of a scaled input image
    int MAX_SIZE = 1000;

    //# Images to use per minibatch
    int IMS_PER_BATCH = 1;

    //# Minibatch size (number of regions of interest [ROIs])
    int BATCH_SIZE = 128;

    //# Fraction of minibatch that is labeled foreground (i.e. class > 0)
    float FG_FRACTION = 0.25;

    //# Overlap threshold for a ROI to be considered foreground (if >= FG_THRESH)
    float FG_THRESH = 0.5;

    //# Overlap threshold for a ROI to be considered background (class = 0 if
    //# overlap in [LO, HI))
    float BG_THRESH_HI = 0.5;
    float BG_THRESH_LO = 0.0;

    //# Use horizontally-flipped images during training?
    bool USE_FLIPPED = true;

    //# Train bounding-box regressors
    bool BBOX_REG = true;

    //# Overlap required between a ROI and ground-truth box in order for that ROI to
    //# be used as a bounding-box regression training example
    float BBOX_THRESH = 0.5;

    //# Iterations between snapshots
    int SNAPSHOT_ITERS = 5000;

    //# solver.prototxt specifies the snapshot path prefix, this adds an optional
    //# infix to yield the path: <prefix>[_<infix>]_iters_XYZ.caffemodel
    string SNAPSHOT_PREFIX = "VGGnet_fast_rcnn";
    string SNAPSHOT_INFIX = "";

    //# Use a prefetch thread in roi_data_layer.layer
    //# So far I haven't found this useful; likely more engineering work is required
    bool USE_PREFETCH = false;

    //# Normalize the targets (subtract empirical mean, divide by empirical stddev)
    bool BBOX_NORMALIZE_TARGETS = true;
    //# Deprecated (inside weights)
    vector<float> BBOX_INSIDE_WEIGHTS = {1.0, 1.0, 1.0, 1.0};
    //# Normalize the targets using "precomputed" (or made up) means and stdevs
    //# (BBOX_NORMALIZE_TARGETS must also be True)
    bool BBOX_NORMALIZE_TARGETS_PRECOMPUTED = true;
    vector<float> BBOX_NORMALIZE_MEANS = {0.0, 0.0, 0.0, 0.0};
    vector<float> BBOX_NORMALIZE_STDS = {0.1, 0.1, 0.2, 0.2};

    //# Train using these proposals
    string PROPOSAL_METHOD = "gt";

    //# Make minibatches from images that have similar aspect ratios (i.e. both
    //# tall and thin or both short and wide) in order to avoid wasting computation
    //# on zero-padding.
    bool ASPECT_GROUPING = true;

    //# Use RPN to detect objects
    bool HAS_RPN = true;
    //# IOU >= thresh: positive example
    float RPN_POSITIVE_OVERLAP = 0.7;
    //# IOU < thresh: negative example
    float RPN_NEGATIVE_OVERLAP = 0.3;
    //# If an anchor statisfied by positive and negative conditions set to negative
    bool RPN_CLOBBER_POSITIVES = false;
    //# Max number of foreground examples
    float RPN_FG_FRACTION = 0.5;
    //# Total number of examples
    int RPN_BATCHSIZE = 256;
    //# NMS threshold used on RPN proposals
    float RPN_NMS_THRESH = 0.7;
    //# Number of top scoring boxes to keep before apply NMS to RPN proposals
    int RPN_PRE_NMS_TOP_N = 12000;
    //# Number of top scoring boxes to keep after applying NMS to RPN proposals
    int RPN_POST_NMS_TOP_N = 2000;
    //# Proposal height and width both need to be greater than RPN_MIN_SIZE (at orig image scale)
    int RPN_MIN_SIZE = 16;
    //# Deprecated (outside weights)
    vector<float> RPN_BBOX_INSIDE_WEIGHTS = {1.0, 1.0, 1.0, 1.0};
    //# Give the positive RPN examples weight of p * 1 / {num positives}
    //# and give negatives a weight of (1 - p)
    //# Set to -1.0 to use uniform example weighting
    int RPN_POSITIVE_WEIGHT = -1.0;

    //# Enable timeline generation
    bool DEBUG_TIMELINE = false;

};
//#
//# Testing options
//#
struct Test
{
    //# Scales to use during testing (can list multiple scales)
    //# Each scale is the pixel size of an image's shortest side
    vector<int> SCALES = {600}; // (600,)

    //# Max pixel size of the longest side of a scaled input image
    float MAX_SIZE = 1000;

    //# Overlap threshold used for non-maximum suppression (suppress boxes with
    //# IoU >= this threshold)
    float NMS = 0.3;

    //# Experimental: treat the (K+1) units in the cls_score layer as linear
    //# predictors (trained, eg, with one-vs-rest SVMs).
    bool SVM = false;

    //# Test using bounding-box regressors
    bool BBOX_REG = true;

    //# Propose boxes
    bool HAS_RPN = true;

    //# Test using these proposals
    string PROPOSAL_METHOD = "selective_search";

    //## NMS threshold used on RPN proposals
    float RPN_NMS_THRESH = 0.7;
    //## Number of top scoring boxes to keep before apply NMS to RPN proposals
    int RPN_PRE_NMS_TOP_N = 6000;
    //#int RPN_PRE_NMS_TOP_N = 12000
    //## Number of top scoring boxes to keep after applying NMS to RPN proposals
    int RPN_POST_NMS_TOP_N = 300;
    //int RPN_POST_NMS_TOP_N = 2000
    //# Proposal height and width both need to be greater than RPN_MIN_SIZE (at orig image scale)
    int RPN_MIN_SIZE = 16;

    //# Enable timeline generation
    bool DEBUG_TIMELINE = false;

};
//#
//# Total options - MISC
//#
struct CFG
{
    Train TRAIN;
    Test TEST;

    //string NET_NAME = "VGGnet";

    //# The mapping from image coordinates to feature map coordinates might cause
    //# some boxes that are distinct in image space to become identical in feature
    //# coordinates. If DEDUP_BOXES > 0, then DEDUP_BOXES is used as the scale factor
    //# for identifying duplicate boxes.
    //# 1/16 is correct for {Alex,Caffe}Net, VGG_CNN_M_1024, and VGG16
    float DEDUP_BOXES = 1/16;

    //# Pixel mean values (BGR order) as a (1, 1, 3) array
    //# We use the same pixel mean for all networks even though it's not exactly what
    //# they were trained with
    initializer_list<float> PIXEL_MEANS = {102.9801, 115.9465, 122.7717};

    //# For reproducibility
    int RNG_SEED = 3;

    //# A small number that's used many times
    float EPS = 1e-14;

    //# Root directory of project
    //#该root_dir 操作方式：__file__为该config.py文件的路径，所有总路径为abs（__file__/../..）此时即为Fsater-RCNN_TF目录
    string ROOT_DIR = getRootPath();

    //# Data directory
    string DATA_DIR = ROOT_DIR + "/data/";

    //# Model directory
    string MODELS_DIR = ROOT_DIR + "/models" + "/pascal_voc/";

    //# Name (or path to) the matlab executable
    string MATLAB = "/matlab/";

    //# Place outputs under an experiments directory
    string EXP_DIR = "/default/";


#if GOOGLE_CUDA
    //# Use GPU implementation of non-maximum suppression
    bool USE_GPU_NMS = true;

    //# Default GPU device id
    int GPU_ID = 0;
#else
    bool USE_GPU_NMS = false;
#endif

};

//对外接口
extern const CFG cfg; //extern只声明,不定义
string get_output_dir(string imdb_name, string weights_filename = "");

} //namespace

#endif // CONFIG_H
