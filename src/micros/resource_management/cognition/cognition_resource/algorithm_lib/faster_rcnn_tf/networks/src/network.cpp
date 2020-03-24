#include <algorithm>
#include "network.h"
#include "roi_pool.h"
#include "proposal.h"
#include "anchor_target.h"
#include "proposal_target.h"
#include "time_op.h"

using namespace tensorflow;
using namespace tensorflow::ops;

Network::Network(const Scope &scope): scope(scope), session(scope)
{

}
Network::~Network()
{

}
Network *Network::feed(const std::vector<std::string> &names)
{
    int i, n = names.size();
    inputs.clear();
    for(i=0; i<n; i++)
    {
        std::map<std::string, std::vector<Output> >::iterator p = layers.find(names[i]);
        if(p != layers.end())
        {
            std::vector<Output> list = (*p).second;
            inputs.insert(inputs.end(), list.begin(), list.end());
        }
        else std::cerr << "The key is not found: " << names[i] << std::endl;
    }
    return this;
}
Network *Network::load(ClientSession &session, std::string file_path)
{
    const std::vector<std::string> weight_list = get_weight_list();
    const std::vector<std::string> bias_list = get_bias_list();
    const int count = weight_list.size();
    for(int i=0; i<count; i++)
    {
        TF_CHECK_OK(session.Run({Assign(scope, get_weight(weight_list[i]), Restore(scope, file_path, weight_list[i] + "/weights", DT_FLOAT))}, NULL));
    }
    for(int i=0; i<count; i++)
    {
        TF_CHECK_OK(session.Run({Assign(scope, get_bias(bias_list[i]), Restore(scope, file_path, bias_list[i] + "/biases", DT_FLOAT))}, NULL));
    }
    return this;
}
// 用于维护 inputs 和 layers 两个数据结构；在封装 Op 的函数返回之前调用。
Network *Network::add_output(std::string name, Output op)
{
    std::map<std::string, std::vector<Output> >::iterator i = layers.find(name);
    if (i == layers.end())
    {
        i = layers.insert(std::map<std::string, std::vector<Output> >::value_type(name, std::vector<Output>())).first;
    }
    (*i).second.push_back(op);
    inputs.clear();
    inputs.push_back(op);
    return this;
}
const std::vector<Output> &Network::get_output(std::string name)
{
    return (*(layers.find(name))).second;
}
const std::vector<Output> &Network::get_trainable_variables()
{
    return trainable_variables;
}
const std::vector<Output> &Network::get_non_trainable_variables()
{
    return non_trainable_variables;
}
const std::vector<TensorShape> &Network::get_trainable_variable_shapes()
{
    return trainable_variable_shapes;
}
bool Network::output_exists(std::string name)
{
    return layers.find(name) != layers.end();
}
std::vector<std::string> Network::get_weight_list()
{
    std::vector<std::string> weight_list;
    for(std::map<std::string, Output>::iterator i = weights.begin(); i != weights.end(); i++)
    {
        weight_list.push_back((*i).first);
    }
    return weight_list;
}
std::vector<std::string> Network::get_bias_list()
{
    std::vector<std::string> bias_list;
    for(std::map<std::string, Output>::iterator i = biases.begin(); i != biases.end(); i++)
    {
        bias_list.push_back((*i).first);
    }
    return bias_list;
}
GraphDef Network::get_graph_def()
{
    GraphDef graph_def;
    scope.ToGraphDef(&graph_def);
    return graph_def;
}
Output Network::get_weight(std::string name, TensorShape shape, bool trainable)
{
    std::map<std::string, Output>::iterator i = weights.find(name);
    if (i == weights.end())
    {
        auto variable = Variable(scope.WithOpName(name+"/weights"), shape, DT_FLOAT);
        i = weights.insert(std::map<std::string, Output>::value_type(name, variable)).first;
        weight_shapes.insert(std::map<std::string, TensorShape>::value_type(name, shape));
        if(trainable)
        {
            trainable_variables.push_back(variable);
            trainable_variable_shapes.push_back(shape);
        }
        else
        {
            non_trainable_variables.push_back(variable);
        }
    }
    return (*i).second;
}
Output Network::get_bias(std::string name, TensorShape shape, bool trainable)
{
    std::map<std::string, Output>::iterator i = biases.find(name);
    if (i == biases.end())
    {
        auto variable = Variable(scope.WithOpName(name+"/biases"), shape, DT_FLOAT);
        i = biases.insert(std::map<std::string, Output>::value_type(name, variable)).first;
        bias_shapes.insert(std::map<std::string, TensorShape>::value_type(name, shape));
        if(trainable)
        {
            trainable_variables.push_back(variable);
            trainable_variable_shapes.push_back(shape);
        }
        else
        {
            non_trainable_variables.push_back(variable);
        }
    }
    return (*i).second;
}
bool Network::validate_padding(std::string padding)
{
    return padding == "SAME" || padding == "VALID";
}
Network *Network::conv(int k_h, int k_w, int c_o, int s_h, int s_w, std::string name, bool relu, std::string padding, int group, bool trainable)
{
    if(validate_padding(padding) == false)
    {
        std::cerr << "Warning: the padding parameter is neither SAME nor VALID; this parameter is set to SAME by default." << std::endl;
        padding = "SAME";
    }
    // 感觉更优的实现方法：将创建的 ClientSession / Scope 以局部变量的形式存在。
    // 存在问题： input 的 Scope 和 Shape 的 Scope 不一致。
    std::vector<Tensor> output_list;
    std::vector<Output> conv_list;
    Output input = inputs.front();
    TF_CHECK_OK(session.Run(feed_map, {Shape(scope, input)}, &output_list));
    auto dimensions = output_list[0].vec<int>();
    std::cout << name << ": " << inputs.front().name() << std::endl << dimensions << std::endl;
    const int c_i = (int)(dimensions((int)(dimensions.dimension(0))-1));
    assert(c_i%group==0);
    assert(c_o%group==0);
    Output kernel = get_weight(name, {k_h, k_w, c_i/group, c_o}, trainable);
    Output biases = get_bias(name, {c_o}, trainable);
    std::cout << kernel.name() << std::endl;
    std::cout << biases.name() << std::endl;
    TF_CHECK_OK(session.Run({Assign(scope, kernel, Multiply(scope, TruncatedNormal(scope, {k_h, k_w, c_i/group, c_o}, DT_FLOAT), Const(scope, 0.01f)))}, NULL));
    TF_CHECK_OK(session.Run({Assign(scope, biases, Tensor(DT_FLOAT, {c_o}))}, NULL));
    assert(group==1);
    if (group==1)
    {
        conv_list.push_back(Conv2D(scope, input, kernel, {1, s_h, s_w, 1}, padding));
    }
    else
    {
        OutputList input_group = Split(scope, Const(scope, 3), input, group).output;
        OutputList kernel_group = Split(scope, Const(scope, 3), kernel, group).output;
        int i, group_size = std::min(input_group.size(), kernel_group.size());
        std::vector<Input> output_group;
        for(i=0; i<group_size; i++) output_group.push_back(Conv2D(scope, input_group[i], kernel_group[i], {1, s_h, s_w, 1}, padding));
        conv_list.push_back(Concat(scope, tensorflow::gtl::ArraySlice<Input>(output_group), 3));
    }
    if (relu==true)
    {
        add_output(name, Relu(scope.WithOpName(name), BiasAdd(scope, conv_list.back(), biases)));
    }
    else
    {
        add_output(name, BiasAdd(scope.WithOpName(name), conv_list.back(), biases));
    }
    return this;
}
Network *Network::relu(std::string name)
{
    Input input = inputs.front();
    add_output(name, Relu(scope.WithOpName(name), input));
    return this;
}
Network *Network::max_pool(int k_h, int k_w, int s_h, int s_w, std::string name, std::string padding)
{
    if(validate_padding(padding) == false)
    {
        std::cerr << "Warning: the padding parameter is neither SAME nor VALID; this parameter is set to SAME by default." << std::endl;
        padding = "SAME";
    }
    Input input = inputs.front();
    add_output(name, MaxPool(scope.WithOpName(name), input, {1, k_h, k_w, 1}, {1, s_h, s_w, 1}, padding));
    return this;
}
Network *Network::avg_pool(int k_h, int k_w, int s_h, int s_w, std::string name, std::string padding)
{
    if(validate_padding(padding) == false)
    {
        std::cerr << "Warning: the padding parameter is neither SAME nor VALID; this parameter is set to SAME by default." << std::endl;
        padding = "SAME";
    }
    Input input = inputs.front();
    add_output(name, AvgPool(scope.WithOpName(name), input, {1, k_h, k_w, 1}, {1, s_h, s_w, 1}, padding));
    return this;
}
Network *Network::roi_pool(int pooled_height, int pooled_width, float spatial_scale, std::string name)
{
    // std::vector<Tensor> outputs;
    // TF_CHECK_OK(session.Run(feed_map, {Shape(scope, inputs.at(0))}, &outputs));
    // std::cout << name << ": " << outputs[0].vec<int>() << std::endl;
    // TF_CHECK_OK(session.Run(feed_map, {Shape(scope, inputs.at(1))}, &outputs));
    // std::cout << name << ": " << outputs[0].vec<int>() << std::endl;
    add_output(name, RoiPool(scope.WithOpName(name), inputs.at(0), inputs.at(1), pooled_height, pooled_width, spatial_scale).top_data);
    return this;
}
Network *Network::roi_align(int pooled_height, int pooled_width, float spatial_scale, std::string name)
{
    Output feature_map = inputs.at(0);
    Output rois = inputs.at(1);
    auto shape_map = Cast(scope, Sub(scope, Shape(scope, feature_map), 1), DT_FLOAT);
    auto index_list = Cast(scope, Sum(scope, Slice(scope, rois, {0,0}, {-1,1}), {1}), DT_INT32);
    auto x1 = Div(scope, Multiply(scope, Slice(scope, rois, {0,1}, {-1,1}), spatial_scale), Slice(scope, shape_map, {2}, {1}));
    auto y1 = Div(scope, Multiply(scope, Slice(scope, rois, {0,2}, {-1,1}), spatial_scale), Slice(scope, shape_map, {1}, {1}));
    auto x2 = Div(scope, Multiply(scope, Slice(scope, rois, {0,3}, {-1,1}), spatial_scale), Slice(scope, shape_map, {2}, {1}));
    auto y2 = Div(scope, Multiply(scope, Slice(scope, rois, {0,4}, {-1,1}), spatial_scale), Slice(scope, shape_map, {1}, {1}));
    auto rois_processed = Concat(scope, tensorflow::gtl::ArraySlice<Input>({y1, x1, y2, x2}), 1);
    add_output(name, CropAndResize(scope.WithOpName(name), feature_map, rois_processed, index_list, {pooled_height, pooled_width}));
    return this;
}
Network *Network::proposal_layer(const std::vector<int> &_feat_stride, const std::vector<int> &anchor_scales, std::string cfg_key, std::string name)
{
    // std::vector<Tensor> outputs;
    // TF_CHECK_OK(session.Run(feed_map, {Shape(scope, inputs.at(0))}, &outputs));
    // std::cout << name << ": " << outputs[0].vec<int>() << std::endl;
    // TF_CHECK_OK(session.Run(feed_map, {Shape(scope, inputs.at(1))}, &outputs));
    // std::cout << name << ": " << outputs[0].vec<int>() << std::endl;
    // TF_CHECK_OK(session.Run(feed_map, {Shape(scope, inputs.at(2))}, &outputs));
    // std::cout << name << ": " << outputs[0].vec<int>() << std::endl;
    add_output(name, Proposal(scope.WithOpName(name), inputs.at(0), inputs.at(1), inputs.at(2), cfg_key == "TRAIN", _feat_stride.front()));
    return this;
}
Network *Network::anchor_target_layer(const std::vector<int> &_feat_stride, const std::vector<int> &anchor_scales, std::string name)
{
    auto anchor_target = AnchorTarget(scope, inputs[0], inputs[1], inputs[2], inputs[3], _feat_stride.front());
    add_output(name, Cast(scope, anchor_target.rpn_labels, DT_INT32));
    add_output(name, anchor_target.rpn_bbox_targets);
    add_output(name, anchor_target.rpn_bbox_inside_weights);
    add_output(name, anchor_target.rpn_bbox_outside_weights);
    return this;
}
Network *Network::proposal_target_layer(int classes, std::string name)
{
    auto proposal_target = ProposalTarget(scope, inputs[0], inputs[1], classes);
    add_output(name, proposal_target.rois);
    add_output(name, Cast(scope, proposal_target.labels, DT_INT32));
    add_output(name, proposal_target.bbox_targets);
    add_output(name, proposal_target.bbox_inside_weights);
    add_output(name, proposal_target.bbox_outside_weights);
    return this;
}
Network *Network::reshape_layer(int d, std::string name)
{
    std::vector<Tensor> outputs;
    Input input = inputs.front();
    auto input_shape = Shape(scope, input);
    TF_CHECK_OK(session.Run(feed_map, {input_shape}, &outputs));
    auto dimensions = outputs[0].vec<int>();
    std::cout << name << ": " << inputs.front().name() << std::endl << dimensions << std::endl;
    std::cout << name << ": " << d << std::endl;
    assert((dimensions(1)*dimensions(3)) % d == 0);
    Output dim0 = Slice(scope, input_shape, {0}, {1});
    Output dim1 = Slice(scope, input_shape, {1}, {1});
    Output dim2 = Slice(scope, input_shape, {2}, {1});
    Output dim3 = Slice(scope, input_shape, {3}, {1});
    auto new_shape = Concat(scope, {dim0, (Output)Const(scope, {d}), (Output)Div(scope, Multiply(scope, dim1, dim3), {d}), dim2}, 0);
    add_output(name, Transpose(scope.WithOpName(name), Reshape(scope, Transpose(scope, input, {0,3,1,2}), new_shape), {0,2,3,1}));
    return this;
}
Network *Network::lrn(int radius, float alpha, float beta, std::string name, float bias)
{
    Input input = inputs.front();
    add_output(name, LRN(scope.WithOpName(name), input, LRN::DepthRadius(radius).Alpha(alpha).Beta(beta).Bias(bias)));
    return this;
}
Network *Network::fc(int num_out, std::string name, bool relu, bool trainable)
{
    int i, dim; float standard_deviation;
    std::vector<Tensor> outputs;
    std::vector<Output> feed_in;
    Output input = inputs.front();
    TF_CHECK_OK(session.Run(feed_map, {Shape(scope, input)}, &outputs));
    auto dimensions = outputs[0].vec<int>();
    std::cout << name << ": " << inputs.front().name() << std::endl << dimensions << std::endl;
    const int input_dims = dimensions.dimension(0);
    if(input_dims == 4)
    {
        dim = 1; for(i=1; i<input_dims; i++) dim *= dimensions(i);
        std::cout << name << ": " << dim << std::endl;
        feed_in.push_back(Reshape(scope, Transpose(scope, input, {0,3,1,2}), {-1, dim}));
    }
    else
    {
        assert(input_dims == 2);
        dim = dimensions(1);
        feed_in.push_back(input);
    }
    if(name == "bbox_pred")
    {
        standard_deviation = 0.001f;
    }
    else
    {
        standard_deviation = 0.01f;
    }
    Output weights = get_weight(name, {dim, num_out}, trainable);
    Output biases = get_bias(name, {num_out}, trainable);
    std::cout << weights.name() << std::endl;
    std::cout << biases.name() << std::endl;
    std::cout << dim << ", " << num_out << std::endl;
    TF_CHECK_OK(session.Run({Assign(scope, weights, Multiply(scope, TruncatedNormal(scope, {dim, num_out}, DT_FLOAT), Const(scope, standard_deviation)))}, NULL));
    TF_CHECK_OK(session.Run({Assign(scope, biases, Tensor(DT_FLOAT, {num_out}))}, NULL));
    if(relu == true)
    {
        add_output(name, Relu(scope.WithOpName(name), Add(scope, MatMul(scope, feed_in.back(), weights), biases)));
    }
    else
    {
        add_output(name, Add(scope.WithOpName(name), MatMul(scope, feed_in.back(), weights), biases));
    }
    return this;
}
Network *Network::softmax(std::string name)
{
    Input input = inputs.front();
    std::vector<Tensor> outputs;
    auto input_shape = Shape(scope, input);
    TF_CHECK_OK(session.Run(feed_map, {input_shape}, &outputs));
    auto dimensions = outputs[0].vec<int>();
    std::cout << name << ": " << inputs.front().name() << std::endl << dimensions << std::endl;
    if(name == "rpn_cls_prob")
    {
        Output dim1 = Slice(scope, input_shape, {1}, {1});
        Output dim2 = Slice(scope, input_shape, {2}, {1});
        Output dim3 = Slice(scope, input_shape, {3}, {1});
        auto new_shape = Concat(scope, {(Output)Const(scope, {-1}), dim1, dim2, dim3}, 0);
        add_output(name, Reshape(scope.WithOpName(name), Softmax(scope, Reshape(scope, input, Concat(scope, {(Output)Const(scope, {-1}), dim3}, 0))), new_shape));
    }
    else
    {
        add_output(name, Softmax(scope.WithOpName(name), input));
    }
    return this;
}
Network *Network::dropout(float keep_prob, std::string name)
{
    Input input = inputs.front();
    add_output(name, Multiply(scope.WithOpName(name), Div(scope, input, Const(scope, keep_prob)), Cast(scope, Less(scope, RandomUniform(scope, Shape(scope, input), DT_FLOAT), Const(scope, keep_prob)), DT_FLOAT)));
    return this;
}
Network *Network::timer(std::string name)
{
    Output input = inputs.back();
    auto timer = Time(scope.WithOpName(name), input);
    timers.push_back(timer.time);
    add_output(name, timer.output_data);
    return this;
}
Network *Network::timer_leaf(std::string name)
{
    Output input = inputs.back();
    auto timer = Time(scope.WithOpName(name), input);
    timers.push_back(timer.time);    
    return this;
}