#ifndef CognitionSoftBusBase_H
#define CognitionSoftBusBase_H

#include <string>
#include <vector>
#include <future>
#include <iostream>
#include <boost/any.hpp>

/**
 * @brief namespace cognition_bus
 */
namespace cognition_bus {
using namespace std;

/**
 * @brief The SoftBusBase class
 * Use base classes to create subclasses of different categories
 *
 * @note Naming conventions
 * 1.Xxx is the categorie of bus;
 * 2.All kinds of subclasses must named as
 *   "class SoftBusXxx";
 * 3.All micROS Packages about subclasses must named as
 *   "cognition_bus_xxx";
 * 4.micROS Packages "cognition_bus_xxx" must located in
 *   "../cognition_softbus/core/framwork/"
 */
class SoftBusBase{
public:
    virtual ~SoftBusBase()=default;

    virtual bool call(vector<boost::any>& inputs, vector<boost::any>& results) = 0;

    bool async_call(vector<boost::any>& inputs, vector<boost::any>& results, int duration);

private:
    bool _creat_async_thread(vector<boost::any>& inputs);
    bool _get_wait_for(int duration);
    int  _buf_insert(std::future<bool>& fut, int insert_id);
    // check and return right buffer id
    int  _get_insert_id();

    int buf_size = 10;
    std::future<bool> future_buf_[2];
    int current_fut_id_ = -1;
    int insert_fut_id_ = 0;

    vector<boost::any> results_;
    //boost::mutex callMutex_;
    //boost::unique_lock<boost::mutex> callLock(callMutex_);
    //boost::mutex::scoped_lock lock(callMutex_);
};

}//namespace


#endif // CognitionSoftBusBase_H
