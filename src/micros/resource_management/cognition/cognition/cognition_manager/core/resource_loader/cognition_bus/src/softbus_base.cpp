#include "cognition_bus/softbus_base.h"

namespace cognition_bus {

bool SoftBusBase::async_call(vector<boost::any>& inputs, vector<boost::any>& results, int duration)
{
    //creat async thread
    bool res = this->_creat_async_thread(inputs);

    if (res) {
        std::cerr << "creat_async_thread succ!" << std::endl;
        if( this->_get_wait_for(duration)){ // wait for xx seconds.
            std::cerr << "Detecte succ!" << std::endl;
            results = results_;
            return true;
        }
        else {
            std::cerr << "async detector time out!" << std::endl;
            return false;
        }
    }
    else {
        std::cerr << "ERROR : async detector time out and no object detected!" << std::endl;
        return false;
    }
}


bool SoftBusBase::_creat_async_thread(vector<boost::any>& inputs) {
    results_.clear();

    int insert_id = _get_insert_id();
    if(insert_id == -1)
        return false;

    std::future<bool> fut = std::async(std::launch::async, &SoftBusBase::call, this, std::ref(inputs), std::ref(results_));

    current_fut_id_ = _buf_insert(fut, insert_id);

    return true;
}

bool SoftBusBase::_get_wait_for(int duration) {
    std::future_status status;
    status = future_buf_[current_fut_id_].wait_for(std::chrono::seconds(duration));
    if (status == std::future_status::ready) {
        //std::cout << "ready!" <<endl;
        return true;
    }
    if (status == std::future_status::deferred) {
        //std::cout << "deferred" <<endl;
        return false;
    }
    if (status == std::future_status::timeout) {
        //std::cout << "timeout" <<endl;
        return false;
    }
}

int SoftBusBase::_buf_insert(std::future<bool>& fut, int insert_id) {
    future_buf_[insert_id] = std::move(fut);
    current_fut_id_ = insert_id;
    insert_fut_id_++;
    return  current_fut_id_;
}

int SoftBusBase::_get_insert_id() {
    if (insert_fut_id_ < 1) {
        return  insert_fut_id_;
    }

    std::future_status status;
    for (int i = 0; i < 1; ++i) {
        status = future_buf_[i].wait_for(std::chrono::milliseconds(1));
        if (status == std::future_status::ready) {
            return i;
        }
    }
    //no ready thread can use
    return -1;
}


}//namespace
