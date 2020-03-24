#include <sys/time.h>
#include "utils/myTimer.h"

myTimer::myTimer()
{
    total_time = 0;
    calls = 0;
    start_time = 0;
    diff = 0;
    average_time = 0;
}
myTimer::~myTimer()
{

}

void myTimer::tic()
{
    struct timeval tv;
    gettimeofday(&tv,NULL);
    start_time = (tv.tv_sec)*1000.0+ tv.tv_usec/1000.0;
    //return start_time;
}

double  myTimer::toc(bool average)
{
    struct timeval tv;
    gettimeofday(&tv,NULL);
    double my_time = (tv.tv_sec)*1000.0 + tv.tv_usec/1000.0;
    diff = my_time - start_time;
    total_time += diff;
    calls += 1;
    average_time = total_time/calls;
    if(average)
    {
        return average_time;
    }
    else
        return diff;
}
