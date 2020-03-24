
#ifndef _MYTIME_H
#define _MYTIME_H

#include <iostream>
#include <sys/time.h>

class myTimer{
public:
    myTimer();
    ~myTimer();
    double total_time;
    int calls;
    double start_time;
    double diff;
    double average_time;
    void tic();
    double toc(bool average = true);
};

#endif
