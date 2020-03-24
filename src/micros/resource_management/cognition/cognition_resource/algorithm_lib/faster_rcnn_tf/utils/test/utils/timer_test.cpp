#include <unistd.h>
#include "utils/myTimer.h"

int main()
{
    myTimer test_time;
    test_time.tic();
    std::cout << test_time.start_time << std::endl;
    sleep(2);
    long long a = 0;
    a = test_time.toc(true);
	long long b = 0, c = 0;
	sleep(3);
	b = test_time.toc(false);
	c = test_time.toc(true);
	long long d =test_time.toc(false);
    std::cout << a << std::endl;
	std::cout << b << std::endl;
	std::cout << c << " " << d<< std::endl;
	std::cout << test_time.total_time << "  " << test_time.calls << std::endl;
}
