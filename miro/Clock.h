#include <vector>
#include <time.h>


class Clock {
public:
	void start();

	float stop();

protected:
	std::vector<struct timespec> vStart;
};