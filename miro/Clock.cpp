#include "Clock.h"

//timing code adapted from http://stackoverflow.com/questions/2962785/c-using-clock-to-measure-time-in-multi-threaded-programs

void
Clock::start() {
	struct timespec start;
	clock_gettime(CLOCK_MONOTONIC, &start);
	vStart.push_back(start);
}

float
Clock::stop() {
	struct timespec stop, start;
	clock_gettime(CLOCK_MONOTONIC, &stop);
	start = vStart.back();
	vStart.pop_back();
	float seconds = (stop.tv_sec - start.tv_sec);
	float nanoseconds = (stop.tv_nsec - start.tv_nsec) / 1000000000.0;
	return seconds + nanoseconds;
}