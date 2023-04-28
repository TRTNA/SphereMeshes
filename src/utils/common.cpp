#include <utils/common.h>

bool isInRangeIncl(float n, float min, float max) {
    return n >= min && n <= max;
}

bool isInRangeExcl(float n, float min, float max) {
    return n > min && n < max;
}