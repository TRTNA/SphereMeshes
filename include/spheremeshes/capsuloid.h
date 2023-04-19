#pragma once

#include <utils/types.h>

struct Capsuloid {
    float factor;
    uint s0, s1;
    Capsuloid() = default;
    Capsuloid(uint s0, uint s1);
    Capsuloid(uint s0, uint s1, float factor);
    void updateFactor(float factor);
};

std::ostream& operator<<(std::ostream& ost, const Capsuloid& val);
std::istream& operator>>(std::istream& ost, Capsuloid& val);