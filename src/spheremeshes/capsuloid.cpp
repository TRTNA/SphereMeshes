#include <spheremeshes/capsuloid.h>

Capsuloid::Capsuloid(uint s0, uint s1) : s0(s0), s1(s1) {}

void Capsuloid::updateFactor(float factor) {
    this->factor = factor;
}

std::ostream& operator<<(std::ostream& ost, const Capsuloid& val) {
    ost << val.s0 << " ";
    ost << val.s1;
    return ost;

}

std::istream& operator>>(std::istream& ist, Capsuloid& val) {
    ist >> val.s0 >> val.s1;
    return ist;
}