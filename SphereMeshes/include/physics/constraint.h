#pragma once

struct Constraint {
    virtual void enforce() = 0;
};