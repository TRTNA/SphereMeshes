#pragma once

class Particle;
class Plane;
class Cloth;

void enforceEquidistanceConstraint(Particle& p1, Particle& p2, const float dist);

bool enforceAbovePlaneConstraint(Particle& p1, const Plane& plane);

bool enforceAbovePlaneConstraint(Cloth& c, const Plane& plane);