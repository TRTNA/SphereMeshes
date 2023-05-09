#pragma once

#include <vector>
#include <utility>
#include <string>


#include <glm/glm.hpp>

typedef unsigned int uint;
typedef std::pair<glm::uvec2, glm::uvec2> SpringEdge;
struct Point;

class Cloth {
    private:
        Point** points;
        uint dim;
        float dist;
        std::vector<SpringEdge> edges;
        bool enforceConstraint(glm::vec3& p1, glm::vec3& p2);
    public:
        Cloth(uint dim, float dist);
        ~Cloth();
        std::string toString() const;
        void enforceConstraints();
        uint getPoints(Point**& outPoints);
        std::vector<SpringEdge> getEdges() const;
};

SpringEdge connectToRight(uint x, uint y);
SpringEdge connectToBottom(uint x, uint y);