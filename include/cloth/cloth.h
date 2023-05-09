#pragma once

#include <vector>
#include <utility>
#include <string>

#include <glm/glm.hpp>

typedef unsigned int uint;
typedef std::pair<glm::uvec2, glm::uvec2> SpringEdge;
class Particle;

class Cloth {
    protected:
        Particle* particles;
        uint dim;
        float dist;
    private:
        std::vector<SpringEdge> edges;
        bool enforceConstraint(glm::vec3& p1, glm::vec3& p2);
    public:
        Cloth(uint dim, float dist);
        ~Cloth();
        std::string toString() const;
        void enforceConstraints();
        uint getParticles(Particle*& outParticles);
        std::vector<SpringEdge> getEdges() const;
        void addForce(const glm::vec3& force);
        void timeStep();
};

SpringEdge connectToRight(uint x, uint y);
SpringEdge connectToBottom(uint x, uint y);