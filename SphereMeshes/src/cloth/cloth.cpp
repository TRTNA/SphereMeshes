#include <cloth/cloth.h>

#include <glm/gtx/string_cast.hpp>
#include <spheremeshes/point.h>

#include <utils/common.h>
#include <utils/plane.h>
#include <physics/particle.h>
#include <physics/constraints.h>
#include <stdio.h>

using glm::vec2;
using glm::vec3;
using std::string;
using std::vector;

Cloth::Cloth(uint dim, float dist) : dim(dim), dimSqrd(dim * dim), dist(dist)
{
    particles = new Particle[dimSqrd];
    for (size_t i = 0; i < dimSqrd; i++)
    {
        vec3 startingPos = vec3((float)(i % dim) * dist, 0, (float)(i / dim) * dist);
        particles[i].setPos(startingPos);
        particles[i].setLastPos(startingPos);
        particles[i].setMass(0.1f);
    }

    bool mustConnectToRight, mustConnectToBottom;
    for (size_t x = 0; x < dim; x++)
    {
        // do not connect to bottom when last row
        bool topBorder = x == dim - 1;
        for (size_t y = 0; y < dim; y++)
        {
            // do not connect to right when last col
            bool rightBorder = y == dim - 1;
            if (!rightBorder && !topBorder)
            {
                Particle *p1 =  &particles[linearizedIndexSquareGrid(dim, x, y)];
                Particle *p1R = &particles[linearizedIndexSquareGrid(dim, x, y + 1)];
                Particle *p1B = &particles[linearizedIndexSquareGrid(dim, x + 1, y)];
                Particle *p1D = &particles[linearizedIndexSquareGrid(dim, x + 1, y + 1)];
                constraints.push_back(new ParticleEquidistanceConstraint(p1,  p1R, dist));
                constraints.push_back(new ParticleEquidistanceConstraint(p1,  p1B, dist));
                constraints.push_back(new ParticleEquidistanceConstraint(p1,  p1D, dist * glm::sqrt(2.0f)));
                constraints.push_back(new ParticleEquidistanceConstraint(p1B, p1R, dist * glm::sqrt(2.0f)));
            }
            else if (rightBorder && !topBorder)
            {
                Particle *p1 = &particles[linearizedIndexSquareGrid(dim, x, y)];
                Particle *p1B = &particles[linearizedIndexSquareGrid(dim, x + 1, y)];
                constraints.push_back(new ParticleEquidistanceConstraint(p1, p1B, dist));
            }
            else if (topBorder && !rightBorder)
            {
                Particle *p1 = &particles[linearizedIndexSquareGrid(dim, x, y)];
                Particle *p1R = &particles[linearizedIndexSquareGrid(dim, x, y + 1)];
                constraints.push_back(new ParticleEquidistanceConstraint(p1, p1R, dist));
            }
            else
            {
                // bottom-right --> do nothing for now
            }
        }
    }
    //Pinning
/*     
        particles[0].pin();
        particles[dim - 1].pin();   */
    
}
Cloth::~Cloth()
{
    delete[] particles;
}

void Cloth::addConstraint(Constraint * constraint) {
    constraints.push_back(constraint);
}

void Cloth::transform(const glm::mat4 &matrix)
{
    for (size_t i = 0; i < dimSqrd; i++)
    {
        const glm::vec3 &pos = particles[i].getPos();
        const glm::vec3 &lastPos = particles[i].getLastPos();
        particles[i].setPos(glm::vec3(matrix * glm::vec4(pos, 1.0f)));
        particles[i].setLastPos(glm::vec3(matrix * glm::vec4(lastPos, 1.0f)));
    }
}

uint Cloth::getParticles(Particle *&outParticles)
{
    outParticles = particles;
    return dim;
}

void Cloth::enforceConstraints()
{
    const uint maxTries = 5U;
    for (size_t i = 0; i < maxTries; i++)
    {
        for (auto &c : constraints)
        {
            c->enforce();
        }
    }
}


std::string Cloth::toString() const
{
    string s;
    s += "Points:\n";
    for (size_t i = 0; i < dim; i++)
    {
        for (size_t j = 0; j < dim; j++)
        {
            s += glm::to_string(particles[linearizedIndexSquareGrid(dim, i, j)].getPos());
        }
        s += "\n";
    }
    return s;
}

float Cloth::getMass() const
{
    float accumMass = 0.0f;
    for (size_t p = 0; p < dimSqrd; p++)
    {
        accumMass += particles[p].getMass();
    }
    return accumMass;
}

void Cloth::addForce(const glm::vec3 &force)
{
    for (size_t i = 0; i < dimSqrd; i++)
    {
        particles[i].addForce(force);
    }
}
void Cloth::timeStep()
{
    for (size_t i = 0; i < dimSqrd; i++)
    {
        particles[i].timeStep();
    }
}
