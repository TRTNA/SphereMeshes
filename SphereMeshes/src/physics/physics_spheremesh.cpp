#include <physics/physics_spheremesh.h>
#include <utils/common.h>

#include <physics/constraint.h>

#include <glm/gtx/string_cast.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/vector_angle.hpp>

#include <physics/plane_constraint.h>
#include <utils/common.h>

PhysicsSphereMesh::PhysicsSphereMesh(std::shared_ptr<SphereMesh> sphereMesh, glm::vec3 translation) : sphereMesh(sphereMesh), totalMass(0.0f), modelMatrix(1.0f)
{
    if (sphereMesh->spheres.size() == 1) {
        type = PhysicsSphereMeshType::ONE_SPHERE;
    }
    else if (sphereMesh->spheres.size() == 2 && (sphereMesh->capsuloids.size() == 1 || sphereMesh->singletons.size() == 2)) {
        type = PhysicsSphereMeshType::TWO_SPHERES;
        twoSpheresDist = glm::length(sphereMesh->spheres.at(0).center - sphereMesh->spheres.at(1).center);
    }
    else {
        type = PhysicsSphereMeshType::GENERIC;
    }


    for (const auto& s : sphereMesh->spheres) {
        float mass = computeVolume(s);
        particles.emplace_back(s.center + translation, glm::vec3(0.0f), mass);
        radii.emplace_back(s.radius);
        totalMass += mass;
    }

}

void PhysicsSphereMesh::addForce(const glm::vec3 &forceVec)
{
    for (auto &p : particles)
    {
        p.addForce(forceVec);
    }
}
void PhysicsSphereMesh::enforceConstraints()
{
     for (auto &c : constraints)
     {
        c->enforce();
     }

     if (type == PhysicsSphereMeshType::ONE_SPHERE) {
         modelMatrix = glm::mat4(1.0f);
         modelMatrix[3] = glm::vec4(particles.at(0).getPos(), 1.0f);
     }
     else if (type == PhysicsSphereMeshType::TWO_SPHERES) {
         twoSphereEnforce();
     }
     else {
         nSpheresEnforce();
     }
}

void PhysicsSphereMesh::twoSphereEnforce() {
    Particle& p1 = particles.at(0);
    Particle& p2 = particles.at(1);
    glm::vec3 v = p2.getPos() - p1.getPos();
    float currDist = glm::length(v);
    v /= currDist;
    float delta = currDist - twoSpheresDist;
    float t = p1.getMass() / (p1.getMass() + p2.getMass());
    glm::vec3 displacementVector1 = (1-t) * delta * v;
    glm::vec3 displacementVector2 = t * delta * -v;
    p1.displace(displacementVector1);
    p2.displace(displacementVector2);
    //update matrix for two spheres
    //from to
    modelMatrix = fromToRotate(sphereMesh->spheres.at(1).center - sphereMesh->spheres.at(0).center, p2.getPos() - p1.getPos());
    modelMatrix[3] = glm::vec4(computeWorldSpaceBarycentre(), 1.0f);
}

void PhysicsSphereMesh::nSpheresEnforce()
{
    modelMatrix = computeModelMatrix();
    for (int i = 0; i < sphereMesh->spheres.size(); i++)
    {
        glm::vec3 localSpaceVector = sphereMesh->spheres.at(i).center;
        glm::vec3 newWorldSpaceVector = glm::vec3(modelMatrix * glm::vec4(localSpaceVector, 1.0f));
        auto& p = particles.at(i);
        p.pos = newWorldSpaceVector;
    }
}

void PhysicsSphereMesh::addConstraint(Constraint *constraint)
{
    constraints.push_back(constraint);
}

float PhysicsSphereMesh::getMass() const
{
    return totalMass;
}
void PhysicsSphereMesh::timeStep()
{
    for (auto &p : particles)
    {
        p.timeStep();
    }
}

glm::mat4 PhysicsSphereMesh::getModelMatrix() const
{
    return modelMatrix;
}

glm::mat4 PhysicsSphereMesh::computeModelMatrix()
{
    glm::vec3 worldSpaceBarycentre = computeWorldSpaceBarycentre();
    
    glm::mat3 rotMatrix = glm::mat3(0.0f);
    for (int i = 0; i < particles.size(); i++)
    {
        const glm::vec3& localSpaceVec = sphereMesh->spheres.at(i).center;
        const glm::vec3 worldSpaceVec = particles.at(i).pos - worldSpaceBarycentre;
        rotMatrix += (glm::outerProduct(localSpaceVec, worldSpaceVec) * particles.at(i).getMass());
    }


    rotMatrix /= totalMass;
    rotMatrix = glm::transpose(rotMatrix);

    // orthonormalization of matrix
    //più la ripeti più diventa precisa
    const int orthoNormIter = 5;
    float det = glm::determinant(rotMatrix);

    for (int i = 0; i < orthoNormIter; i++) {
        rotMatrix = (rotMatrix + glm::inverseTranspose(rotMatrix)) * 0.5f;
    }
 

    //rotMatrix[0] = glm::normalize(glm::cross(rotMatrix[1], rotMatrix[2]));
    //rotMatrix[1] = glm::normalize(glm::cross(rotMatrix[2], rotMatrix[0]));
    //rotMatrix[2] = glm::normalize(glm::cross(rotMatrix[0], rotMatrix[1]));
    //rotMatrix /= glm::determinant(rotMatrix);

    glm::vec3 translation = worldSpaceBarycentre;

    glm::mat4 result = glm::mat4(rotMatrix);
    
    result[3] = glm::vec4(translation, 1.0f);
    return result;
}

glm::vec3 PhysicsSphereMesh::computeWorldSpaceBarycentre() const
{
    glm::vec3 pb(0.0f);
    for (const auto& p : particles)
    {
        pb += (p.getPos() * p.getMass());
    }
    return pb / totalMass;
}