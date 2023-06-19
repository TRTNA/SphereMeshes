#include <physics/physics_spheremesh.h>
#include <utils/common.h>

#include <physics/constraint.h>

#include <glm/gtx/string_cast.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/vector_angle.hpp>

#include <physics/plane_constraint.h>
#include <utils/common.h>

PhysicsSphereMesh::PhysicsSphereMesh(std::shared_ptr<SphereMesh> sphereMesh) : sphereMesh(sphereMesh)
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
    setup();

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
         modelMatrix[3] = glm::vec4(particles.at(0).getPos() - sphereMesh->spheres.at(0).center, 1.0f);
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
    //TODO SERVE ESPRIMERE LA SPHEREMESH CON COORDINATE RISPETTO AL BARICENTRO
    glm::vec3 localSpaceBarycenter = sphereMesh->spheres.at(0).center * p1.getMass() + sphereMesh->spheres.at(1).center * p2.getMass();
    localSpaceBarycenter /= (p1.getMass() + p2.getMass());
    modelMatrix[3] = glm::vec4(computeWorldSpaceBarycenter() + localSpaceBarycenter, 1.0f);
}

void PhysicsSphereMesh::nSpheresEnforce()
{
    modelMatrix = computeModelMatrix();
    for (int i = 0; i < localSpaceVectors.size(); i++)
    {
        glm::vec3 l = localSpaceVectors.at(i);
        l = glm::vec3(modelMatrix * glm::vec4(l, 1.0f));
        auto& p = particles.at(i);
        p.pos = l;
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
    glm::vec3 worldSpaceBarycenter = computeWorldSpaceBarycenter();
    
    glm::mat3 rotMatrix = glm::mat3(0.0f);
    for (int i = 0; i < particles.size(); i++)
    {
        const glm::vec3& localSpaceVec = localSpaceVectors.at(i);
        const glm::vec3 worldSpaceVec = particles.at(i).pos - worldSpaceBarycenter;
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

    glm::vec3 translation = worldSpaceBarycenter;

    glm::mat4 result = glm::mat4(rotMatrix);
    
    result[3] = glm::vec4(translation, 1.0f);
    return result;
}

glm::vec3 PhysicsSphereMesh::computeWorldSpaceBarycenter() const
{
    glm::vec3 pb(0.0f);
    for (const auto& p : particles)
    {
        pb += (p.getPos() * p.getMass());
    }
    return pb / totalMass;
}

void PhysicsSphereMesh::setup()
{
    particles.clear();
    // somma pesata dei vettori posizione in spazio oggetto
    float M = 0.0f;
    glm::vec3 pb(0.0f);
    for (const auto &idx : sphereMesh->singletons)
    {
        float m = computeVolume(sphereMesh->spheres[idx]);
        glm::vec3 center = sphereMesh->spheres[idx].center;
        pb += m * center;
        M += m;
        particles.emplace_back(center, glm::vec3(0.0f), m);
        radii.emplace_back(sphereMesh->spheres[idx].radius);
    }
    for (const auto &caps : sphereMesh->capsuloids)
    {
        float m0 = computeVolume(sphereMesh->spheres[caps.s0]);
        glm::vec3 center0 = sphereMesh->spheres[caps.s0].center;
        pb += m0 * center0;
        M += m0;
        particles.emplace_back(center0, glm::vec3(0.0f), m0);
        radii.emplace_back(sphereMesh->spheres[caps.s0].radius);

        float m1 = computeVolume(sphereMesh->spheres[caps.s1]);
        glm::vec3 center1 = sphereMesh->spheres[caps.s1].center;
        pb += m1 * center1;
        M += m1;
        particles.emplace_back(center1, glm::vec3(0.0f), m1);
        radii.emplace_back(sphereMesh->spheres[caps.s1].radius);

    }

    for (const auto &st : sphereMesh->sphereTriangles)
    {
        float m0 = computeVolume(sphereMesh->spheres[st.vertices[0]]);
        glm::vec3 center0 = sphereMesh->spheres[st.vertices[0]].center;
        pb += m0 * center0;
        M += m0;
        particles.emplace_back(center0, glm::vec3(0.0f), m0);
        radii.emplace_back(sphereMesh->spheres[st.vertices[0]].radius);


        float m1 = computeVolume(sphereMesh->spheres[st.vertices[1]]);
        glm::vec3 center1 = sphereMesh->spheres[st.vertices[1]].center;
        pb += m1 * center1;
        M += m1;
        particles.emplace_back(center1, glm::vec3(0.0f), m1);
        radii.emplace_back(sphereMesh->spheres[st.vertices[1]].radius);


        float m2 = computeVolume(sphereMesh->spheres[st.vertices[2]]);
        glm::vec3 center2 = sphereMesh->spheres[st.vertices[2]].center;
        pb += m2 * center2;
        M += m2;
        particles.emplace_back(center2, glm::vec3(0.0f), m2);
        radii.emplace_back(sphereMesh->spheres[st.vertices[2]].radius);

    }
    glm::vec3 localSpaceBarycenter = pb / M;
    totalMass = M;

    for (auto &p : particles)
    {
        localSpaceVectors.emplace_back(p.getPos() - localSpaceBarycenter);
        //REMOVE
        //p.pinned = true;
    }
}
