#include <physics/physics_spheremesh.h>
#include <utils/common.h>

#include <physics/constraint.h>

#include <glm/gtx/string_cast.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/vector_angle.hpp>

PhysicsSphereMesh::PhysicsSphereMesh(std::shared_ptr<SphereMesh> sphereMesh) : sphereMesh(sphereMesh)
{
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
/*     for (auto &c : constraints)
    {
        c->enforce();
    } */

    for (auto& p : particles) {
        if (p.pos.y - 0.5f  < -2.0f) {
            p.pos.y = -1.5f;
        }   
    }

    modelMatrix = computeModelMatrix();
    for (int i = 0; i < localSpaceVectors.size(); i++)
    {
        glm::vec3 l = localSpaceVectors.at(i);
        l = glm::vec3(modelMatrix * glm::vec4(l, 1.0f));
        auto &p = particles.at(i);
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


glm::mat3 fromToRotate(glm::vec3 from, glm::vec3 to) {
    glm::vec3 axis = glm::cross(from, to);
    if (glm::dot(axis, axis) < EPSILON) return glm::mat3(1.0f);
    float angle = glm::angle(glm::normalize(from), glm::normalize(to));
    return glm::mat3(glm::rotate(angle, axis));
}

glm::mat3 fromToRotate(glm::vec3 from1, glm::vec3 to1, glm::vec3 from2, glm::vec3 to2) {
     glm::mat3 rotMatrixA = fromToRotate(from1, to1);
     from2 = rotMatrixA * from2;
     to2 = glm::cross(to1, glm::cross(to2, to1));
     from2 = glm::cross(to1, glm::cross(from2, to1));
    glm::mat3 rotMatrixB = glm::rotate(glm::angle(glm::normalize(from2), glm::normalize(to2)), to1);
    return rotMatrixB * rotMatrixA;
}

glm::mat4 PhysicsSphereMesh::computeModelMatrix()
{
    glm::vec3 pb(0.0f);
    for (const auto &p : particles)
    {
        pb += (p.getPos() * p.getMass());
    }
    glm::vec3 worldSpaceBarycenter = pb / totalMass;

    
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

    rotMatrix = (rotMatrix + glm::inverseTranspose(rotMatrix)) / 2.0f;
        rotMatrix = (rotMatrix + glm::inverseTranspose(rotMatrix)) / 2.0f;
    rotMatrix = (rotMatrix + glm::inverseTranspose(rotMatrix)) / 2.0f;

/*     rotMatrix[0] = glm::normalize(glm::cross(rotMatrix[1], rotMatrix[2]));
    rotMatrix[1] = glm::normalize(glm::cross(rotMatrix[2], rotMatrix[0]));
    rotMatrix[2] = glm::normalize(glm::cross(rotMatrix[0], rotMatrix[1])); */
   // rotMatrix = glm::mat3(1.0f);
    glm::vec3 translation = worldSpaceBarycenter;
   /*  rotMatrix = glm::transpose(rotMatrix);

    rotMatrix = fromToRotate(localSpaceVectors.at(1), particles.at(1).pos - worldSpaceBarycenter, 
                             localSpaceVectors.at(0), particles.at(0).pos - worldSpaceBarycenter);
 */
    glm::mat4 result = glm::mat4(rotMatrix);
    
    result[3] = glm::vec4(translation, 1.0f);
    return result;
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
    }
    for (const auto &caps : sphereMesh->capsuloids)
    {
        float m1 = computeVolume(sphereMesh->spheres[caps.s0]);
        glm::vec3 center1 = sphereMesh->spheres[caps.s0].center;
        pb += m1 * center1;
        M += m1;
        particles.emplace_back(center1, glm::vec3(0.0f), m1);

        float m2 = computeVolume(sphereMesh->spheres[caps.s1]);
        glm::vec3 center2 = sphereMesh->spheres[caps.s1].center;
        pb += m2 * sphereMesh->spheres[caps.s1].center;
        M += m2;
        particles.emplace_back(center2, glm::vec3(0.0f), m2);
    }

    for (const auto &st : sphereMesh->sphereTriangles)
    {
        float m1 = computeVolume(sphereMesh->spheres[st.vertices[0]]);
        glm::vec3 center1 = sphereMesh->spheres[st.vertices[0]].center;
        pb += m1 * sphereMesh->spheres[st.vertices[0]].center;
        M += m1;
        particles.emplace_back(center1, glm::vec3(0.0f), m1);

        float m2 = computeVolume(sphereMesh->spheres[st.vertices[1]]);
        glm::vec3 center2 = sphereMesh->spheres[st.vertices[1]].center;
        pb += m2 * sphereMesh->spheres[st.vertices[1]].center;
        M += m2;
        particles.emplace_back(center2, glm::vec3(0.0f), m2);

        float m3 = computeVolume(sphereMesh->spheres[st.vertices[2]]);
        glm::vec3 center3 = sphereMesh->spheres[st.vertices[2]].center;
        pb += m3 * sphereMesh->spheres[st.vertices[2]].center;
        M += m3;
        particles.emplace_back(center3, glm::vec3(0.0f), m3);
    }
    localSpaceBarycenter = pb / M;
    totalMass = M;

    for (auto &p : particles)
    {
        localSpaceVectors.emplace_back(p.getPos() - localSpaceBarycenter);
        // REMOVE
        // p.pinned = true;
    }
    localSpaceBarycenter = glm::vec3(0.0f);
}
