#pragma once

#include <glm/vec3.hpp>

class Plane {
    private:
        glm::vec3 normal;
        glm::vec3 origin;
    public:
        Plane(glm::vec3 origin, glm::vec3 normal);
        glm::vec3 getNormal() const;
        glm::vec3 getOrigin() const;
        void setNormal(glm::vec3 normal) ;
        void setOrigin(glm::vec3 origin) ;

        bool contains(glm::vec3 point) const;
        bool isPerpendicular(glm::vec3 vec) const;
};
