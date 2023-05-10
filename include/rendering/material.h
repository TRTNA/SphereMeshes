#pragma once

#include <glm/vec4.hpp>
//TODO stavo implementando material per avere info colore e altri parametri per ogni IglRenderable
// ho implementato camera semplice per viewMatrix e projectionMatrix
// poi nel renderer ci sarà il setting di tutti gli uniform e il rendering degli oggetti
struct Material {
    glm::vec4 diffuseColor;
    glm::vec4 specularColor;
    float shininess;
};