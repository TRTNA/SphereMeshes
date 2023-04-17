#version 410 core
layout (location = 0) in vec3 position;
layout (location = 1) in vec3 normal

uniform mat4 projectionMatrix;
uniform mat4 viewMatrix;
uniform mat3 normalMatrix;
uniform mat4 modelMatrix;

out vec3 vNormal;

void main()
{
    gl_Position = projectionMatrix * viewMatrix * modelMatrix * vec4(position, 1.0);
    vNormal = normalize(normalMatrix * normal);
}