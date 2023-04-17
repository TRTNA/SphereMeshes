#version 410 core
layout (location = 0) in vec3 position;
layout (location = 1) in vec3 normal;
layout (location = 2) in vec3 color;

uniform mat4 modelMatrix;
uniform mat4 viewMatrix;
uniform mat4 projectionMatrix;
uniform mat3 normalMatrix;

vec3 lightDir = vec3(0.0f, 1.0f, 0.0f);

out vec3 vNormal;
out vec3 vLightDir;
out vec3 interpColor;

void main()
{
    gl_Position = projectionMatrix * viewMatrix * modelMatrix * vec4(position, 1.0);
    vNormal = normalize(normalMatrix * normal);
    vLightDir = vec3(normalize(projectionMatrix * viewMatrix * modelMatrix * vec4(lightDir, 0.0)));
    interpColor = color;
}