#version 410 core
layout (location = 0) in vec3 position;
layout (location = 1) in vec3 normal;
layout (location = 2) in vec3 color;

uniform mat4 modelMatrix;
uniform mat4 viewMatrix;
uniform mat4 projectionMatrix;
uniform mat3 normalMatrix;

vec3 lightDir = vec3(0.0f, 1.0f, 0.0f);

out vec3 vPos;
out vec3 vNormal;
out vec3 vLightDir;
out vec3 interpColor;

void main()
{
    vec4 vPosVec4 = viewMatrix * modelMatrix * vec4(position, 1.0);
    vPos = vec3(-vPosVec4);
    gl_Position = projectionMatrix * vPosVec4;
    vNormal = normalize(normalMatrix * normal);
    vLightDir = vec3(normalize(projectionMatrix * viewMatrix * modelMatrix * vec4(lightDir, 0.0)));
    interpColor = color;
}