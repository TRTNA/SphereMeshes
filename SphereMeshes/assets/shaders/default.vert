#version 410 core
layout (location = 0) in vec3 position;
layout (location = 1) in vec3 normal;
layout (location = 2) in int dimensionality;

uniform mat4 modelMatrix;
uniform mat4 viewMatrix;
uniform mat4 projectionMatrix;
uniform mat3 normalMatrix;

out vec3 vPos;
out vec3 vNormal;
flat out int fragDimensionality;

void main()
{
    vec4 vPosVec4 = viewMatrix * modelMatrix * vec4(position, 1.0);
    vPos = vec3(vPosVec4);
    gl_Position = projectionMatrix * vPosVec4;
    vNormal = normalize(normalMatrix * normal);
    fragDimensionality = dimensionality;
}