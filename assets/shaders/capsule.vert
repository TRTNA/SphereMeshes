#version 410 core
layout (location = 0) in vec3 aPos;

uniform vec3 capsA;
uniform vec3 capsB;
uniform float radiusA;
uniform float radiusB;

uniform mat4 projectionMatrix;
uniform mat4 viewMatrix;
uniform mat3 normalMatrix;
uniform mat4 modelMatrix;

out vec3 normal;

void main()
{
    
    vec3 BminusA = capsB - capsA;
    float BminusAsqrd = dot(BminusA, BminusA);
    float k = dot(aPos - capsA, BminusA) / BminusAsqrd;
    float clampedK = clamp(k, 0.0, 1.0); 
    vec3 C = capsA + clampedK*BminusA;
    float interpRadius = radiusA * (1.0 - clampedK) + radiusB * clampedK;
    normal = normalize(aPos - C);
    gl_Position = viewMatrix * modelMatrix * vec4(C + interpRadius*normal, 1.0);
    normal = normalMatrix * normal;


}