#version 410 core
layout (location = 0) in vec3 aPos;

vec3 capsA = vec3(-0.5, 0.0, 0);
vec3 capsB = vec3(0.5, 0.0, 0);
float radius = 0.5;

uniform mat4 projectionMatrix;
uniform mat4 viewMatrix;
uniform mat3 normalMatrix;

flat out uint toDiscard;
out vec3 normal;

void main()
{
    
    vec3 BminusA = capsB - capsA;
    float BminusAsqrd = dot(BminusA, BminusA);
    float k = dot(aPos - capsA, BminusA) / BminusAsqrd;
    float clampedK = clamp(k, 0.0, 1.0); 
    vec3 C = capsA + clampedK*BminusA;
    float distSqrd = dot(C - aPos, C - aPos);
    if (distSqrd <= radius*radius) {
        toDiscard = 0;
        normal = normalize(aPos - C);
        gl_Position = viewMatrix * vec4(C + radius*normal, 1.0);
        normal = normalMatrix * normal;
    } else {
        toDiscard = 1;
    }

}