#version 410 core
layout (location = 0) in vec3 aPos;

uniform vec3 capsA;
uniform vec3 capsB;
uniform float radius;

out bool toDiscard;
out vec3 normal;

void main()
{
    vec3 BminusA = capsB - capsA;
    float BminusAsqrd = dot(BminusA, BminusA);
    float k = dot(aPos - capsA, BminusA) / BminusAsqrd;
    float clampedK = clamp(k, 0.0, 1.0); 
    vec3 C = A + k*BminusA;
    float distSqrd = dot(C - aPos, C - aPos);
    if (distSqrd <= radius*radius) {
        toDiscard = false;
        gl_Position = vec4(C + radius*(C - aPos) , 1.0);
        normal = normalize(C - aPos);
    } else {
        toDiscard = true;
    }

}