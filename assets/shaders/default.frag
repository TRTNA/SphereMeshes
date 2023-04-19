#version 410 core

out vec4 FragColor;

in vec3 interpColor;
in vec3 vNormal;
in vec3 vLightDir;
in vec3 vPos;

flat in int fragDimensionality;


subroutine vec4 ColoringType();

vec3 colors[3] =  vec3[3](vec3(1.0f, 0.0f, 0.0f), vec3(0.0f, 1.0f, 0.0f), vec3(0.0f, 0.0f, 1.0f));

subroutine(ColoringType)
vec4 diffuseColoring() {
    return vec4(colors[fragDimensionality], 1.0);
}

subroutine(ColoringType)
vec4 normalColoring() {
    vec3 scaledNormal = vNormal * 0.5 + 0.5;
    return vec4(scaledNormal, 1.0);
}

subroutine uniform ColoringType coloringSubroutine;

void main()
{
    if (dot(vPos, vNormal) < 0.0) discard;
    FragColor = coloringSubroutine();
}
