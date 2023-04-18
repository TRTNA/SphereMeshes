#version 410 core
out vec4 FragColor;

in vec3 interpColor;
in vec3 vNormal;
in vec3 vLightDir;
in vec4 vPos;


subroutine vec4 ColoringType();

subroutine(ColoringType)
vec4 diffuseColoring() {
    return interpColor;
}

subroutine(ColoringType)
vec4 normalColoring() {
    vec3 scaledNormal = (vNormal + 1.0) / 2.0;
    return vec4(scaledNormal, 1.0);
}

subroutine uniform ColoringType coloringSubroutine;

void main()
{
    FragColor = coloringSubroutine();
}
