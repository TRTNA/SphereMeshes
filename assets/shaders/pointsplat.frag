#version 410 core
out vec4 FragColor;

in vec3 vNormal;

const vec4 diffuseColor = vec4(0.8f, 0.4f, 0.2f, 1.0f);

subroutine vec4 ColouringType();

subroutine(ColouringType)
vec4 diffuseColouring() {
    return diffuseColor;
}

subroutine(ColouringType)
vec4 normalColouring() {
    vec3 scaledNormal = (vNormal + 1.0) / 2.0;
    return vec4(scaledNormal, 1.0);
}

subroutine uniform ColouringType colouringSubroutine;

void main()
{
    FragColor = colouringSubroutine();
}