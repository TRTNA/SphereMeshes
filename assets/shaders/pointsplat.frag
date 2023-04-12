#version 410 core
out vec4 FragColor;
in vec3 normal;

const vec4 diffuseColor = vec4(0.8f, 0.4f, 0.2f, 1.0f);

subroutine vec4 ColouringType();

subroutine(ColouringType)
vec4 diffuseColouring() {
    return diffuseColor;
}

subroutine(ColouringType)
vec4 normalColouring() {
    return vec4(normal, 1.0);
}

subroutine uniform ColouringType colouringSubroutine;

void main()
{
    FragColor = colouringSubroutine();
}