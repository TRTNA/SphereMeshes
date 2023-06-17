#version 410 core
out vec4 FragColor;

in vec3 vNormal;

const vec4 diffuseColor = vec4(0.8f, 0.4f, 0.2f, 1.0f);

void main()
{
    FragColor = diffuseColor;
}