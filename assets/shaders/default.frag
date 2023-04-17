#version 410 core
out vec4 FragColor;

in vec3 interpColor;
in vec3 vNormal;
in vec3 vLightDir;
in vec4 vPos;

void main()
{
    //FragColor = vec4(dot(vLightDir,vNormal) * interpColor, 1.0);
    FragColor = vec4(interpColor, 1.0);
}