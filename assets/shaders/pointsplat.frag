#version 410 core
out vec4 FragColor;


flat in uint toDiscard;
in vec3 normal;

void main()
{
    if (toDiscard == 1) discard;
    else FragColor = vec4(1.0f, 0.5f, 0.2f, 1.0f);
}