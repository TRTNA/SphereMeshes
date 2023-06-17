#version 410 core

out vec4 FragColor;

in vec3 interpColor;
in vec3 vNormal;
in vec3 vPos;

flat in int fragDimensionality;

uniform vec3 vLightDir;
uniform bool backFaceCulling;
uniform float opacity = 1.0f;

uniform vec3 ambientColor = vec3(0.1, 0.0, 0.0);
uniform vec3 diffuseColor = vec3(0.5, 0.0, 0.0);
uniform vec3 specColor = vec3(1.0, 1.0, 1.0);
uniform float shininess = 16.0;


subroutine vec4 ColoringType();

vec3 colors[3] =  vec3[3](vec3(1.0f, 0.0f, 0.0f), vec3(0.0f, 1.0f, 0.0f), vec3(0.0f, 0.0f, 1.0f));

subroutine(ColoringType)
vec4 flatColoring() {
    return vec4(diffuseColor, opacity);
}

vec3 flatNormal() {
    return normalize(cross(dFdx(vPos), dFdy(vPos)));
}

subroutine(ColoringType)
vec4 doubleSidedShadedColoring() {
    float lambertian = dot(vLightDir, vNormal);
    if (gl_FrontFacing) return vec4(ambientColor + diffuseColor.bgr * lambertian, opacity);
    float specular = 0.0;
    if (lambertian > 0.0) {
        vec3 vViewDir = normalize(-vPos);
        vec3 H = normalize(vLightDir + vViewDir);
        float specAngle = max(dot(H, vNormal), 0.0);
        specular = pow(specAngle, shininess);
    } 
    vec3 color = ambientColor + diffuseColor * lambertian + specColor * specular; 
    return vec4(color, opacity);

}

subroutine(ColoringType)
vec4 shadedColoring() {
    float lambertian = dot(vLightDir, vNormal);
    float specular = 0.0;
    if (lambertian > 0.0) {
        vec3 vViewDir = normalize(-vPos);
        vec3 H = normalize(vLightDir + vViewDir);
        float specAngle = max(dot(H, vNormal), 0.0);
        specular = pow(specAngle, shininess);
    } 
    vec3 color = ambientColor + diffuseColor * lambertian + specColor * specular; 
    return vec4(color, opacity);

}

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
    if (backFaceCulling && dot(-vPos, vNormal) < 0.0) discard;
    FragColor = coloringSubroutine();
}
