//Variation of https://github.com/JoeyDeVries/LearnOpenGL/blob/master/includes/learnopengl/shader.h

#pragma once

#include <glad/glad.h>

#include <string>


enum class ShaderType {
    VERTEX, GEOMETRY, FRAGMENT
};

static std::string shaderTypeStr(ShaderType type);


class Shader
{
public:
    GLuint Program;

    Shader(const GLchar* vertexPath, const GLchar* fragmentPath);
    void Use();

    void Delete();

private:

    bool loadShaderCode(const GLchar *path, std::string& outCode);
    bool compileShader(const GLchar* code, ShaderType type, GLuint& id);
    GLenum getOpenGLShaderType(const ShaderType& type) const;
};
