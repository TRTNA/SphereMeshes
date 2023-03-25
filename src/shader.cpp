#include <utils/shader.h>
#include <fstream>
#include <sstream>
#include <iostream>

using std::cerr;
using std::cout;
using std::endl;
using std::ifstream;
using std::string;
using std::stringstream;

Shader::Shader(const GLchar *vertexPath, const GLchar *fragmentPath)
{
    string vertexCode;
    GLuint vertex;
    bool vertexShaderLoaded = loadShaderCode(vertexPath, vertexCode);
    bool vertexShaderCompiled = false;
    if (!vertexShaderLoaded)
    {
        cerr << "Error while reading vertex shader file at " << vertexPath << endl;
    } else {
        const GLchar *vShaderCode = vertexCode.c_str();
        vertexShaderCompiled = compileShader(vShaderCode, ShaderType::VERTEX, vertex);
    }

    string fragmentCode;
    GLuint fragment;
    bool fragmentShaderLoaded = loadShaderCode(fragmentPath, fragmentCode);
    GLint fragmentShaderCompiled = false;
    if (!fragmentShaderLoaded)
    {
        cerr << "Error while loading fragment shader file at " << fragmentPath << endl;
    } else {
        const GLchar *fShaderCode = fragmentCode.c_str();
        fragmentShaderCompiled = compileShader(fShaderCode, ShaderType::FRAGMENT, fragment);
    }

    if (vertexShaderCompiled && fragmentShaderCompiled)
    {
        this->Program = glCreateProgram();
        glAttachShader(this->Program, vertex);
        glAttachShader(this->Program, fragment);
        glLinkProgram(this->Program);
        GLint success;
        GLchar infoLog[1024];
        glGetProgramiv(Program, GL_LINK_STATUS, &success);
        if (!success)
        {
            glGetProgramInfoLog(Program, 1024, NULL, infoLog);
            std::cerr << "Program linking error\n"
                      << infoLog << std::endl;
        }
    }
    glDeleteShader(vertex);
    glDeleteShader(fragment);
}

void Shader::Use() { glUseProgram(this->Program); }

void Shader::Delete() { glDeleteProgram(this->Program); }


bool Shader::loadShaderCode(const GLchar *path, string &outCode)
{
    ifstream shaderFile;

    shaderFile.exceptions(ifstream::failbit | ifstream::badbit);
    try
    {
        shaderFile.open(path);
        stringstream vShaderStream, fShaderStream;
        stringstream shaderStream;
        shaderStream << shaderFile.rdbuf();
        shaderFile.close();
        outCode = shaderStream.str();
        return true;
    }
    catch (ifstream::failure e)
    {
        if (shaderFile.is_open())
            shaderFile.close();
        return false;
    }
}

bool Shader::compileShader(const GLchar *code, ShaderType type, GLuint &id)
{
    id = glCreateShader(getOpenGLShaderType(type));
    glShaderSource(id, 1, &code, NULL);
    glCompileShader(id);
    GLint success;
    glGetShaderiv(id, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        GLchar infoLog[1024];
        glGetShaderInfoLog(id, 1024, NULL, infoLog);
        std::cerr << shaderTypeStr(type) << " shader compilation error\n"
                  << infoLog << endl;
    }
    return success;
}

GLenum Shader::getOpenGLShaderType(const ShaderType &type) const
{
    switch (type)
    {
    case ShaderType::VERTEX:
        return GL_VERTEX_SHADER;
    case ShaderType::GEOMETRY:
        return GL_GEOMETRY_SHADER;
    case ShaderType::FRAGMENT:
        return GL_FRAGMENT_SHADER;
    default:
        return GL_NONE;
    }
}

static std::string shaderTypeStr(ShaderType type)
{
    switch (type)
    {
    case ShaderType::VERTEX:
        return "Vertex";
    case ShaderType::GEOMETRY:
        return "Geometry";
    case ShaderType::FRAGMENT:
        return "Fragment";
    default:
        return GL_NONE;
    }
}
