#pragma once
#include <glad/glad.h>

#include <glm/glm.hpp>

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <utils/mesh.h>

#include <vector>
#include <string>

class Model
{
public:
    std::vector<Mesh*> meshes;

    Model(const Model &model) = delete;
    Model &operator=(const Model &copy) = delete;

    Model &operator=(Model &&move) noexcept = default;
    Model(Model &&model) = default;

    Model(const std::string &path);
    void Draw() const;

private:
    void loadModel(const std::string& path);
    void processNode(aiNode *node, const aiScene *scene);
    Mesh* processMesh(const aiScene *scene, aiMesh *mesh);

};
