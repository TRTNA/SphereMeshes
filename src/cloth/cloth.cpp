#include <cloth/cloth.h>

#include <glm/gtx/string_cast.hpp>

#include <utils/common.h>


using std::vector;
using glm::vec3;
using glm::vec2;
using std::string;

Cloth::Cloth(uint dim, uint dist) : dim(dim), dist(dist) {
    points = (vec3**) malloc(dim*sizeof(points)); 
    for(size_t i = 0; i < dim; i++) {
        points[i] = (vec3*)malloc(dim*sizeof(vec3));
        for (size_t j = 0; j < dim; j++) {
            points[i][j] = vec3(i * dist, j * dist, 0.0f);
        }
    }

    bool mustConnectToRight, mustConnectToBottom;
    for (size_t x = 0; x < dim; x++) {
        //do not connect to bottom when last row
        mustConnectToBottom = x != dim - 1;
        for (size_t y = 0; y < dim; y++) {
            //do not connect to right when last col
            mustConnectToRight = y != dim -1;
            if (mustConnectToRight) edges.push_back(connectToRight(x, y));
            if (mustConnectToBottom) edges.push_back(connectToBottom(x, y));
        }
    } 
}

Cloth::~Cloth() {
    for(size_t i = 0; i < dim; i++) {
        delete points[i];
    }
    delete points;
}

uint Cloth::getPoints(vec3**& outPoints) {
    outPoints = points;
    return dim;
}

bool Cloth::enforceConstraint(glm::vec3& p1, glm::vec3 p2) {
    vec3 v = p1 - p2;
    float currDist = glm::length(v);
    if(isInRangeIncl(currDist, dist - 0.001f, dist + 0.001f)) {
        return true;
    }
    v = glm::normalize(v);
    float delta = currDist - dist;
    p1 += (0.5f * delta) * v;
    p2 += (0.5f * delta) * v;
    return false;
}


void Cloth::enforceConstraints() {
    //TODO implementare funzione che mantiene distanza tra posizioni collegate
    bool allEnforced = false;
    const uint maxTries = 10;
    uint tries = 0;
    while (! allEnforced && tries < maxTries) {
        for (const auto& e : edges) {
            vec3& p1 = points[e.first.x][e.first.y];
            vec3& p2 = points[e.second.x][e.second.y];
            enforceConstraint(p1, p2);
        }
        tries++;
    }

}


std::string Cloth::toString() const {
    string s;
    s+="Points:\n";
    for (size_t i = 0; i < dim; i++) {
        for (size_t j = 0; j < dim; j++) {
            s += glm::to_string(points[i][j]);
        }
        s += "\n";
    }
    s+="Edges\n";
    for (const auto& e : edges) {
        s += glm::to_string(e.first) + " --> " + glm::to_string(e.second) + "\n";
    }
    return s;

}

SpringEdge connectToRight(uint x, uint y) {
    return SpringEdge(glm::vec2(x, y), glm::vec2(x, y + 1));
}
SpringEdge connectToBottom(uint x, uint y) {
    return SpringEdge(glm::vec2(x, y), glm::vec2(x + 1, y));
}

