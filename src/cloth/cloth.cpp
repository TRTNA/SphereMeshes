#include <cloth/cloth.h>

#include <glm/gtx/string_cast.hpp>
#include <spheremeshes/point.h>

#include <utils/common.h>
#include <stdio.h>


using std::vector;
using glm::vec3;
using glm::vec2;
using std::string;

Cloth::Cloth(uint dim, float dist) : dim(dim), dist(dist) {
    points = (Point**) malloc(dim*sizeof(points)); 
    for(size_t i = 0; i < dim; i++) {
        points[i] = (Point*)malloc(dim*sizeof(Point));
        for (size_t j = 0; j < dim; j++) {
            points[i][j] = Point(vec3((float)i * dist, (float)j * dist, 0.0f), vec3(0.0f));
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

uint Cloth::getPoints(Point**& outPoints) {
    outPoints = points;
    return dim;
}
std::vector<SpringEdge> Cloth::getEdges() const {
    return edges;
}


bool Cloth::enforceConstraint(glm::vec3& p1, glm::vec3& p2) {
    vec3 v = p2 - p1;
    float currDist = glm::length(v);
    if (isInRangeIncl(currDist, dist - 0.0001f, dist + 0.0001f)) {
        return false;
    }
    v /= currDist;
    float delta = currDist - dist;
    p1 += (0.5f * delta) * v;
    p2 -= (0.5f * delta) * v;
    return true;
}


void Cloth::enforceConstraints() {
    //TODO implementare funzione che mantiene distanza tra posizioni collegate
    bool AllNotDisplaced;
    const uint maxTries = 10U;
    uint tries = 0;
    do {
        AllNotDisplaced = true;
        for (const auto& e : edges) {
            vec3& p1 = points[e.first.x][e.first.y].pos;
            vec3& p2 = points[e.second.x][e.second.y].pos;
            bool displaced = enforceConstraint(p1, p2);
            AllNotDisplaced = AllNotDisplaced && (! displaced); 
        }
        tries++;
    }
    while (! AllNotDisplaced && tries < maxTries);
    printf("Exited with allNotDisplaced = %s and tries = %d\n", AllNotDisplaced ? "true" : "false", tries);
}


std::string Cloth::toString() const {
    string s;
    s+="Points:\n";
    for (size_t i = 0; i < dim; i++) {
        for (size_t j = 0; j < dim; j++) {
            s += glm::to_string(points[i][j].pos);
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
