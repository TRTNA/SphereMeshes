#include <utils/glrend_spheremesh.h>

using std::vector;

GlRendSphereMesh::GlRendSphereMesh(std::vector<Sphere>& pSpheres, std::vector<Edge>& pEdges, std::vector<Triangle>& pTriangles, unsigned int pointsNumber) 
: SphereMesh() {}

void GlRendSphereMesh::Draw(const Shader &shader)
{
}
