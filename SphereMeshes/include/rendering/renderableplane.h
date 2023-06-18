#pragma once

#include "iglrenderable.h"

#include <vector>

typedef unsigned int uint;

class Plane;

class RenderablePlane : public IglRenderable
{
private:
	uint VAO, EBO, VBO;

public:
	RenderablePlane(Plane plane, float dim);
		~RenderablePlane();
	void draw() override;
	
};

