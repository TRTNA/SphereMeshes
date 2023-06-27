#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <iostream>
#include <vector>
#include <memory>
#include <string>

#include <spheremeshes/spheremeshes.h>
#include <utils/utils.h>
#include <rendering/rendering.h>
#include <physics/physics.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/string_cast.hpp>

#define OPENGL_VERSION_MAJOR 4
#define OPENGL_VERSION_MINOR 1

using glm::vec3;
using glm::vec4;
using std::cout;
using std::endl;
using std::string;
using std::vector;

GLFWwindow* glfwSetup(std::string windowTitle, float width, float height);
void imGuiSetup(GLFWwindow* window);
void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow* window);
// callback functions for keyboard and mouse events
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mode);
void mouse_callback(GLFWwindow* window, double xpos, double ypos);
void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);
void cursor_position_callback(GLFWwindow* window, double xpos, double ypos);
glm::vec3 get_arcball_vector(int x, int y);
void apply_key_commands();

// Screen data
const unsigned int SCR_WIDTH = 1280;
const unsigned int SCR_HEIGHT = 720;

// I/O data
bool keys[1024];
double last_mx = 0.0, last_my = 0.0, cur_mx = 0.0, cur_my = 0.0;
bool archball = false;

// Time data
float deltaTime = 0.0f;

//Rendering data
Camera camera;
Scene scene;
Renderer renderer;
const glm::vec3 lightDir(0.0f, 1.0f, 0.0f);
const glm::vec3 backgroundColor{0.8f, 0.8f, 0.8f};
const glm::vec3 ambientColor{0.1f, 0.1f, 0.1f};
const glm::vec3 specColor{1.0f, 1.0f, 1.0f};

const glm::vec3 sphereMeshColor{0.4f, 1.0f, 0.4f};
const glm::vec3 planeColor{0.9f, 0.0f, 0.2f};
const glm::vec3 boundingSphereColor{0.0f, 0.0f, 0.5f};

const float shininess = 16.0f;
const float pointsSize = 5.0f;

bool backFaceCulling = false;
bool renderBoundingSphere = false;

//Sphere mesh data
SphereMesh sm;
int pointsNumber = 100000;
int boundingSpherePointsNumber = pointsNumber;

//Physics data
PhysicsEngine engine;
PhysicsSphereMesh* physSphereMesh;
const Plane plane1(glm::vec3(-6.0f, 0.0f, 0.0f), glm::normalize(glm::vec3(0.2f, 1.0f, 0.0f)));
const Plane plane2(glm::vec3(1.0f, 0.0f, 0.0f), glm::normalize(glm::vec3(-0.5f, 1.0f, 0.0f)));

vec3 physSphereMeshStartingPos{0.0f, 10.0f, 0.0f};

int main(int argc, char* argv[])
{
	/*
	* ---------------------------------------------------------
	*   CONTEXT
	* ---------------------------------------------------------
	*/
	GLFWwindow* window = glfwSetup("SphereMeshes", SCR_WIDTH, SCR_HEIGHT);
	if (window == nullptr)
	{
		std::cerr << "Failed to initialize GLFW" << std::endl;
		return -1;
	}

	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
	{
		std::cerr << "Failed to initialize GLAD" << std::endl;
		return -1;
	}

	imGuiSetup(window);

	/*
	* ---------------------------------------------------------
	*   SPHERE MESH LOADING
	* ---------------------------------------------------------
	*/    std::string smToLoad = argc > 1 ? argv[1] : "caps.sm";
	string readErrorMsg;
	bool read = readFromFile("assets/spheremeshes/" + smToLoad, sm, readErrorMsg);
	if (!read)
	{
		std::cerr << readErrorMsg << std::endl;
		return 1;
	}
	cout << "Loaded a sphere mesh:\n"
		<< sm << endl;

	/*
	* ---------------------------------------------------------
	*   PHYSICS
	* ---------------------------------------------------------
	*/

	// Physics engine setup
	engine = PhysicsEngine();
	engine.start();
	cout << "Started physics engine...\n";

	//Physics sphere mesh setup
	physSphereMesh = new PhysicsSphereMesh(std::make_shared<SphereMesh>(sm), physSphereMeshStartingPos);

	//Physics sphere mesh - plane constraints
	PhysSphereMeshPlaneConstraint planeConstr1 = PhysSphereMeshPlaneConstraint(&plane1, physSphereMesh);
	physSphereMesh->addConstraint(&planeConstr1);
	PhysSphereMeshPlaneConstraint planeConstr2 = PhysSphereMeshPlaneConstraint(&plane2, physSphereMesh);
	physSphereMesh->addConstraint(&planeConstr2);

	engine.addObject(physSphereMesh);

	/*
	* ---------------------------------------------------------
	*   RENDERING
	* ---------------------------------------------------------
	*/
	cout << "Setting up rendering and scene...\n";

	// Shader setup
	Shader shader("assets/shaders/default.vert", "assets/shaders/default.frag");

	// Camera setup
	float fovY = 45.0f;
	glm::vec3 viewPos = glm::vec3(0.0f, 4.0f, sm.boundingSphere.radius*5.0f);
	camera = Camera(viewPos, glm::normalize(-viewPos), 0.1f, 100.0f, (float)SCR_WIDTH, (float)SCR_HEIGHT, fovY);

	// Light setup
	Light light(glm::normalize(lightDir));

	// Scene setup
	scene = Scene(&camera, &light);

	// Renderer setup
	renderer = Renderer(&shader);
	renderer.setBackgroundColor(backgroundColor);
	renderer.setAmbientColor(ambientColor);
	renderer.setBackfaceCulling(backFaceCulling);

	std::vector<Plane> shadowPlanes{plane1, plane2};
	renderer.enableShadowing(shadowPlanes, planeColor);

	// Other openGL params
	glPointSize(pointsSize);

	// World axis setup
	Material xAxisMaterial(glm::vec3(1.0f, 0.0f, 0.0f), specColor, shininess, MaterialType::FLAT);
	ArrowLine xAxis(glm::vec3(1.0f, 0.0f, 0.0f));

	ArrowLine yAxis(glm::vec3(0.0f, 1.0f, 0.0f));
	Material yAxisMaterial(glm::vec3(0.0f, 1.0f, 0.0f), specColor, shininess, MaterialType::FLAT);

	ArrowLine zAxis(glm::vec3(0.0f, 0.0f, 1.0f));
	Material zAxisMaterial(glm::vec3(0.0f, 0.0f, 1.0f), specColor, shininess, MaterialType::FLAT);

	glm::mat4 identity(1.0f);
	scene.addObject(&xAxis, &identity, &xAxisMaterial);
	scene.addObject(&yAxis, &identity, &yAxisMaterial);
	scene.addObject(&zAxis, &identity, &zAxisMaterial);

	// Sphere mesh setup
	PointCloud pc = PointCloud();
	pc.repopulate(pointsNumber, sm);
	std::shared_ptr<PointCloud> pc_ptr = std::make_shared<PointCloud>(pc);
	RenderablePointCloud rpc = RenderablePointCloud(pc_ptr);

	Material sphereMeshmat(sphereMeshColor, specColor, shininess, MaterialType::BLINN_PHONG, true);

	glm::mat4 sphereMeshModelMatrix = physSphereMesh->getModelMatrix();

	uint sphereMeshSceneIdx = scene.addObject(&rpc, &sphereMeshModelMatrix, &sphereMeshmat);


	// Sphere mesh's bounding sphere setup
	SphereMesh boundingSphereFakeSm;
	PointCloud boundingSphereFakePc = PointCloud();
	boundingSphereFakePc.repopulate(boundingSpherePointsNumber, sm);
	std::shared_ptr<PointCloud> boundingSphereFakePc_ptr = std::make_shared<PointCloud>(boundingSphereFakePc);
	RenderablePointCloud boundingSphereFakeRpc = RenderablePointCloud(boundingSphereFakePc_ptr);
	Material boundingSphereMat(boundingSphereColor, glm::vec3(1.0f, 1.0f, 1.0f), 1.0f, MaterialType::FLAT);
	//boundingSphereSceneIdx = scene.addObject(&boundingSphereFakeRpc, &sphereMeshModelMatrix, &boundingSphereMat);


	//Renderable plane setup
	RenderablePlane rendPlane1 = RenderablePlane(plane1, 20.0f);
	RenderablePlane rendPlane2 = RenderablePlane(plane2, 20.0f);

	Material rendPlaneMat(planeColor, glm::vec3(1.0f), 6.0f, MaterialType::BLINN_PHONG);
	glm::mat4 rendPlaneModelMat(1.0f);

	scene.addObject(&rendPlane1, &rendPlaneModelMat, &rendPlaneMat);
	scene.addObject(&rendPlane2, &rendPlaneModelMat, &rendPlaneMat);

	float wallTime = 0.0f;


	/*
	* ---------------------------------------------------------
	*   MAIN LOOP
	* ---------------------------------------------------------
	*/
	while (!glfwWindowShouldClose(window))
	{
		//Time
		float currentFrame = glfwGetTime();
		deltaTime = currentFrame - wallTime;
		wallTime = currentFrame;

		//I/O
		glfwPollEvents();
		apply_key_commands();

		//Physics
		const uint maxIter = 5U;
		uint iter = 0U;
		while (!engine.isPaused() && wallTime > engine.getVirtualTime())
		{
			engine.timeStep();
			if (iter++ > maxIter)
			{
				engine.synchronizeWithWallTime(wallTime);
				break;
			}
		}
		//Updating model matrix for static sphere mesh with the model matrix computed by the physics sphere mesh (dynamic)
		glm::mat4* smModelMat = scene.getModelMatrixOf(sphereMeshSceneIdx);
		*smModelMat = physSphereMesh->getModelMatrix();

		//GUI setup
		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();
		ImGui::Begin("Controls");
		if (ImGui::Button("Restart physics sim")) {
			physSphereMesh->reset(physSphereMeshStartingPos);
		}
		float smPos[3] = { physSphereMeshStartingPos.x, physSphereMeshStartingPos.y, physSphereMeshStartingPos.z };
		if (ImGui::SliderFloat3("Sphere mesh starting pos", smPos, -10.0f, 10.0f)) 
			physSphereMeshStartingPos = glm::vec3(smPos[0], smPos[1], smPos[2]);
		
		ImGui::End();


		//Rendering
		renderer.renderScene(&scene);
		ImGui::Render();
		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

		glfwSwapBuffers(window);
	}


	/*
	* ---------------------------------------------------------
	*   CONTEXT CLEANING
	* ---------------------------------------------------------
	*/
	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();
	
	glfwTerminate();
	return 0;
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
	// make sure the viewport matches the new window dimensions; note that width and
	// height will be significantly larger than specified on retina displays.
	glViewport(0, 0, width, height);
}

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mode)
{
	// if ESC is pressed, we close the application
	if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
		glfwSetWindowShouldClose(window, GL_TRUE);

	// we keep trace of the pressed keys
	// with this method, we can manage 2 keys pressed at the same time:
	// many I/O managers often consider only 1 key pressed at the time (the first pressed, until it is released)
	// using a boolean array, we can then check and manage all the keys pressed at the same time
	if (action == GLFW_PRESS)
		keys[key] = true;
	else if (action == GLFW_RELEASE)
		keys[key] = false;
}

void apply_key_commands()
{
	if (keys[GLFW_KEY_A])
	{
		camera.rotateAroundY(-glm::radians(30.0f * deltaTime));
		return;
	}
	if (keys[GLFW_KEY_S])
	{
		camera.translate(-glm::normalize(camera.getForward()) * 1.5f * deltaTime);
		return;
	}
	if (keys[GLFW_KEY_D])
	{
		camera.rotateAroundY(glm::radians(30.0f * deltaTime));
		return;
	}
	if (keys[GLFW_KEY_W])
	{
		camera.translate(glm::normalize(camera.getForward()) * 1.5f * deltaTime);
		return;
	}
	if (keys[GLFW_KEY_Q])
	{
		engine.start();
		return;
	}
	if (keys[GLFW_KEY_E])
	{
		engine.pause();
		return;
	}
}

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{
	static double mouse_x = 0.0;
	static double mouse_y = 0.0;
	glfwGetCursorPos(window, &mouse_x, &mouse_y);
	if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS)
	{
		last_mx = cur_mx = mouse_x;
		last_my = cur_my = mouse_y;
	}

	if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_RELEASE)
	{
		Camera const* sceneCam = scene.getCamera();
		vec3 rayDir = screenToWorldDir(glm::vec2(mouse_x, mouse_y), SCR_WIDTH, SCR_HEIGHT, sceneCam->getViewMatrix(), sceneCam->getProjectionMatrix());
		Ray r = Ray(camera.getPos(), rayDir);
		Point null;
		if (intersects(r, sm.boundingSphere, null))
		{
			physSphereMesh->addImpulse(glm::vec3(0.0f, 3.0f, 0.0f));
		}
	}
}

GLFWwindow* glfwSetup(std::string windowTitle, float width, float height)
{
	// glfw: initialize and configure
	// ------------------------------
	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, OPENGL_VERSION_MAJOR);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, OPENGL_VERSION_MINOR);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

#ifdef __APPLE__
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

	// glfw window creation
	// --------------------
	GLFWwindow* window = glfwCreateWindow(width, height, windowTitle.c_str(), NULL, NULL);
	if (window == NULL)
	{
		std::cout << "Failed to create GLFW window" << std::endl;
		glfwTerminate();
		return nullptr;
	}
	glfwMakeContextCurrent(window);
	// we put in relation the window and the callbacks
	glfwSetKeyCallback(window, key_callback);
	glfwSetMouseButtonCallback(window, mouse_button_callback);
	glfwSetCursorPosCallback(window, cursor_position_callback);
	glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
	glfwSwapInterval(1);

	return window;
}

void cursor_position_callback(GLFWwindow* window, double xpos, double ypos)
{
	if (archball)
	{
		cur_mx = xpos;
		cur_my = ypos;
	}
}

/**
 * Get a normalized vector from the center of the virtual ball O to a
 * point P on the virtual ball surface, such that P is aligned on
 * screen's (X,Y) coordinates.  If (X,Y) is too far away from the
 * sphere, return the nearest point on the virtual ball surface.
 */
glm::vec3 get_arcball_vector(int x, int y)
{
	glm::vec3 P = glm::vec3(1.0 * x / SCR_WIDTH * 2 - 1.0,
		1.0 * y / SCR_HEIGHT * 2 - 1.0,
		0);
	P.y = -P.y;
	float OP_squared = P.x * P.x + P.y * P.y;
	if (OP_squared <= 1 * 1)
		P.z = sqrt(1 * 1 - OP_squared); // Pythagoras
	else
		P = glm::normalize(P); // nearest point
	return P;
}

void imGuiSetup(GLFWwindow* window)
{
	// ImGui SETUP
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGui::StyleColorsDark();
	ImGui_ImplGlfw_InitForOpenGL(window, true);
	ImGui_ImplOpenGL3_Init("#version 410");
}
