#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <iostream>
#include <vector>
#include <memory>
#include <string>

#include <spheremeshes/spheremesh.h>
#include <spheremeshes/capsuloid.h>
#include <spheremeshes/spheretriangle.h>

#include <utils/ray.h>
#include <utils/common.h>
#include <utils/pointcloud.h>

#include <rendering/renderablepointcloud.h>
#include <rendering/shader.h>
#include <rendering/model.h>

#include <cloth/cloth.h>
#include <rendering/renderablecloth.h>

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

GLFWwindow *glfwSetup(std::string windowTitle, float width, float height);
void imGuiSetup(GLFWwindow *window);
void framebuffer_size_callback(GLFWwindow *window, int width, int height);
void processInput(GLFWwindow *window);
// callback functions for keyboard and mouse events
void key_callback(GLFWwindow *window, int key, int scancode, int action, int mode);
void mouse_callback(GLFWwindow *window, double xpos, double ypos);
void mouse_button_callback(GLFWwindow *window, int button, int action, int mods);
void cursor_position_callback(GLFWwindow *window, double xpos, double ypos);
glm::vec3 get_arcball_vector(int x, int y);
// if one of the WASD keys is pressed, we call the corresponding method of the Camera class
void apply_key_commands();

// settings
const unsigned int SCR_WIDTH = 1280;
const unsigned int SCR_HEIGHT = 720;

// we initialize an array of booleans for each keyboard key
bool keys[1024];

// Mouse events
double last_mx = 0.0, last_my = 0.0, cur_mx = 0.0, cur_my = 0.0;
bool archball = false;

// parameters for time calculation (for animations)
GLfloat deltaTime = 0.0f;
GLfloat lastFrame = 0.0f;

const float defaultRotationSpeed = 1.0f;

float lightPos[3] = {0.0f, 3.0f, 3.0f};
float ambientColor[3] = {0.1, 0.0, 0.0};
float diffuseColor[3] = {0.5, 0.0, 0.0};
float boundingSphereColor[3] = {0.0, 0.0, 0.5};

float specColor[3] = {1.0, 1.0, 1.0};
float shininess = 16.0;

glm::mat4 modelMatrix = glm::mat4(1.0f);
glm::mat3 normalMatrix;
glm::mat4 viewMatrix = glm::mat4(1.0f);
glm::mat4 projectionMatrix = glm::mat4(1.0f);

const glm::vec3 defaultViewPos = glm::vec3(0.0f, 0.0f, 3.0f);
glm::vec3 viewPos = defaultViewPos;

GLuint subroutinesIdxs[4];
int activeSubroutineIdx = 0;

bool backFaceCulling = false;
bool renderBoundingSphere = false;

SphereMesh sm;
int pointsNumber = 100;
float pointsSize = 5.0f;
int boundingSpherePointsNumber = pointsNumber;

int main(int argc, char *argv[])
{
    // Context creation
    GLFWwindow *window = glfwSetup("SphereMeshes", SCR_WIDTH, SCR_HEIGHT);
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

    // Sphere mesh loading
    std::string smToLoad = argc > 1 ? argv[1] : "default.sm";
    string readErrorMsg;
    bool read = readFromFile("assets/spheremeshes/" + smToLoad, sm, readErrorMsg);
    if (!read)
    {
        std::cerr << readErrorMsg << std::endl;
        return 1;
    }
    cout << "Sphere mesh:\n"
         << sm << endl;

    // Sphere mesh rendering setup
    PointCloud pc = PointCloud();
    pc.repopulate(pointsNumber, sm);
    std::shared_ptr<PointCloud> pc_ptr = std::make_shared<PointCloud>(pc);
    RenderablePointCloud rpc = RenderablePointCloud(pc_ptr);

    // Sphere mesh's bounding sphere rendering setup
    SphereMesh boundingSphereFakeSm;
    PointCloud boundingSphereFakePc = PointCloud();
    boundingSphereFakePc.repopulate(boundingSpherePointsNumber, sm);
    std::shared_ptr<PointCloud> boundingSphereFakePc_ptr = std::make_shared<PointCloud>(boundingSphereFakePc);
    RenderablePointCloud boundingSphereFakeRpc = RenderablePointCloud(boundingSphereFakePc_ptr);

    // Setting up viewMatrix and projection matrix to take into account sphere mesh dimension
    float fovY = 45.0f;
    float aspect = (float)SCR_WIDTH / (float)SCR_HEIGHT;
    projectionMatrix = glm::perspective(fovY, aspect, 0.1f, 1000.0f);
    float oppositeFovY = 90.0f - 45.0f;
    float dist = sm.boundingSphere.radius * glm::tan(oppositeFovY);
    viewPos = sm.boundingSphere.center;
    viewPos.z += aspect * dist;
    viewMatrix = glm::lookAt(viewPos, glm::vec3(0.0f, 0.0f, -1.0f), glm::vec3(0.0f, 1.0f, 0.0f));

    // Shader setup
    Shader shader("assets/shaders/default.vert", "assets/shaders/default.frag");
    shader.Use();
    glPointSize(pointsSize);
    glUniformMatrix4fv(glGetUniformLocation(shader.Program, "projectionMatrix"), 1, GL_FALSE, glm::value_ptr(projectionMatrix));
    glUniform3fv(glGetUniformLocation(shader.Program, "vLightPos"), 1, glm::value_ptr(glm::vec3(viewMatrix * glm::vec4(lightPos[0], lightPos[1], lightPos[2], 1.0))));
    glUniform1i(glGetUniformLocation(shader.Program, "backFaceCulling"), backFaceCulling);
    subroutinesIdxs[0] = glGetSubroutineIndex(shader.Program, GL_FRAGMENT_SHADER, "shadingColoring");
    subroutinesIdxs[1] = glGetSubroutineIndex(shader.Program, GL_FRAGMENT_SHADER, "diffuseColoring");
    subroutinesIdxs[2] = glGetSubroutineIndex(shader.Program, GL_FRAGMENT_SHADER, "normalColoring");
    subroutinesIdxs[3] = glGetSubroutineIndex(shader.Program, GL_FRAGMENT_SHADER, "flatColoring");
    GLint activeSubroutineCount;
    glGetProgramStageiv(shader.Program, GL_FRAGMENT_SHADER, GL_ACTIVE_SUBROUTINE_UNIFORM_LOCATIONS, &activeSubroutineCount);

    glClearColor(0.3f, 0.3f, 0.6f, 1.0f);

    RenderableCloth cloth(4U, 1.0f);
    //cout << cloth.toString() << endl;
    // render loop
    // -----------
    cloth.addForce(glm::vec3(0.0f, -9.8f, 0.0f));

    while (!glfwWindowShouldClose(window))
    {
        // we determine the time passed from the beginning
        // and we calculate time difference between current frame rendering and the previous one
        GLfloat currentFrame = glfwGetTime();
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;

        // Check is an I/O event is happening
        glfwPollEvents();

        apply_key_commands();


        // we "clear" the frame and z buffer
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glEnable(GL_DEPTH_TEST);

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // render your GUI
        ImGui::Begin("Controls");
        ImGui::Text("Sphere mesh:");
        if (ImGui::Checkbox("Render bounding sphere", &renderBoundingSphere))
        {
            boundingSphereFakeSm.singletons.clear();
            sm.updateBoundingSphere();
            boundingSphereFakeSm.addSphere(sm.boundingSphere);
            boundingSphereFakeSm.addSingleton(0U);
            boundingSphereFakeSm.updateBoundingSphere();
            boundingSphereFakePc_ptr->clear();
            boundingSphereFakePc_ptr->repopulate(boundingSpherePointsNumber, boundingSphereFakeSm);
            boundingSphereFakeRpc.updateBuffers();
        }
        if (renderBoundingSphere)
        {
            if (ImGui::SliderInt("Bounding sphere points number", &boundingSpherePointsNumber, 100, 1000000))
            {
                boundingSphereFakePc_ptr->repopulate(boundingSpherePointsNumber, boundingSphereFakeSm);
                boundingSphereFakeRpc.updateBuffers();
            }
        }
        ImGui::Text("Pointcloud:");
        if (ImGui::SliderInt("Points number", &pointsNumber, 100, 1000000))
        {
            pc_ptr->repopulate(pointsNumber, sm);
            rpc.updateBuffers();
        }
        if (ImGui::SliderFloat("Points size", &pointsSize, 0.1f, 10.0f))
        {
            glPointSize(pointsSize);
        }
        ImGui::Text("Shading:");
        ImGui::RadioButton("Phong", &activeSubroutineIdx, 0);
        ImGui::SameLine();
        ImGui::RadioButton("Dimensionality", &activeSubroutineIdx, 1);
        ImGui::SameLine();
        ImGui::RadioButton("Normal", &activeSubroutineIdx, 2);
        ImGui::Text("Lighting:");
        if (ImGui::SliderFloat3("Light position", lightPos, -100.0f, 100.0f))
        {
            glUniform3fv(glGetUniformLocation(shader.Program, "vLightPos"), 1, glm::value_ptr(glm::vec3(viewMatrix * glm::vec4(lightPos[0], lightPos[1], lightPos[2], 1.0))));
        }
        if (ImGui::SliderFloat3("Diffuse color", diffuseColor, 0.0f, 1.0f))
        {
            glUniform3fv(glGetUniformLocation(shader.Program, "diffuseColor"), 1, diffuseColor);
        }
        if (ImGui::SliderFloat3("Specular color", specColor, 0.0f, 1.0f))
        {
            glUniform3fv(glGetUniformLocation(shader.Program, "specColor"), 1, specColor);
        }
        if (ImGui::SliderFloat3("Ambient color", ambientColor, 0.0f, 1.0f))
        {
            glUniform3fv(glGetUniformLocation(shader.Program, "ambientColor"), 1, ambientColor);
        }
        if (ImGui::SliderFloat("Shininess", &shininess, 0.0f, 100.0f))
        {
            glUniform1f(glGetUniformLocation(shader.Program, "shininess"), shininess);
        }
        ImGui::Text("Optimization:");
        if (ImGui::Checkbox("Backface culling ", &backFaceCulling))
        {
            glUniform1i(glGetUniformLocation(shader.Program, "backFaceCulling"), backFaceCulling);
        }

        ImGui::End();

        //phys simulation
        cloth.timeStep();

        viewMatrix = glm::lookAt(viewPos, glm::vec3(0.0f, 0.0f, -1.0f), glm::vec3(0.0f, 1.0f, 0.0f));
        glUniformMatrix4fv(glGetUniformLocation(shader.Program, "viewMatrix"), 1, GL_FALSE, glm::value_ptr(viewMatrix));

        // Archball rotation of sphere mesh
        if (cur_mx != last_mx || cur_my != last_my)
        {
            glm::vec3 va = get_arcball_vector(last_mx, last_my);
            glm::vec3 vb = get_arcball_vector(cur_mx, cur_my);
            float angle = glm::acos(glm::min(1.0f, glm::dot(va, vb)));
            glm::vec3 axis_in_camera_coord = glm::cross(va, vb);
            glm::mat3 camera2object = glm::inverse(glm::mat3(viewMatrix) * glm::mat3(modelMatrix));
            glm::vec3 axis_in_object_coord = camera2object * axis_in_camera_coord;
            modelMatrix = glm::rotate(modelMatrix, glm::degrees(angle * deltaTime), axis_in_object_coord);
            last_mx = cur_mx;
            last_my = cur_my;
        }

        shader.Use();
        glUniformMatrix4fv(glGetUniformLocation(shader.Program, "modelMatrix"), 1, GL_FALSE, glm::value_ptr(modelMatrix));

        normalMatrix = glm::transpose(glm::inverse(glm::mat3(viewMatrix) * glm::mat3(modelMatrix)));
        glUniformMatrix3fv(glGetUniformLocation(shader.Program, "normalMatrix"), 1, GL_FALSE, glm::value_ptr(normalMatrix));

        if (renderBoundingSphere)
        {
            glUniform3fv(glGetUniformLocation(shader.Program, "diffuseColor"), 1, boundingSphereColor);
            glUniformSubroutinesuiv(GL_FRAGMENT_SHADER, activeSubroutineCount, &subroutinesIdxs[3]);
            boundingSphereFakeRpc.draw();
        }
        glUniformSubroutinesuiv(GL_FRAGMENT_SHADER, activeSubroutineCount, &subroutinesIdxs[activeSubroutineIdx]);

        glUniform3fv(glGetUniformLocation(shader.Program, "diffuseColor"), 1, diffuseColor);
        rpc.draw();

        glUniformSubroutinesuiv(GL_FRAGMENT_SHADER, activeSubroutineCount, &subroutinesIdxs[2]);
        cloth.draw();

        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        // glfw: swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
        // -------------------------------------------------------------------------------
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // optional: de-allocate all resources once they've outlived their purpose:
    // ------------------------------------------------------------------------
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    shader.Delete();

    // glfw: terminate, clearing all previously allocated GLFW resources.
    // ------------------------------------------------------------------
    glfwTerminate();
    return 0;
}

// glfw: whenever the window size changed (by OS or user resize) this callback function executes
// ---------------------------------------------------------------------------------------------
void framebuffer_size_callback(GLFWwindow *window, int width, int height)
{
    // make sure the viewport matches the new window dimensions; note that width and
    // height will be significantly larger than specified on retina displays.
    glViewport(0, 0, width, height);
}

//////////////////////////////////////////
// callback for keyboard events
void key_callback(GLFWwindow *window, int key, int scancode, int action, int mode)
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

//////////////////////////////////////////
// If one of the WASD keys is pressed, the camera is moved accordingly (the code is in utils/camera.h)
void apply_key_commands()
{
    if (keys[GLFW_KEY_R])
    {
        viewPos = defaultViewPos;
        modelMatrix = glm::mat4(1.0f);
        return;
    }
    if (keys[GLFW_KEY_A])
    {
        modelMatrix = glm::rotate(modelMatrix, -defaultRotationSpeed * deltaTime, glm::vec3(0.0f, 1.0f, 0.0f));
        return;
    }
    if (keys[GLFW_KEY_S])
    {
        modelMatrix = glm::rotate(modelMatrix, -defaultRotationSpeed * deltaTime, glm::vec3(1.0f, 0.0f, 0.0f));
        return;
    }
    if (keys[GLFW_KEY_D])
    {
        modelMatrix = glm::rotate(modelMatrix, defaultRotationSpeed * deltaTime, glm::vec3(0.0f, 1.0f, 0.0f));
        return;
    }
    if (keys[GLFW_KEY_W])
    {
        modelMatrix = glm::rotate(modelMatrix, defaultRotationSpeed * deltaTime, glm::vec3(1.0f, 0.0f, 0.0f));
        return;
    }

    if (keys[GLFW_KEY_Z] && keys[GLFW_KEY_I])
    {
        viewPos.z = glm::max(0.0f, viewPos.z - 0.5f * deltaTime);
        return;
    }

    if (keys[GLFW_KEY_Z] && keys[GLFW_KEY_O])
    {
        viewPos.z += 0.5f * deltaTime;
        return;
    }
}

void mouse_button_callback(GLFWwindow *window, int button, int action, int mods)
{
    static double mouse_x = 0.0;
    static double mouse_y = 0.0;
    glfwGetCursorPos(window, &mouse_x, &mouse_y);
    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS && keys[GLFW_KEY_LEFT_ALT])
    {
        archball = true;
        last_mx = cur_mx = mouse_x;
        last_my = cur_my = mouse_y;
    }
    else if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_RELEASE || !keys[GLFW_KEY_LEFT_ALT])
    {
        archball = false;
    }

    if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS)
    {
        vec3 rayDir = screenToWorldDir(glm::vec2(mouse_x, mouse_y), SCR_WIDTH, SCR_HEIGHT, viewMatrix, projectionMatrix);
        Ray r = Ray(viewPos, rayDir);
        Point null;
        if (intersects(r, sm.boundingSphere, null))
        {
            printf("Interseca!\n");
        }
    }
}

GLFWwindow *glfwSetup(std::string windowTitle, float width, float height)
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
    GLFWwindow *window = glfwCreateWindow(width, height, windowTitle.c_str(), NULL, NULL);
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

    return window;
}

void cursor_position_callback(GLFWwindow *window, double xpos, double ypos)
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

void imGuiSetup(GLFWwindow *window)
{
    // ImGui SETUP
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 410");
}
