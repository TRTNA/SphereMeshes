#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <iostream>
#include <vector>
#include <memory>

#include <spheremeshes/spheremesh.h>
#include <spheremeshes/edge.h>
#include <spheremeshes/triangle.h>
#include <utils/renderablepointcloud.h>

#include <utils/shader.h>
#include <utils/model.h>
#include <utils/pointcloud.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/string_cast.hpp>

using std::cout;
using std::endl;
using std::vector;

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow *window);
// callback functions for keyboard and mouse events
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mode);
void mouse_callback(GLFWwindow* window, double xpos, double ypos);
// if one of the WASD keys is pressed, we call the corresponding method of the Camera class
void apply_key_commands();


// settings
const unsigned int SCR_WIDTH = 1200;
const unsigned int SCR_HEIGHT = 800;


// we initialize an array of booleans for each keyboard key
bool keys[1024];

// parameters for time calculation (for animations)
GLfloat deltaTime = 0.0f;
GLfloat lastFrame = 0.0f;

const float defaultRotationSpeed = 1.0f;
const glm::vec3 defaultViewPos = glm::vec3(0.0f, 0.0f, 3.0f);

glm::mat4 modelMatrix = glm::mat4(1.0f);
glm::mat4 viewMatrix = glm::mat4(1.0f);
glm::mat4 projectionMatrix = glm::mat4(1.0f);


glm::vec3 viewPos = defaultViewPos;

bool useNormalColouring = false;

int main()
{
    // glfw: initialize and configure
    // ------------------------------
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    // glfw window creation
    // --------------------
    GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "SphereMesh_PointCloudViewer", NULL, NULL);
    if (window == NULL)
    {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    // we put in relation the window and the callbacks
    glfwSetKeyCallback(window, key_callback);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    // glad: load all OpenGL function pointers
    // ---------------------------------------
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    // ImGui SETUP
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO &io = ImGui::GetIO();
    (void)io;
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 410");

    // build and compile our shader program
    Shader shader("assets/shaders/default.vert", "assets/shaders/default.frag");
    glClearColor(0.3f, 0.3f, 0.6f, 1.0f);  
    
    // Projection matrix: FOV angle, aspect ratio, near and far planes
    projectionMatrix = glm::perspective(45.0f, (float)SCR_WIDTH/(float)SCR_HEIGHT, 0.1f, 1000.0f);
    // View matrix (=camera): position, view direction, camera "up" vector
    viewMatrix = glm::lookAt(viewPos, glm::vec3(0.0f, 0.0f, -1.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    glm::mat3 normalMatrix;

    vector<Sphere> spheres{Sphere(glm::vec3(-0.5f, 0.0f, 0.0f), 0.3f), Sphere(glm::vec3(0.5f, 0.0f, 0.0f), 0.5f), Sphere(glm::vec3(0.5f, 1.0f, 0.0f), 0.3f), Sphere(glm::vec3(-0.5f, 1.0f, 0.0f), 0.1f)};
    vector<Edge> edges{Edge(0, 1), Edge(1, 2), Edge(0, 3)};
    SphereMesh sm = SphereMesh(spheres, edges, vector<Triangle>{});
    PointCloud pc = PointCloud();
    int pointsNumber = 10000;

    pc.repopulate(pointsNumber, sm);
    std::shared_ptr<PointCloud> pc_ptr = std::make_shared<PointCloud>(pc);
    RenderablePointCloud rpc = RenderablePointCloud(pc_ptr);


    shader.Use();
    glPointSize(5.0f);
    glUniformMatrix4fv(glGetUniformLocation(shader.Program, "projectionMatrix"), 1, GL_FALSE, glm::value_ptr(projectionMatrix));
    /* GLuint normalColouringSubroutineIndex = glGetSubroutineIndex(shader.Program, GL_FRAGMENT_SHADER, "normalColouring");
    GLuint diffuseColouringSubroutineIndex = glGetSubroutineIndex(shader.Program, GL_FRAGMENT_SHADER, "diffuseColouring");
    GLint activeSubroutineCount;
    glGetProgramStageiv(shader.Program, GL_FRAGMENT_SHADER, GL_ACTIVE_SUBROUTINE_UNIFORM_LOCATIONS, &activeSubroutineCount);

     */
    // render loop
    // -----------

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
        viewMatrix = glm::lookAt(viewPos, glm::vec3(0.0f, 0.0f, -1.0f), glm::vec3(0.0f, 1.0f, 0.0f));

        // we "clear" the frame and z buffer
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // render your GUI
        ImGui::Begin("Controls");
        bool pointsNumberChanged = ImGui::SliderInt("Points number", &pointsNumber, 100, 1000000);
        if (pointsNumberChanged) {
            pc_ptr->repopulate(pointsNumber, sm);
            rpc.updateBuffers();
        }


        ImGui::End();

        glUniformMatrix4fv(glGetUniformLocation(shader.Program, "modelMatrix"), 1, GL_FALSE, glm::value_ptr(modelMatrix));
        glUniformMatrix4fv(glGetUniformLocation(shader.Program, "viewMatrix"), 1, GL_FALSE, glm::value_ptr(viewMatrix));
        normalMatrix = glm::transpose(glm::inverse(glm::mat3(viewMatrix)));
        glUniformMatrix3fv(glGetUniformLocation(shader.Program, "normalMatrix"), 1, GL_FALSE, glm::value_ptr(normalMatrix));
/*         if (useNormalColouring) {
            glUniformSubroutinesuiv(GL_FRAGMENT_SHADER, activeSubroutineCount, &normalColouringSubroutineIndex);
        } else {
            glUniformSubroutinesuiv(GL_FRAGMENT_SHADER, activeSubroutineCount, &diffuseColouringSubroutineIndex);
        } */
        rpc.Draw(shader);



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
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    // make sure the viewport matches the new window dimensions; note that width and 
    // height will be significantly larger than specified on retina displays.
    glViewport(0, 0, width, height);
}

//////////////////////////////////////////
// callback for keyboard events
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mode)
{
    // if ESC is pressed, we close the application
    if(key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(window, GL_TRUE);
    
    // we keep trace of the pressed keys
    // with this method, we can manage 2 keys pressed at the same time:
    // many I/O managers often consider only 1 key pressed at the time (the first pressed, until it is released)
    // using a boolean array, we can then check and manage all the keys pressed at the same time
    if(action == GLFW_PRESS)
        keys[key] = true;
    else if(action == GLFW_RELEASE)
        keys[key] = false;
}

//////////////////////////////////////////
// If one of the WASD keys is pressed, the camera is moved accordingly (the code is in utils/camera.h)
void apply_key_commands()
{
    if(keys[GLFW_KEY_R]){
        viewPos = defaultViewPos;
        modelMatrix = glm::mat4(1.0f);
        return;
    }
    if(keys[GLFW_KEY_A]) {
        modelMatrix = glm::rotate(modelMatrix, -defaultRotationSpeed*deltaTime, glm::vec3(0.0f, 1.0f, 0.0f));
        return;
    }
    if(keys[GLFW_KEY_S]) {
        modelMatrix = glm::rotate(modelMatrix, -defaultRotationSpeed*deltaTime, glm::vec3(1.0f, 0.0f, 0.0f));
        return;
    }
    if(keys[GLFW_KEY_D])
    {        
        modelMatrix = glm::rotate(modelMatrix, defaultRotationSpeed*deltaTime, glm::vec3(0.0f, 1.0f, 0.0f));
        return;
    }
    if(keys[GLFW_KEY_W]) {
        modelMatrix = glm::rotate(modelMatrix, defaultRotationSpeed*deltaTime, glm::vec3(1.0f, 0.0f, 0.0f));
        return;
    }

    if(keys[GLFW_KEY_Z] && keys[GLFW_KEY_I]) {
        viewPos.z = glm::max(0.0f, viewPos.z - 0.5f*deltaTime);
        return;
    }

    if(keys[GLFW_KEY_Z] && keys[GLFW_KEY_O]) {
        viewPos.z += 0.5f*deltaTime;
        return;
    }

    if(keys[GLFW_KEY_N]) {
        useNormalColouring = true;
        return;
    }
    if(keys[GLFW_KEY_C]) {
        useNormalColouring = false;
        return;
    }


}


