#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <glad/glad.h>
#include <GLFW/glfw3.h>


#include <iostream>
#include <vector>

#include <utils/shader.h>
#include <spheremeshes/spheremeshes.h>
#include <utils/model.h>
#include <utils/pointcloud.h>
#include <utils/random.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/type_ptr.hpp>


using std::cout;
using std::endl;
using std::vector;

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow *window);

// settings
const unsigned int SCR_WIDTH = 1600;
const unsigned int SCR_HEIGHT = 1200;

typedef GlRendSphereMesh glSphereMesh;

int main()
{
    // glfw: initialize and configure
    // ------------------------------
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    // glfw window creation
    // --------------------
    GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "LearnOpenGL", NULL, NULL);
    if (window == NULL)
    {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
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
    Shader shader("assets/shaders/capsule.vert", "assets/shaders/pointsplat.frag");
    shader.Use();

    
    // Projection matrix: FOV angle, aspect ratio, near and far planes
    glm::mat4 projectionMatrix = glm::perspective(45.0f, (float)SCR_WIDTH/(float)SCR_HEIGHT, 0.1f, 10000.0f);
    // View matrix (=camera): position, view direction, camera "up" vector
    glm::mat4 viewMatrix = glm::lookAt(glm::vec3(0.0f, 0.0f, 7.0f), glm::vec3(0.0f, 0.0f, -7.0f), glm::vec3(0.0f, 1.0f, 0.0f));

    glSphereMesh sm(vector<Sphere>{Sphere(glm::vec3(-0.5f, 0.0f, -5.0f), 0.3f), Sphere(glm::vec3(0.5f, 0.0f, -5.0f), 0.3f), Sphere(glm::vec3(-0.0f, 0.5f, -5.0f), 0.3f), Sphere(glm::vec3(-0.0f, 1.5f, -5.0f), 0.3f)}, vector<Edge>{Edge(2, 3)}, vector<Triangle>{Triangle(0, 1, 2)});
    Model sphereModel("assets/models/sphere.obj");
    glSphereMesh::sphereModel = &sphereModel;

    shader.Use();
    glUniformMatrix4fv(glGetUniformLocation(shader.Program, "viewMatrix"), 1, GL_FALSE, glm::value_ptr(viewMatrix));
    glUniformMatrix4fv(glGetUniformLocation(shader.Program, "projectionMatrix"), 1, GL_FALSE, glm::value_ptr(projectionMatrix));

    PointCloud PC;
    std::pair<float, float> xRange(-0.5f, 0.5f);
    std::pair<float, float> yRange(-0.5f, 0.5f);
    std::pair<float, float> zRange(-1.0f, 1.0f);
    for (size_t i = 0; i < 1000; i++) {
        PC.addPoint(generatePoint(xRange, yRange, zRange));
    }
    unsigned int VAO, VBO;
        glGenVertexArrays(1, &VAO);
        glGenBuffers(1, &VBO);

        glBindVertexArray(VAO);
        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        std::vector<glm::vec3> points = PC.getPoints();
        glBufferData(GL_ARRAY_BUFFER, 1000* sizeof(glm::vec3), points.data() , GL_STATIC_DRAW);

        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (GLvoid*)0);

        glUniform3fv(glGetUniformLocation(shader.Program, "capsA"), 1, glm::value_ptr(glm::vec3(-0.5f, 0.0f, 5.0f)));
        glUniform3fv(glGetUniformLocation(shader.Program, "capsB"), 1, glm::value_ptr(glm::vec3(0.5f, 0.0f, 5.0f)));
        glUniform1f(glGetUniformLocation(shader.Program, "radius"), 0.5f);


    // render loop
    // -----------
    while (!glfwWindowShouldClose(window))
    {
        // input
        // -----
        processInput(window);

        // render
        // ------
        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // render your GUI
        ImGui::Begin("Demo window");
        ImGui::Button("Hello!");
        ImGui::End();

        glBindVertexArray(VAO);
        glDrawArrays(GL_POINTS, 0, 1000);

        
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

// process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
// ---------------------------------------------------------------------------------------------------------
void processInput(GLFWwindow *window)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
}

// glfw: whenever the window size changed (by OS or user resize) this callback function executes
// ---------------------------------------------------------------------------------------------
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    // make sure the viewport matches the new window dimensions; note that width and 
    // height will be significantly larger than specified on retina displays.
    glViewport(0, 0, width, height);
}