#version 330

uniform vec3 line_color;
out vec4 f_color;

void main() {
    f_color = vec4(line_color, 1.0);
}
