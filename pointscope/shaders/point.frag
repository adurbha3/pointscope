#version 330

in vec3 v_color;
out vec4 f_color;

void main() {
    // Render each point sprite as a soft circle instead of a hard square.
    vec2 c = gl_PointCoord - vec2(0.5);
    float d2 = dot(c, c);
    if (d2 > 0.25) {
        discard;
    }
    float edge = smoothstep(0.25, 0.18, d2);
    f_color = vec4(v_color, edge);
}
