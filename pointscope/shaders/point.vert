#version 330

uniform mat4 mvp;
uniform mat4 view;
uniform int color_mode;      // 0 = color by height, 1 = color by intensity
uniform float height_min;
uniform float height_max;
uniform float point_size_base;

in vec3 in_position;
in float in_intensity;

out vec3 v_color;

// Cheap 5-stop approximation of a perceptually-ordered colormap
// (blue -> teal -> green -> yellow -> red), good enough for a debug view.
vec3 ramp_color(float t) {
    t = clamp(t, 0.0, 1.0);
    vec3 c0 = vec3(0.19, 0.07, 0.23);
    vec3 c1 = vec3(0.13, 0.55, 0.80);
    vec3 c2 = vec3(0.35, 0.80, 0.40);
    vec3 c3 = vec3(0.97, 0.80, 0.20);
    vec3 c4 = vec3(0.90, 0.20, 0.20);
    float seg = t * 4.0;
    if (seg < 1.0) return mix(c0, c1, seg);
    if (seg < 2.0) return mix(c1, c2, seg - 1.0);
    if (seg < 3.0) return mix(c2, c3, seg - 2.0);
    return mix(c3, c4, min(seg - 3.0, 1.0));
}

void main() {
    gl_Position = mvp * vec4(in_position, 1.0);

    float t = color_mode == 1
        ? in_intensity
        : clamp((in_position.z - height_min) / max(height_max - height_min, 0.001), 0.0, 1.0);
    v_color = ramp_color(t);

    // Attenuate point size with view-space distance so near points read as
    // dense surfaces and far points don't overdraw into a solid blob --
    // the same trick real LiDAR viewers use.
    vec4 view_pos = view * vec4(in_position, 1.0);
    float dist = max(-view_pos.z, 0.01);
    gl_PointSize = clamp(point_size_base * (20.0 / dist), 1.0, 14.0);
}
