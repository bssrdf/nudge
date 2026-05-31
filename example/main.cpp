//
// Copyright (c) 2017 Rasmus Barringer
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//

#include <nudge.h>
#include <immintrin.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#ifdef __APPLE__
#include <GLUT/GLUT.h>
#include <OpenGL/gl.h>
#else
#include <GLUT/glut.h>
#include <GLUT/freeglut_ext.h>  // for glutMouseWheelFunc
#include <gl/gl.h>
#ifdef _WIN32
#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0601  // Windows 7+
#endif
#include <windows.h>
#endif
#endif

static const unsigned max_body_count = 19200;
static const unsigned max_box_count = 19200;
static const unsigned max_sphere_count = 4096;

static const nudge::Transform identity_transform = { {}, 0, { 0.0f, 0.0f, 0.0f, 1.0f } };

static nudge::Arena arena;
static nudge::BodyData bodies;
static nudge::ColliderData colliders;
static nudge::ContactData contact_data;
static nudge::ContactCache contact_cache;
static nudge::ActiveBodies active_bodies;

// Camera state — OrbitControls style (three.js inspired)
// Spherical coordinates: camera orbits around a target point
static float orbit_target[3]   = { 0.0f, 10.0f, 0.0f };  // center of scene, mid-height of skyscraper
static float orbit_theta  = 0.5f;   // horizontal angle (radians)
static float orbit_phi   = 0.6f;   // vertical angle (radians)
static float orbit_radius = 70.0f;  // distance from target — far enough to see the scene

// Damping (smooth deceleration)
static float orbit_theta_delta  = 0.0f;
static float orbit_phi_delta    = 0.0f;
static float orbit_radius_delta = 0.0f;
static float orbit_pan_x_delta  = 0.0f;
static float orbit_pan_y_delta  = 0.0f;
static float orbit_pan_z_delta  = 0.0f;

// Mouse drag state
static bool mouse_left_down  = false;   // orbit
static bool mouse_right_down = false;   // pan
static int  mouse_last_x = 0, mouse_last_y = 0;

// Camera settings (OrbitControls-style)
static const float orbit_damping_factor     = 0.08f;   // higher = less damping (three.js default ~0.05-0.1)
static const float orbit_rotate_speed       = 0.001f;  // radians per pixel
static const float orbit_zoom_speed         = 0.04f;   // radius change per scroll tick
static const float orbit_pan_speed          = 0.008f;  // pan scale per pixel
static const float orbit_min_radius         = 2.0f;
static const float orbit_max_radius         = 500.0f;
static const float orbit_min_phi            = 0.01f;   // near straight down
static const float orbit_max_phi            = 3.12f;   // near straight up (PI - 0.02)

// Key state tracked by GLUT callbacks
static bool key_state[256] = {};
static int debug_frame = 0;

// No-op — raw input replaced by GLUT mouse callbacks
static void raw_input_poll() {}

static void draw_sky();

// Forward declarations (defined below)
static inline unsigned add_box(float mass, float cx, float cy, float cz);
static inline unsigned add_sphere(float mass, float radius);
static void build_old_scene();

// =============================================================================
// KEVLA-style plank builder
// =============================================================================
// Plank dimensions (length x thickness x width)
// Ratio ~ 10 : 1 : 2 — thin, flat, like a wooden plank or K'NEX/KEVLA plank
static const float plank_length   = 15.0f;
static const float plank_thickness = 1.0f;
static const float plank_width    = 3.0f;
static const float plank_mass     = 2.0f;

// Tiny random offset to prevent perfect stacking (avoids penetration explosions)
static inline float jitter() {
    return ((float)rand() / (float)RAND_MAX) * 0.1f - 0.05f;
}

// Ground surface Y (ground box is at -20, height 10 → surface at -15)
static const float ground_y = -15.0f;

// Place a plank at a given world position with given orientation.
// orientation: 0=horizontal (length along X, stacked on Y)
//              1=flat (length along X, flat on Z — for floors/ceilings)
//              2=horizontal_z (length along Z, stacked on Y — walls facing X)
//              3=vertical (standing on end — for posts/columns)
static inline unsigned place_plank(float px, float py, float pz, unsigned orientation) {
    float sx, sy, sz;
    switch (orientation) {
        case 0: // horizontal, length along X
            sx = plank_length;   sy = plank_thickness; sz = plank_width;
            break;
        case 1: // flat (floor/ceiling), length along X
            sx = plank_length;   sy = plank_width;     sz = plank_thickness;
            break;
        case 2: // horizontal, length along Z
            sx = plank_width;    sy = plank_thickness; sz = plank_length;
            break;
        case 3: // vertical post
            sx = plank_width;    sy = plank_length;    sz = plank_thickness;
            break;
        default:
            sx = plank_length;   sy = plank_thickness; sz = plank_width;
    }
    unsigned body = add_box(plank_mass, sx, sy, sz);
    if (body) {
        bodies.transforms[body].position[0] = px + jitter();
        bodies.transforms[body].position[1] = py + jitter();
        bodies.transforms[body].position[2] = pz + jitter();
    }
    return body;
}

// Build a hollow rectangular box (walls only, no floor/ceiling).
// Walls are staggered for stability. 'gap_x'/'gap_z' leave openings.
static void build_wall_box(float cx, float cz, float half_w, float half_d, unsigned num_layers) {
    // Front wall (along X, at z = cz + half_d)
    for (unsigned layer = 0; layer < num_layers; layer++) {
        float y = ground_y + (layer + 0.5f) * plank_thickness + plank_thickness * 0.5f;
        float stagger = (layer % 2 == 0) ? 0.0f : plank_length * 0.5f;
        for (float x = -half_w; x < half_w; x += plank_length) {
            place_plank(cx + x + stagger, y, cz + half_d, 0);
        }
    }
    // Back wall (along X, at z = cz - half_d)
    for (unsigned layer = 0; layer < num_layers; layer++) {
        float y = ground_y + (layer + 0.5f) * plank_thickness + plank_thickness * 0.5f;
        float stagger = (layer % 2 == 0) ? 0.0f : plank_length * 0.5f;
        for (float x = -half_w; x < half_w; x += plank_length) {
            place_plank(cx + x + stagger, y, cz - half_d, 0);
        }
    }
    // Left wall (along Z, at x = cx - half_w)
    for (unsigned layer = 0; layer < num_layers; layer++) {
        float y = ground_y + (layer + 0.5f) * plank_thickness + plank_thickness * 0.5f;
        float stagger = (layer % 2 == 0) ? 0.0f : plank_length * 0.5f;
        for (float z = -half_d; z < half_d; z += plank_length) {
            place_plank(cx - half_w, y, cz + z + stagger, 2);
        }
    }
    // Right wall (along Z, at x = cx + half_w)
    for (unsigned layer = 0; layer < num_layers; layer++) {
        float y = ground_y + (layer + 0.5f) * plank_thickness + plank_thickness * 0.5f;
        float stagger = (layer % 2 == 0) ? 0.0f : plank_length * 0.5f;
        for (float z = -half_d; z < half_d; z += plank_length) {
            place_plank(cx + half_w, y, cz + z + stagger, 2);
        }
    }
}

// Build a floor/ceiling slab
static void build_slab(float cx, float cz, float half_w, float half_d, float y) {
    for (float x = -half_w; x < half_w; x += plank_length) {
        for (float z = -half_d; z < half_d; z += plank_width) {
            place_plank(cx + x, y, cz + z, 1);
        }
    }
}

// Build a simple house: walls + roof slabs
static void build_house(float cx, float cz, unsigned half_w_planks, unsigned half_d_planks, unsigned wall_layers) {
    float half_w = (float)half_w_planks * plank_length;
    float half_d = (float)half_d_planks * plank_length;

    // Walls
    build_wall_box(cx, cz, half_w, half_d, wall_layers);

    // Floor (slightly above ground)
    build_slab(cx, cz, half_w, half_d, ground_y + plank_thickness * 0.5f);

    // Roof (top of walls + slight overhang)
    float roof_y = ground_y + wall_layers * plank_thickness + plank_thickness;
    build_slab(cx, cz, half_w + plank_length, half_d + plank_length, roof_y);

    // Cross-bracing every few layers for stability
    for (unsigned layer = 0; layer < wall_layers; layer += 4) {
        float y = ground_y + (layer + 0.5f) * plank_thickness;
        // Diagonal-ish supports: vertical posts at corners
        place_plank(cx - half_w, y, cz - half_d, 3);
        place_plank(cx + half_w, y, cz + half_d, 3);
    }
}

// Build a skyscraper: tall walls with periodic cross-bracing and floor slabs
static void build_skyscraper(float cx, float cz, unsigned half_w_planks, unsigned half_d_planks, unsigned num_floors) {
    float half_w = (float)half_w_planks * plank_length;
    float half_d = (float)half_d_planks * plank_length;
    unsigned planks_per_floor = 3; // 3 plank-layers per floor level

    // Walls: continuous stack
    unsigned total_layers = num_floors * planks_per_floor;
    build_wall_box(cx, cz, half_w, half_d, total_layers);

    // Floor slabs at each floor level
    for (unsigned floor = 0; floor <= num_floors; floor++) {
        float y = ground_y + floor * planks_per_floor * plank_thickness;
        build_slab(cx, cz, half_w - plank_length * 0.5f, half_d - plank_length * 0.5f, y + plank_thickness);
    }

    // Corner columns for extra stability
    for (int ix = -1; ix <= 1; ix += 2) {
        for (int iz = -1; iz <= 1; iz += 2) {
            float px = cx + ix * half_w;
            float pz = cz + iz * half_d;
            // Stack vertical posts
            for (unsigned col = 0; col < total_layers / 2; col++) {
                place_plank(px, ground_y + col * plank_length, pz, 3);
            }
        }
    }
}

// Build a tower: circular-ish, staggered rings
static void build_tower(float cx, float cz, unsigned radius_planks, unsigned num_layers) {
    float radius = (float)radius_planks * plank_length;
    unsigned planks_per_ring = (unsigned)(6.283f * radius / plank_length) + 1;

    for (unsigned layer = 0; layer < num_layers; layer++) {
        float y = ground_y + (layer + 0.5f) * plank_thickness;
        float stagger_angle = (layer % 2 == 0) ? 0.0f : 3.14159f / planks_per_ring;

        for (unsigned p = 0; p < planks_per_ring; p++) {
            float angle = (float)p / (float)planks_per_ring * 6.283f + stagger_angle;
            float px = cx + cosf(angle) * radius;
            float pz = cz + sinf(angle) * radius;

            // Orient plank tangent to ring
            if (cosf(angle) > 0.7f || cosf(angle) < -0.7f) {
                place_plank(px, y, pz, 0); // along X
            } else {
                place_plank(px, y, pz, 2); // along Z
            }
        }
    }
}

// Build a bridge between two points
static void build_bridge(float x1, float z1, float x2, float z2, unsigned deck_layers, float height_above_ground) {
    float dx = x2 - x1;
    float dz = z2 - z1;
    float length = sqrtf(dx * dx + dz * dz);
    float nx = dx / length;
    float nz = dz / length;

    unsigned num_planks = (unsigned)(length / plank_length) + 1;
    float support_interval = plank_length * 6.0f; // supports every ~30 units
    unsigned num_supports = (unsigned)(length / support_interval) + 2;

    // Deck: two layers for stability
    for (unsigned layer = 0; layer < deck_layers; layer++) {
        float y = ground_y + height_above_ground + (layer + 0.5f) * plank_thickness;
        for (unsigned p = 0; p < num_planks; p++) {
            float t = (float)p / (float)num_planks;
            float px = x1 + dx * t;
            float pz = z1 + dz * t;
            // Place planks perpendicular to bridge direction
            if (fabsf(nx) > fabsf(nz)) {
                place_plank(px, y, pz, 2); // along Z
            } else {
                place_plank(px, y, pz, 0); // along X
            }
        }
    }

    // Support towers at intervals
    for (unsigned s = 0; s < num_supports; s++) {
        float t = (float)s / (float)(num_supports - 1);
        float px = x1 + dx * t;
        float pz = z1 + dz * t;

        // Build a small support column
        unsigned support_height = (unsigned)(height_above_ground / plank_thickness) + 1;
        for (unsigned h = 0; h < support_height; h++) {
            float y = ground_y + (h + 0.5f) * plank_thickness;
            place_plank(px, y, pz, 0);
            place_plank(px, y, pz, 2);
        }
    }
}

// Build a small wall segment (for fences, barriers, interior walls)
static void build_wall_segment(float sx, float sz, float ex, float ez, unsigned layers) {
    float dx = ex - sx;
    float dz = ez - sz;
    float length = sqrtf(dx * dx + dz * dz);
    float nx = dx / length;
    float nz = dz / length;
    unsigned num_planks = (unsigned)(length / plank_length) + 1;

    for (unsigned layer = 0; layer < layers; layer++) {
        float y = ground_y + (layer + 0.5f) * plank_thickness;
        float stagger = (layer % 2 == 0) ? 0.0f : plank_length * 0.5f;
        for (unsigned p = 0; p < num_planks; p++) {
            float t = (float)(p + 0.5f) / (float)num_planks;
            float px = sx + dx * t;
            float pz = sz + dz * t;
            if (fabsf(nx) > fabsf(nz)) {
                place_plank(px, y, pz, 2);
            } else {
                place_plank(px, y, pz, 0);
            }
        }
    }
}

// Build a pyramid-like stepped structure
static void build_pyramid(float cx, float cz, unsigned base_planks, unsigned total_layers) {
    for (unsigned layer = 0; layer < total_layers; layer++) {
        float y = ground_y + (layer + 0.5f) * plank_thickness;
        // Shrink the footprint each 4 layers
        unsigned shrink = layer / 4;
        unsigned current_size = base_planks - shrink;
        if (current_size < 2) current_size = 2;
        float half_w = (float)current_size * plank_length * 0.5f;
        float half_d = half_w;

        // Build a square ring at this layer
        float stagger = (layer % 2 == 0) ? 0.0f : plank_length * 0.5f;
        // Front/back
        for (float x = -half_w; x < half_w; x += plank_length) {
            place_plank(cx + x + stagger, y, cz + half_d, 0);
            place_plank(cx + x + stagger, y, cz - half_d, 0);
        }
        // Left/right
        for (float z = -half_d; z < half_d; z += plank_length) {
            place_plank(cx - half_w, y, cz + z + stagger, 2);
            place_plank(cx + half_w, y, cz + z + stagger, 2);
        }
    }
}

// Build the old random scene (8192 boxes + 1024 spheres)
static void build_old_scene() {
    printf("[scene] building old random scene...\n");

    for (unsigned i = 0; i < 8192; ++i) {
        float sx = (float)rand() / (float)RAND_MAX + 0.5f;
        float sy = (float)rand() / (float)RAND_MAX + 0.5f;
        float sz = (float)rand() / (float)RAND_MAX + 0.5f;

        unsigned body = add_box(8.0f * sx * sy * sz, sx, sy, sz);
        if (body) {
            bodies.transforms[body].position[0] = (float)rand() / (float)RAND_MAX * 10.0f - 5.0f;
            bodies.transforms[body].position[1] = (float)rand() / (float)RAND_MAX * 300.0f;
            bodies.transforms[body].position[2] = (float)rand() / (float)RAND_MAX * 10.0f - 5.0f;
        }
    }

    for (unsigned i = 0; i < 1024; ++i) {
        float s = (float)rand() / (float)RAND_MAX + 0.5f;

        unsigned body = add_sphere(4.18879f * s * s * s, s);
        if (body) {
            bodies.transforms[body].position[0] = (float)rand() / (float)RAND_MAX * 10.0f - 5.0f;
            bodies.transforms[body].position[1] = (float)rand() / (float)RAND_MAX * 300.0f;
            bodies.transforms[body].position[2] = (float)rand() / (float)RAND_MAX * 10.0f - 5.0f;
        }
    }

    printf("[scene] done! bodies: %u, colliders: %u\n", bodies.count, colliders.boxes.count + colliders.spheres.count);
}

// Build the complete scene with architecture
static void build_scene() {
    printf("[scene] building structures...\n");

    // --- Small house ---
    printf("[scene] building small house at (-15,-10)\n");
    build_house(-15.0f, -10.0f, 2, 2, 6);

    // --- Medium house ---
    printf("[scene] building medium house at (15,-10)\n");
    build_house(15.0f, -10.0f, 3, 2, 8);

    // --- Skyscraper (small) ---
    printf("[scene] building skyscraper at (0,0)\n");
    build_skyscraper(0.0f, 0.0f, 2, 2, 5);

    // --- Tower ---
    printf("[scene] building tower at (20,15)\n");
    build_tower(20.0f, 15.0f, 2, 15);

    // --- Bridge ---
    printf("[scene] building bridge\n");
    build_bridge(-20.0f, 5.0f, 20.0f, 5.0f, 2, 4.0f);

    // --- Pyramid ---
    printf("[scene] building pyramid at (-20,20)\n");
    build_pyramid(-20.0f, 20.0f, 4, 12);

    // --- Spheres for impact ---
    printf("[scene] adding spheres\n");
    for (unsigned i = 0; i < 16; ++i) {
        float s = (float)rand() / (float)RAND_MAX * 0.5f + 0.5f;
        unsigned body = add_sphere(4.18879f * s * s * s, s);
        if (body) {
            bodies.transforms[body].position[0] = (float)rand() / (float)RAND_MAX * 50.0f - 25.0f;
            bodies.transforms[body].position[1] = ground_y + 15.0f + (float)rand() / (float)RAND_MAX * 20.0f;
            bodies.transforms[body].position[2] = (float)rand() / (float)RAND_MAX * 40.0f - 10.0f;
        }
    }

    printf("[scene] done! bodies: %u, box_colliders: %u, sphere_colliders: %u\n",
           bodies.count, colliders.boxes.count, colliders.spheres.count);
}

static inline void quaternion_concat(float r[4], const float a[4], const float b[4]) {
	r[0] = b[0]*a[3] + a[0]*b[3] + a[1]*b[2] - a[2]*b[1];
	r[1] = b[1]*a[3] + a[1]*b[3] + a[2]*b[0] - a[0]*b[2];
	r[2] = b[2]*a[3] + a[2]*b[3] + a[0]*b[1] - a[1]*b[0];
	r[3] = a[3]*b[3] - a[0]*b[0] - a[1]*b[1] - a[2]*b[2];
}

static inline void quaternion_transform(float r[3], const float a[4], const float b[3]) {
	float t[3];
	t[0] = a[1]*b[2] - a[2]*b[1];
	t[1] = a[2]*b[0] - a[0]*b[2];
	t[2] = a[0]*b[1] - a[1]*b[0];
	
	t[0] += t[0];
	t[1] += t[1];
	t[2] += t[2];
	
	r[0] = b[0] + a[3]*t[0] + a[1]*t[2] - a[2]*t[1];
	r[1] = b[1] + a[3]*t[1] + a[2]*t[0] - a[0]*t[2];
	r[2] = b[2] + a[3]*t[2] + a[0]*t[1] - a[1]*t[0];
}

static inline void matrix(float r[16], const float s[3], const float q[4], const float t[3]) {
	float kx = q[0] + q[0];
	float ky = q[1] + q[1];
	float kz = q[2] + q[2];
	
	float xx = kx*q[0];
	float yy = ky*q[1];
	float zz = kz*q[2];
	float xy = kx*q[1];
	float xz = kx*q[2];
	float yz = ky*q[2];
	float sx = kx*q[3];
	float sy = ky*q[3];
	float sz = kz*q[3];
	
	r[0] = (1.0f - yy - zz) * s[0];
	r[1] = (xy + sz) * s[0];
	r[2] = (xz - sy) * s[0];
	r[3] = 0.0f;
	
	r[4] = (xy - sz) * s[1];
	r[5] = (1.0f - xx - zz) * s[1];
	r[6] = (yz + sx) * s[1];
	r[7] = 0.0f;
	
	r[8] = (xz + sy) * s[2];
	r[9] = (yz - sx) * s[2];
	r[10] = (1.0f - xx - yy) * s[2];
	r[11] = 0.0f;
	
	r[12] = t[0];
	r[13] = t[1];
	r[14] = t[2];
	r[15] = 1.0f;
}

// Compute camera position from spherical coordinates (orbit around target).
// Returns [eye_x, eye_y, eye_z].
static inline void orbit_camera_eye(float eye[3]) {
    float sin_phi = sinf(orbit_phi);
    eye[0] = orbit_target[0] + orbit_radius * sin_phi * cosf(orbit_theta);
    eye[1] = orbit_target[1] + orbit_radius * cosf(orbit_phi);
    eye[2] = orbit_target[2] + orbit_radius * sin_phi * sinf(orbit_theta);
}

// Compute a look-at target with slight offset (for proper up vector).
static inline void orbit_camera_target(float target[3]) {
    target[0] = orbit_target[0];
    target[1] = orbit_target[1];
    target[2] = orbit_target[2];
}

// Apply damping to orbit deltas (smooth deceleration, three.js style).
static inline void orbit_apply_damping(float dt) {
    // Exponential decay: delta *= (1 - damping)^dt
    float factor = 1.0f - orbit_damping_factor;
    orbit_theta_delta *= factor;
    orbit_phi_delta   *= factor;
    orbit_radius_delta *= factor;
    orbit_pan_x_delta  *= factor;
    orbit_pan_y_delta  *= factor;
    orbit_pan_z_delta  *= factor;

    // Apply accumulated deltas
    orbit_theta  += orbit_theta_delta;
    orbit_phi    += orbit_phi_delta;
    orbit_radius += orbit_radius_delta;
    orbit_target[0] += orbit_pan_x_delta;
    orbit_target[1] += orbit_pan_y_delta;
    orbit_target[2] += orbit_pan_z_delta;

    // Clamp phi (avoid gimbal lock at poles)
    if (orbit_phi < orbit_min_phi)   orbit_phi   = orbit_min_phi;
    if (orbit_phi > orbit_max_phi)   orbit_phi   = orbit_max_phi;
    if (orbit_phi < 0.0f)            orbit_phi   = 0.0f;
    if (orbit_phi > 3.14159265f)     orbit_phi   = 3.14159265f;

    // Clamp radius
    if (orbit_radius < orbit_min_radius) orbit_radius = orbit_min_radius;
    if (orbit_radius > orbit_max_radius) orbit_radius = orbit_max_radius;

    // Zero out very small deltas to avoid floating point noise
    if (fabsf(orbit_theta_delta) < 1e-6f) orbit_theta_delta = 0.0f;
    if (fabsf(orbit_phi_delta)   < 1e-6f) orbit_phi_delta   = 0.0f;
    if (fabsf(orbit_radius_delta) < 1e-6f) orbit_radius_delta = 0.0f;
    if (fabsf(orbit_pan_x_delta)  < 1e-6f) orbit_pan_x_delta  = 0.0f;
    if (fabsf(orbit_pan_y_delta)  < 1e-6f) orbit_pan_y_delta  = 0.0f;
    if (fabsf(orbit_pan_z_delta)  < 1e-6f) orbit_pan_z_delta  = 0.0f;
}

// Handle orbit input from key state (scroll to zoom, Q/E to pan vertically)
static inline void orbit_key_input(float dt) {
    // Scroll-like zoom with Q/E keys as fallback
    if (key_state['Q'] || key_state['q']) {
        orbit_radius_delta += orbit_zoom_speed * 2.0f;  // zoom out
    }
    if (key_state['E'] || key_state['e']) {
        orbit_radius_delta -= orbit_zoom_speed * 2.0f;  // zoom in
    }
}

static inline unsigned add_box(float mass, float cx, float cy, float cz) {
	if (bodies.count == max_body_count || colliders.boxes.count == max_box_count)
		return 0;

	// unsigned body = bodies.count++;
	// unsigned collider = colliders.boxes.count++;
	unsigned body = ++bodies.count;
	unsigned collider = ++colliders.boxes.count;

	float k = mass * (1.0f / 3.0f);

	float kcx2 = k * cx * cx;
	float kcy2 = k * cy * cy;
	float kcz2 = k * cz * cz;

	nudge::BodyProperties properties = {};
	properties.mass_inverse = 1.0f / mass;
	properties.inertia_inverse[0] = 1.0f / (kcy2 + kcz2);
	properties.inertia_inverse[1] = 1.0f / (kcx2 + kcz2);
	properties.inertia_inverse[2] = 1.0f / (kcx2 + kcy2);

	memset(&bodies.momentum[body], 0, sizeof(bodies.momentum[body]));
	bodies.idle_counters[body] = 0;
	bodies.properties[body] = properties;
	bodies.transforms[body] = identity_transform;

	colliders.boxes.transforms[collider] = identity_transform;
	colliders.boxes.transforms[collider].body = body;

	colliders.boxes.data[collider].size[0] = cx;
	colliders.boxes.data[collider].size[1] = cy;
	colliders.boxes.data[collider].size[2] = cz;
	colliders.boxes.tags[collider] = collider;

	return body;
}

static inline unsigned add_sphere(float mass, float radius) {
	if (bodies.count == max_body_count || colliders.spheres.count == max_sphere_count)
		return 0;

	unsigned body = bodies.count++;
	unsigned collider = colliders.spheres.count++;

	float k = 2.5f / (mass * radius * radius);

	nudge::BodyProperties properties = {};
	properties.mass_inverse = 1.0f / mass;
	properties.inertia_inverse[0] = k;
	properties.inertia_inverse[1] = k;
	properties.inertia_inverse[2] = k;

	memset(&bodies.momentum[body], 0, sizeof(bodies.momentum[body]));
	bodies.idle_counters[body] = 0;
	bodies.properties[body] = properties;
	bodies.transforms[body] = identity_transform;

	colliders.spheres.transforms[collider] = identity_transform;
	colliders.spheres.transforms[collider].body = body;

	colliders.spheres.data[collider].radius = radius;
	colliders.spheres.tags[collider] = collider + max_box_count;

	return body;
}

static inline void shoot_sphere() {
    static bool sphere_shot = false;
    if (key_state[' '] && !sphere_shot) {
        sphere_shot = true;
        const float radius = 1.0f;
        const float mass = 4.18879f * radius * radius * radius;
        unsigned body = add_sphere(mass, radius);
        if (body) {
            // Camera eye position and look direction
            float eye[3];
            orbit_camera_eye(eye);
            float sin_phi = sinf(orbit_phi);
            float sin_theta = sinf(orbit_theta);
            float cos_theta = cosf(orbit_theta);
            float cos_phi = cosf(orbit_phi);

            // Direction from eye to target (forward vector)
            float dir_x = orbit_target[0] - eye[0];
            float dir_y = orbit_target[1] - eye[1];
            float dir_z = orbit_target[2] - eye[2];
            float len = sqrtf(dir_x*dir_x + dir_y*dir_y + dir_z*dir_z);
            if (len > 0.001f) {
                dir_x /= len;
                dir_y /= len;
                dir_z /= len;
            }

            float dist = radius + 3.0f;
            bodies.transforms[body].position[0] = eye[0] + dir_x * dist;
            bodies.transforms[body].position[1] = eye[1] + dir_y * dist;
            bodies.transforms[body].position[2] = eye[2] + dir_z * dist;

            // Shoot along view direction
            const float shoot_speed = 200.0f;
            bodies.momentum[body].velocity[0] = dir_x * shoot_speed;
            bodies.momentum[body].velocity[1] = dir_y * shoot_speed;
            bodies.momentum[body].velocity[2] = dir_z * shoot_speed;
            printf("[shoot] sphere body=%u radius=%.1f speed=%.0f\n", body, radius, shoot_speed);
        }
    }
    if (!key_state[' ']) {
        sphere_shot = false;
    }
}

// Update orbit camera: apply damping, key input, and clamp values.
static inline void orbit_camera_update(float dt) {
    debug_frame++;

    if (key_state[27]) exit(0);  // Escape

    shoot_sphere();

    // Drop a box when 'b' is pressed (one-shot per key press)
    static bool box_dropped = false;
    if ((key_state['B'] || key_state['b']) && !box_dropped) {
        box_dropped = true;
        float sx = (float)rand() / (float)RAND_MAX * 2.0f + 0.5f;
        float sy = (float)rand() / (float)RAND_MAX * 2.0f + 0.5f;
        float sz = (float)rand() / (float)RAND_MAX * 2.0f + 0.5f;
        unsigned body = add_box(8.0f * sx * sy * sz, sx, sy, sz);
        if (body) {
            bodies.transforms[body].position[0] = 0.0f;
            bodies.transforms[body].position[1] = 50.0f;
            bodies.transforms[body].position[2] = 0.0f;
            printf("[box] dropped body=%u size=(%.2f,%.2f,%.2f)\n", body, sx, sy, sz);
        }
    }
    if (!key_state['B'] && !key_state['b']) {
        box_dropped = false;
    }

    // Keyboard orbit controls (arrow keys / WASD for orbit without mouse)
    float key_orbit_speed = 2.0f * dt;  // radians per second
    if (key_state['W'] || key_state['w']) { orbit_phi_delta -= key_orbit_speed * 0.5f; }  // look up
    if (key_state['S'] || key_state['s']) { orbit_phi_delta += key_orbit_speed * 0.5f; }  // look down
    if (key_state['A'] || key_state['a']) { orbit_theta_delta += key_orbit_speed; }        // orbit left
    if (key_state['D'] || key_state['d']) { orbit_theta_delta -= key_orbit_speed; }        // orbit right

    orbit_key_input(dt);
    orbit_apply_damping(dt);
}

// GLUT keyboard callbacks - track key state.
static void key_down(unsigned char key, int, int) {
    if (key < 256) key_state[key] = true;
    // if (debug_frame == 0 || (key == 'W' || key == 'w')) {
    //     printf("[key down] %c (ascii=%d)\n", (key >= 32 && key < 127) ? (char)key : '?', (int)key);
    // }
}

static void key_up(unsigned char key, int, int) {
    if (key < 256) key_state[key] = false;
}

static void mouse_motion(int x, int y) {
    if (mouse_left_down) {
        // Orbit: left button drag changes theta and phi (spherical coordinates)
        int dx = x - mouse_last_x;
        int dy = y - mouse_last_y;

        orbit_theta_delta -= dx * orbit_rotate_speed;
        orbit_phi_delta   -= dy * orbit_rotate_speed;
    } else if (mouse_right_down) {
        // Pan: right button drag moves the orbit target in camera's X/Y plane
        int dx = x - mouse_last_x;
        int dy = y - mouse_last_y;

        // Pan speed scales with distance (closer = smaller pan, further = larger pan)
        float pan_scale = orbit_radius * orbit_pan_speed / 100.0f;

        // Compute camera right and up vectors for panning
        float sin_theta = sinf(orbit_theta);
        float cos_theta = cosf(orbit_theta);

        // Camera right vector (tangent to orbit, horizontal)
        float right_x = cos_theta;
        float right_z = -sin_theta;

        // Camera up vector (in the orbit plane, vertical)
        float up_x = sin_theta * cosf(orbit_phi);
        float up_y = cosf(orbit_phi);
        float up_z = -cos_theta * cosf(orbit_phi);

        // Pan: -dx moves target right, -dy moves target up (matching screen direction)
        // Negate dx because dragging right should move camera left (target moves opposite)
        orbit_pan_x_delta += (-dx * right_x + dy * up_x) * pan_scale;
        orbit_pan_y_delta += (dy * up_y) * pan_scale;
        orbit_pan_z_delta += (-dx * right_z + dy * up_z) * pan_scale;
    }

    mouse_last_x = x;
    mouse_last_y = y;
}

static void mouse_button(int button, int state, int x, int y) {
    if (button == GLUT_LEFT_BUTTON) {
        mouse_left_down = (state == GLUT_DOWN);
        if (mouse_left_down) {
            mouse_last_x = x;
            mouse_last_y = y;
        }
    } else if (button == GLUT_RIGHT_BUTTON) {
        mouse_right_down = (state == GLUT_DOWN);
        if (mouse_right_down) {
            mouse_last_x = x;
            mouse_last_y = y;
        }
    } else if (button == GLUT_MIDDLE_BUTTON) {
        // Middle button: pan (alternative to right button)
        mouse_right_down = (state == GLUT_DOWN);
        if (mouse_right_down) {
            mouse_last_x = x;
            mouse_last_y = y;
        }
    }
}

// Mouse wheel callback (freeglut extension — works on Windows/macOS/Linux)
static void mouse_wheel(int wheel, int direction, int x, int y) {
    // direction: +1 = up (zoom in), -1 = down (zoom out)
    // three.js style: exponential zoom
    float zoom_factor = 1.0f - (float)direction * orbit_zoom_speed;
    orbit_radius_delta -= orbit_radius * (zoom_factor - 1.0f);
    (void)x; (void)y; (void)wheel;
}

static void resize(int w, int h) {
    glViewport(0, 0, w, h);
}



static void render() {
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_NORMALIZE);
	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);
	glEnable(GL_LIGHT1);
	
	// Fallback clear color — sky sphere covers the view but this handles edges.
	glClearColor(0.06f, 0.07f, 0.12f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// Specular material — shiny highlights on all surfaces.
	GLfloat specular[] = { 0.4f, 0.4f, 0.4f, 1.0f };
	GLfloat shininess[] = { 32.0f };
	glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, specular);
	glMaterialfv(GL_FRONT_AND_BACK, GL_SHININESS, shininess);

	// Enable color material so glColor3f drives diffuse + ambient per-body.
	glColorMaterial(GL_FRONT_AND_BACK, GL_DIFFUSE);
	glEnable(GL_COLOR_MATERIAL);
	
	// Setup projection.
	GLint viewport[4];
	glGetIntegerv(GL_VIEWPORT, viewport);
	glMatrixMode(GL_PROJECTION);
	
	float fov = 0.25f;
	float aspect = (float)viewport[2]/(float)viewport[3];
	float near_z = 1.0f;
	float far_z = 1000.0f;
	{
		float dy = 2.0f * near_z * tanf(fov);
		float zn = 2.0f * near_z;
		float dz = far_z - near_z;
		
		float m[16] = {
			zn / (dy * aspect), 0.0f, 0.0f, 0.0f,
			0.0f, zn / dy, 0.0f, 0.0f,
			0.0f, 0.0f, (-far_z - near_z) / dz, -1.0f,
			0.0f, 0.0f, (-zn * far_z) / dz, 0.0f,
		};
		
		glLoadMatrixf(m);
	}

	// Switch to modelview and set up orbit camera with gluLookAt.
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	{
		float eye[3];
		float target[3];
		orbit_camera_eye(eye);
		orbit_camera_target(target);
		gluLookAt(
			eye[0], eye[1], eye[2],
			target[0], target[1], target[2],
			0.0f, 1.0f, 0.0f  // world up vector
		);
	}

	// Light 0 — main directional (warm, strong direct).
	GLfloat l0_ambient[] = { 0.15f, 0.15f, 0.18f, 1.0f };
	GLfloat l0_diffuse[] = { 1.0f, 0.95f, 0.9f, 1.0f };
	GLfloat l0_position[] = { 2.0f, 3.0f, 1.0f, 0.0f };
	glLightfv(GL_LIGHT0, GL_AMBIENT, l0_ambient);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, l0_diffuse);
	glLightfv(GL_LIGHT0, GL_POSITION, l0_position);

	// Light 1 — fill from opposite side (cool, weaker).
	GLfloat l1_ambient[] = { 0.0f, 0.0f, 0.0f, 1.0f };
	GLfloat l1_diffuse[] = { 0.3f, 0.35f, 0.5f, 1.0f };
	GLfloat l1_position[] = { -1.0f, 1.0f, -2.0f, 0.0f };
	glLightfv(GL_LIGHT1, GL_AMBIENT, l1_ambient);
	glLightfv(GL_LIGHT1, GL_DIFFUSE, l1_diffuse);
	glLightfv(GL_LIGHT1, GL_POSITION, l1_position);

	// Global ambient — dim fill so shadows are visible.
	GLfloat global_ambient[] = { 0.08f, 0.08f, 0.1f, 1.0f };
	glLightModelfv(GL_LIGHT_MODEL_AMBIENT, global_ambient);

	// Draw procedural gradient sky (before scene, depth write off).
	draw_sky();
	
	// Render boxes.
	for (unsigned i = 0; i < colliders.boxes.count; ++i) {
		unsigned body = colliders.boxes.transforms[i].body;

		float scale[3];
		float rotation[4];
		float position[3];

		memcpy(scale, colliders.boxes.data[i].size, sizeof(scale));

		quaternion_concat(rotation, bodies.transforms[body].rotation, colliders.boxes.transforms[i].rotation);
		quaternion_transform(position, bodies.transforms[body].rotation, colliders.boxes.transforms[i].position);

		position[0] += bodies.transforms[body].position[0];
		position[1] += bodies.transforms[body].position[1];
		position[2] += bodies.transforms[body].position[2];

		float m[16];
		matrix(m, scale, rotation, position);

		// Per-body color.
		if (body == 0) {
			// Ground — dark greenish-grey.
			glColor3f(0.25f, 0.3f, 0.2f);
		} else {
			float hue = (float)(body * 7 + 13) / 256.0f;
			float cr = 0.5f + 0.5f * cosf(6.283f * (hue + 0.0f));
			float cg = 0.5f + 0.5f * cosf(6.283f * (hue + 0.33f));
			float cb = 0.5f + 0.5f * cosf(6.283f * (hue + 0.66f));
			glColor3f(cr, cg, cb);
		}

		glPushMatrix();
		glMultMatrixf(m);
		glutSolidCube(2.0f);
		glPopMatrix();
	}

	// Render spheres.
	for (unsigned i = 0; i < colliders.spheres.count; ++i) {
		unsigned body = colliders.spheres.transforms[i].body;

		float scale[3];
		float rotation[4];
		float position[3];

		scale[0] = scale[1] = scale[2] = colliders.spheres.data[i].radius;

		quaternion_concat(rotation, bodies.transforms[body].rotation, colliders.spheres.transforms[i].rotation);
		quaternion_transform(position, bodies.transforms[body].rotation, colliders.spheres.transforms[i].position);

		position[0] += bodies.transforms[body].position[0];
		position[1] += bodies.transforms[body].position[1];
		position[2] += bodies.transforms[body].position[2];

		float m[16];
		matrix(m, scale, rotation, position);

		// Per-body color — offset hue so spheres differ from boxes.
		float hue = (float)(body * 7 + 13) / 256.0f;
		float cr = 0.5f + 0.5f * cosf(6.283f * (hue + 0.0f + 0.5f));
		float cg = 0.5f + 0.5f * cosf(6.283f * (hue + 0.33f + 0.5f));
		float cb = 0.5f + 0.5f * cosf(6.283f * (hue + 0.66f + 0.5f));
		glColor3f(cr, cg, cb);

		glPushMatrix();
		glMultMatrixf(m);
		glutSolidSphere(1.0f, 32, 16);
		glPopMatrix();
	}
	
	glutSwapBuffers();
	// Reset color material for safety.
	glDisable(GL_COLOR_MATERIAL);
}

// Draw a large procedural gradient sphere as the sky.
// Vertical gradient: dark horizon → mid blue → near-black zenith.
static void draw_sky() {
	const float radius = 500.0f;
	const unsigned bands = 32;
	const unsigned rings = 32;

	glDisable(GL_LIGHTING);
	glDisable(GL_COLOR_MATERIAL);
	glDisable(GL_DEPTH_WRITEMASK);

	// Pre-compute theta angles for each latitude band.
	float theta[bands + 1], sin_t[bands + 1], cos_t[bands + 1];
	for (unsigned i = 0; i <= bands; ++i) {
		theta[i] = (float)i / (float)bands * 3.14159f;
		sin_t[i] = sinf(theta[i]);
		cos_t[i] = cosf(theta[i]);
	}

	glBegin(GL_QUADS);
	for (unsigned i = 0; i < bands; ++i) {
		// Color: interpolate horizon → mid → zenith.
		float t = (float)(i + 0.5f) / (float)bands;
		float r, g, b;
		if (t < 0.5f) {
			float s = t * 2.0f;
			r = 0.15f + (0.25f - 0.15f) * s;
			g = 0.15f + (0.28f - 0.15f) * s;
			b = 0.22f + (0.40f - 0.22f) * s;
		} else {
			float s = (t - 0.5f) * 2.0f;
			r = 0.25f + (0.06f - 0.25f) * s;
			g = 0.28f + (0.07f - 0.28f) * s;
			b = 0.40f + (0.12f - 0.40f) * s;
		}

		for (unsigned j = 0; j < rings; ++j) {
			float phi0 = (float)j       / (float)rings * 6.28318f;
			float phi1 = (float)(j + 1) / (float)rings * 6.28318f;

			// Emit a quad between latitude bands i and i+1.
			glColor3f(r, g, b);
			glVertex3f(sin_t[i]   * cosf(phi0) * radius, cos_t[i]   * radius, sin_t[i]   * sinf(phi0) * radius);
			glVertex3f(sin_t[i + 1] * cosf(phi0) * radius, cos_t[i + 1] * radius, sin_t[i + 1] * sinf(phi0) * radius);
			glVertex3f(sin_t[i + 1] * cosf(phi1) * radius, cos_t[i + 1] * radius, sin_t[i + 1] * sinf(phi1) * radius);
			glVertex3f(sin_t[i]   * cosf(phi1) * radius, cos_t[i]   * radius, sin_t[i]   * sinf(phi1) * radius);
		}
	}
	glEnd();

	glEnable(GL_DEPTH_WRITEMASK);
	glEnable(GL_LIGHTING);
	glEnable(GL_COLOR_MATERIAL);
}

static void simulate() {
	static const unsigned steps = 2;
	static const unsigned iterations = 10;
	static unsigned sim_frame = 0;

	float time_step = 1.0f / (60.0f * (float)steps);

	// Reduce gravity and increase damping initially to prevent explosion
	// from massive initial contacts when planks are stacked tightly.
	float gravity_scale = 1.0f;
	float damping_extra = 0.0f;
	if (sim_frame < 120) {
		gravity_scale = 0.1f; // 10% gravity for first 2 seconds
		damping_extra = 0.1f; // extra damping for first 2 seconds
	}

	for (unsigned n = 0; n < steps; ++n) {
		nudge::Arena temporary = arena;

		nudge::BodyConnections connections = {};
		nudge::collide(&active_bodies, &contact_data, bodies, colliders, connections, temporary);

		float damping = 1.0f - time_step * 0.25f - damping_extra;

		for (unsigned i = 0; i < active_bodies.count; ++i) {
			unsigned index = active_bodies.indices[i];

			bodies.momentum[index].velocity[1] -= 9.82f * gravity_scale * time_step;

			bodies.momentum[index].velocity[0] *= damping;
			bodies.momentum[index].velocity[1] *= damping;
			bodies.momentum[index].velocity[2] *= damping;

			bodies.momentum[index].angular_velocity[0] *= damping;
			bodies.momentum[index].angular_velocity[1] *= damping;
			bodies.momentum[index].angular_velocity[2] *= damping;

			// Clamp velocities to prevent NaN explosion
			for (unsigned ax = 0; ax < 3; ++ax) {
				float v = bodies.momentum[index].velocity[ax];
				float av = fabsf(v);
				if (av > 10000.0f || v != v) {
					bodies.momentum[index].velocity[ax] = (v != v) ? 0.0f : (av > 10000.0f ? (v < 0 ? -10000.0f : 10000.0f) : v);
				}
				v = bodies.momentum[index].angular_velocity[ax];
				av = fabsf(v);
				if (av > 1000.0f || v != v) {
					bodies.momentum[index].angular_velocity[ax] = (v != v) ? 0.0f : (av > 1000.0f ? (v < 0 ? -1000.0f : 1000.0f) : v);
				}
			}
		}

		nudge::ContactImpulseData* contact_impulses = nudge::read_cached_impulses(contact_cache, contact_data, &temporary);
		nudge::ContactConstraintData* contact_constraints = nudge::setup_contact_constraints(active_bodies, contact_data, bodies, contact_impulses, &temporary);

		for (unsigned i = 0; i < iterations; ++i) {
			nudge::apply_impulses(contact_constraints, bodies);
		}

		nudge::update_cached_impulses(contact_constraints, contact_impulses);
		nudge::write_cached_impulses(&contact_cache, contact_data, contact_impulses);
		nudge::advance(active_bodies, bodies, time_step);
	}

	sim_frame++;
	if (sim_frame % 60 == 0) {
		printf("[sim] frame=%u bodies=%u contacts=%u\n", sim_frame, bodies.count, contact_data.count);
	}
}

static void timer(int) {
	orbit_camera_update(1.0f / 60.0f);
	glutPostRedisplay();
	glutTimerFunc(16, timer, 0);
	simulate();
}

int main(int argc, const char* argv[]) {
	// Disable denormals for performance.
	_MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
	_MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
	

	// Print information about instruction set.
#ifdef __AVX512F__
	printf("Using 16-wide AVX\n");
#elif defined(__AVX2__)
	printf("Using 8-wide AVX\n");
#else
	printf("Using 4-wide SSE\n");
#if defined(__SSE4_1__) || defined(__AVX__)
	printf("BLENDVPS: Enabled\n");
#else
	printf("BLENDVPS: Disabled\n");
#endif
#endif
	
#ifdef __FMA__
	printf("FMA: Enabled\n");
#else
	printf("FMA: Disabled\n");
#endif
	
	// Allocate memory for simulation arena.
	// arena.size = 64*1024*1024;
	arena.size = 512*1024*1024;
	arena.data = _mm_malloc(arena.size, 4096);
	
	// Allocate memory for bodies, colliders, and contacts.
	active_bodies.capacity = max_box_count;
	active_bodies.indices = static_cast<uint16_t*>(_mm_malloc(sizeof(uint16_t)*max_body_count, 64));
	
	bodies.idle_counters = static_cast<uint8_t*>(_mm_malloc(sizeof(uint8_t)*max_body_count, 64));
	bodies.transforms = static_cast<nudge::Transform*>(_mm_malloc(sizeof(nudge::Transform)*max_body_count, 64));
	bodies.momentum = static_cast<nudge::BodyMomentum*>(_mm_malloc(sizeof(nudge::BodyMomentum)*max_body_count, 64));
	bodies.properties = static_cast<nudge::BodyProperties*>(_mm_malloc(sizeof(nudge::BodyProperties)*max_body_count, 64));
	
	colliders.boxes.data = static_cast<nudge::BoxCollider*>(_mm_malloc(sizeof(nudge::BoxCollider)*max_box_count, 64));
	colliders.boxes.tags = static_cast<uint16_t*>(_mm_malloc(sizeof(uint16_t)*max_box_count, 64));
	colliders.boxes.transforms = static_cast<nudge::Transform*>(_mm_malloc(sizeof(nudge::Transform)*max_box_count, 64));
	
	colliders.spheres.data = static_cast<nudge::SphereCollider*>(_mm_malloc(sizeof(nudge::SphereCollider)*max_sphere_count, 64));
	colliders.spheres.tags = static_cast<uint16_t*>(_mm_malloc(sizeof(uint16_t)*max_sphere_count, 64));
	colliders.spheres.transforms = static_cast<nudge::Transform*>(_mm_malloc(sizeof(nudge::Transform)*max_sphere_count, 64));
	
	contact_data.capacity = max_body_count*64;
	contact_data.bodies = static_cast<nudge::BodyPair*>(_mm_malloc(sizeof(nudge::BodyPair)*contact_data.capacity, 64));
	contact_data.data = static_cast<nudge::Contact*>(_mm_malloc(sizeof(nudge::Contact)*contact_data.capacity, 64));
	contact_data.tags = static_cast<uint64_t*>(_mm_malloc(sizeof(uint64_t)*contact_data.capacity, 64));
	contact_data.sleeping_pairs = static_cast<uint32_t*>(_mm_malloc(sizeof(uint32_t)*contact_data.capacity, 64));
	
	contact_cache.capacity = max_body_count*64;
	contact_cache.data = static_cast<nudge::CachedContactImpulse*>(_mm_malloc(sizeof(nudge::CachedContactImpulse)*contact_cache.capacity, 64));
	contact_cache.tags = static_cast<uint64_t*>(_mm_malloc(sizeof(uint64_t)*contact_cache.capacity, 64));
	
	// The first body is the static world.
	bodies.count = 1;
	bodies.idle_counters[0] = 0;
	bodies.transforms[0] = identity_transform;
	memset(bodies.momentum, 0, sizeof(bodies.momentum[0]));
	memset(bodies.properties, 0, sizeof(bodies.properties[0]));
	
	// Add ground.
	{
		unsigned collider = colliders.boxes.count++;

		colliders.boxes.transforms[collider] = identity_transform;
		colliders.boxes.transforms[collider].position[1] -= 20.0f;

		colliders.boxes.data[collider].size[0] = 400.0f;
		colliders.boxes.data[collider].size[1] = 10.0f;
		colliders.boxes.data[collider].size[2] = 400.0f;
		colliders.boxes.tags[collider] = collider;
	}

	// Build the scene
	// Change to build_old_scene() for the old random boxes+spheres scene
	build_old_scene();
	// build_scene();
	
	// Start GLUT.
	glutInit(&argc, const_cast<char**>(argv));
	glutInitDisplayMode(GLUT_RGBA | GLUT_DEPTH | GLUT_DOUBLE);
	glutInitWindowSize(1560, 1024);
	glutCreateWindow("nudge");
	glutDisplayFunc(render);
	glutKeyboardFunc(key_down);
	glutKeyboardUpFunc(key_up);
	glutMotionFunc(mouse_motion);
	glutPassiveMotionFunc(mouse_motion);
	glutMouseFunc(mouse_button);
	glutMouseWheelFunc(mouse_wheel);
	glutReshapeFunc(resize);

	printf("OrbitControls: LMB drag=orbit | RMB/MDB drag=pan | scroll=zoom | WASD=keyboard orbit | Q/E=zoom | B=drop box | Space=shoot sphere | Esc=quit\n");

	timer(0);
	
	glutMainLoop();
	
	return 0;
}
