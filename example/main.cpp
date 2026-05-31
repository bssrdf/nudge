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
#include <gl/gl.h>
#ifdef _WIN32
#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0601  // Windows 7+ for RAWMOUSE.lLastMoveX/Y
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

// Camera state
static float camera_position[3] = { 0.0f, 5.0f, 20.0f };
static float camera_yaw = 0.0f;    // radians, horizontal rotation
static float camera_pitch = 0.0f;  // radians, vertical rotation
static bool mouse_locked = false;

// Camera settings
static const float camera_move_speed = 40.0f;
static const float camera_mouse_sensitivity = 0.001f;
static const float camera_pitch_max = 1.5f;  // ~85 degrees

// Key state tracked by GLUT callbacks
static bool key_state[256] = {};
static int debug_frame = 0;

// Raw mouse input (Windows)
#ifdef _WIN32
static int raw_mouse_dx = 0, raw_mouse_dy = 0;
static HWND raw_input_hwnd = NULL;
static WNDPROC raw_input_old_proc = NULL;

static LRESULT CALLBACK RawInputWndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    if (msg == WM_INPUT) {
        UINT size = 0;
        GetRawInputData((HRAWINPUT)lParam, RID_INPUT, NULL, &size, sizeof(RAWINPUTHEADER));
        if (size > 0 && size < 4096) {
            uint8_t buf[4096];
            UINT read = GetRawInputData((HRAWINPUT)lParam, RID_INPUT, buf, &size, sizeof(RAWINPUTHEADER));
            if (read != UINT_MAX && read == size) {
                RAWINPUT* ri = (RAWINPUT*)buf;
                if (ri->header.dwType == RIM_TYPEMOUSE) {
                    LONG dx = ri->data.mouse.lLastX;
                    LONG dy = ri->data.mouse.lLastY;
                    raw_mouse_dx += dx;
                    raw_mouse_dy += dy;
                }
            }
        }
        return 0;
    }
    return CallWindowProc(raw_input_old_proc, hwnd, msg, wParam, lParam);
}

static void raw_input_init(HWND hwnd)
{
    raw_input_hwnd = hwnd;
    RAWINPUTDEVICE rid;
    rid.usUsagePage = 1;  // Generic desktop
    rid.usUsage = 2;       // Mouse
    rid.dwFlags = RIDEV_INPUTSINK;  // Receive even when not focused
    rid.hwndTarget = hwnd;
    RegisterRawInputDevices(&rid, 1, sizeof(rid));

    raw_input_old_proc = (WNDPROC)SetWindowLongPtr(hwnd, GWLP_WNDPROC, (LONG_PTR)RawInputWndProc);
}

static void raw_input_shutdown()
{
    if (raw_input_hwnd && raw_input_old_proc) {
        SetWindowLongPtr(raw_input_hwnd, GWLP_WNDPROC, (LONG_PTR)raw_input_old_proc);
    }
    raw_input_hwnd = NULL;
    raw_input_old_proc = NULL;
}

static void raw_input_poll() {
    if (!mouse_locked) return;
    if (raw_mouse_dx == 0 && raw_mouse_dy == 0) return;

    camera_yaw -= raw_mouse_dx * camera_mouse_sensitivity;
    camera_pitch -= raw_mouse_dy * camera_mouse_sensitivity;
    if (camera_pitch < -camera_pitch_max) camera_pitch = -camera_pitch_max;
    if (camera_pitch > camera_pitch_max) camera_pitch = camera_pitch_max;

    raw_mouse_dx = 0;
    raw_mouse_dy = 0;
}
#else
static void raw_input_poll() {}
#endif

static void draw_sky();

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

// Compute the camera's look-at target from yaw and pitch.
static inline void camera_look_target(float target[3], const float pos[3], float yaw, float pitch) {
    target[0] = pos[0] - sinf(yaw) * cosf(pitch);
    target[1] = pos[1] + sinf(pitch);
    target[2] = pos[2] - cosf(yaw) * cosf(pitch);
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
            // Place sphere just in front of the camera
            float cp = cosf(camera_pitch), sp = sinf(camera_pitch);
            float cy = cosf(camera_yaw), sy = sinf(camera_yaw);
            float dist = radius + 1.0f;
            bodies.transforms[body].position[0] = camera_position[0] - sy * cp * dist;
            bodies.transforms[body].position[1] = camera_position[1] + sp * dist;
            bodies.transforms[body].position[2] = camera_position[2] - cy * cp * dist;

            // Shoot along view direction
            const float shoot_speed = 200.0f;
            bodies.momentum[body].velocity[0] = -sy * cp * shoot_speed;
            bodies.momentum[body].velocity[1] = sp * shoot_speed;
            bodies.momentum[body].velocity[2] = -cy * cp * shoot_speed;
            printf("[shoot] sphere body=%u radius=%.1f speed=%.0f\n", body, radius, shoot_speed);
        }
    }
    if (!key_state[' ']) {
        sphere_shot = false;
    }
}

// Update camera position based on held keys and delta time.
static inline void camera_update(float dt) {
    // Debug: print key states every 30 frames to verify input.
    debug_frame++;
    if (debug_frame % 30 == 0) {
        printf("[camera] pos=(%.1f,%.1f,%.1f) yaw=%.2f pitch=%.2f mouse=%d keys:w=%d s=%d a=%d d=%d\n",
               camera_position[0], camera_position[1], camera_position[2],
               camera_yaw, camera_pitch, mouse_locked,
               key_state['W'], key_state['S'], key_state['A'], key_state['D']);
    }

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

    float speed = camera_move_speed * dt;
    if (key_state['X'] || key_state['x']) speed *= 3.0f;  // Hold X to sprint

    float cy = cosf(camera_yaw), sy = sinf(camera_yaw);
    float cp = cosf(camera_pitch), sp = sinf(camera_pitch);

    // Forward: -z in camera space (yaw + pitch for vertical movement)
    float forward_x = -sy * cp;
    float forward_y = sp;
    float forward_z = -cy * cp;

    // Right: +x in camera space (yaw only, always horizontal)
    float right_x = cy;
    float right_z = -sy;

    float dx = 0.0f, dy = 0.0f, dz = 0.0f;
    if (key_state['W'] || key_state['w']) { dx += forward_x; dy += forward_y; dz += forward_z; }
    if (key_state['S'] || key_state['s']) { dx -= forward_x; dy -= forward_y; dz -= forward_z; }
    if (key_state['A'] || key_state['a']) { dx -= right_x; dz -= right_z; }
    if (key_state['D'] || key_state['d']) { dx += right_x; dz += right_z; }

    // Normalize if diagonal
    float len = sqrtf(dx * dx + dy * dy + dz * dz);
    if (len > 0.001f) {
        dx = (dx / len) * speed;
        dy = (dy / len) * speed;
        dz = (dz / len) * speed;
    }

    camera_position[0] += dx;
    camera_position[1] += dy;
    camera_position[2] += dz;
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
    // No-op: camera rotation is handled by raw_input_poll() on Windows,
    // or falls through to normal GLUT motion when mouse is unlocked.
    (void)x; (void)y;
}

static void mouse_button(int button, int state, int x, int y) {
    if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {
        mouse_locked = !mouse_locked;
        printf("[mouse] locked=%d\n", mouse_locked);
    }
    (void)x; (void)y;
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

	// Switch to modelview and set up camera with gluLookAt.
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	{
		float target[3];
		camera_look_target(target, camera_position, camera_yaw, camera_pitch);
		gluLookAt(
			camera_position[0], camera_position[1], camera_position[2],
			target[0], target[1], target[2],
			0.0f, 1.0f, 0.0f  // up vector
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
	
	float time_step = 1.0f / (60.0f * (float)steps);
	
	for (unsigned n = 0; n < steps; ++n) {
		// Setup a temporary memory arena. The same temporary memory is reused each iteration.
		nudge::Arena temporary = arena;
		
		// Find contacts.
		nudge::BodyConnections connections = {}; // NOTE: Custom constraints should be added as body connections.
		nudge::collide(&active_bodies, &contact_data, bodies, colliders, connections, temporary);
		
		// NOTE: Custom contacts can be added here, e.g., against the static environment.
		
		// Apply gravity and damping.
		float damping = 1.0f - time_step*0.25f;
		
		for (unsigned i = 0; i < active_bodies.count; ++i) {
			unsigned index = active_bodies.indices[i];
			
			bodies.momentum[index].velocity[1] -= 9.82f * time_step;
			
			bodies.momentum[index].velocity[0] *= damping;
			bodies.momentum[index].velocity[1] *= damping;
			bodies.momentum[index].velocity[2] *= damping;
			
			bodies.momentum[index].angular_velocity[0] *= damping;
			bodies.momentum[index].angular_velocity[1] *= damping;
			bodies.momentum[index].angular_velocity[2] *= damping;
		}
		
		// Read previous impulses from contact cache.
		nudge::ContactImpulseData* contact_impulses = nudge::read_cached_impulses(contact_cache, contact_data, &temporary);
		
		// Setup contact constraints and apply the initial impulses.
		nudge::ContactConstraintData* contact_constraints = nudge::setup_contact_constraints(active_bodies, contact_data, bodies, contact_impulses, &temporary);
		
		// Apply contact impulses. Increasing the number of iterations will improve stability.
		for (unsigned i = 0; i < iterations; ++i) {
			nudge::apply_impulses(contact_constraints, bodies);
			// NOTE: Custom constraint impulses should be applied here.
		}
		
		// Update contact impulses.
		nudge::update_cached_impulses(contact_constraints, contact_impulses);
		
		// Write the updated contact impulses to the cache.
		nudge::write_cached_impulses(&contact_cache, contact_data, contact_impulses);
		
		// Move active bodies.
		nudge::advance(active_bodies, bodies, time_step);
	}
}

static void timer(int) {
	raw_input_poll();
	camera_update(1.0f / 60.0f);
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
	
	// Add boxes.
	for (unsigned i = 0; i < 8192; ++i) {
	// for (unsigned i = 0; i < 64; ++i) {
		// float sx = 1.f;
		// float sy = 1.f;
		// float sz = 1.f;
		float sx = (float)rand() * (1.0f/(float)RAND_MAX) + 0.5f;
		float sy = (float)rand() * (1.0f/(float)RAND_MAX) + 0.5f;
		float sz = (float)rand() * (1.0f/(float)RAND_MAX) + 0.5f;
		
		unsigned body = add_box(8.0f*sx*sy*sz, sx, sy, sz);
		
		// bodies.transforms[body].position[0] += (float)rand() * (1.0f/(float)RAND_MAX) * 10.0f - 5.0f;
		// bodies.transforms[body].position[1] += (float)rand() * (1.0f/(float)RAND_MAX) * 300.0f;
		// bodies.transforms[body].position[2] += (float)rand() * (1.0f/(float)RAND_MAX) * 10.0f - 5.0f;
		bodies.transforms[body].position[0] = (float)rand() * (1.0f/(float)RAND_MAX) * 10.0f - 5.0f;
		bodies.transforms[body].position[1] = (float)rand() * (1.0f/(float)RAND_MAX) * 300.0f;
		bodies.transforms[body].position[2] = (float)rand() * (1.0f/(float)RAND_MAX) * 10.0f - 5.0f;
		// bodies.transforms[body].position[0] += i/64.f * 10.0f - 5.0f;
		// bodies.transforms[body].position[1] += (i+10)/1024.f * 300.0f;
		// bodies.transforms[body].position[2] += i/64.f * 10.0f - 5.0f;
	}
	
	// Add spheres.
	for (unsigned i = 0; i < 1024; ++i) {
		float s = (float)rand() * (1.0f/(float)RAND_MAX) + 0.5f;
		
		unsigned body = add_sphere(4.18879f*s*s*s, s);
		
		bodies.transforms[body].position[0] += (float)rand() * (1.0f/(float)RAND_MAX) * 10.0f - 5.0f;
		bodies.transforms[body].position[1] += (float)rand() * (1.0f/(float)RAND_MAX) * 300.0f;
		bodies.transforms[body].position[2] += (float)rand() * (1.0f/(float)RAND_MAX) * 10.0f - 5.0f;
	}
	
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

#ifdef _WIN32
	HWND hwnd = GetForegroundWindow();
	if (hwnd) {
		raw_input_init(hwnd);
		printf("[raw input] initialized on window 0x%p\n", hwnd);
	} else {
		printf("[raw input] WARNING: no foreground window\n");
	}
#endif

	printf("Controls: click to lock/unlock mouse | WASD move (W/S follows pitch) | X sprint | B drop box | Space shoot sphere | Esc quit\n");
	printf("[debug] key states printed every 30 frames\n");

	timer(0);
	
	glutMainLoop();
	
	return 0;
}
