// Implementation unit for tinygltf.
// Defines TINYGLTF_IMPLEMENTATION here so the header emits the function bodies
// exactly once. TINYGLTF_NO_STB_IMAGE / _WRITE prevent ODR collisions with the
// existing stb_impl.cpp that already owns those symbols.
#define TINYGLTF_NO_STB_IMAGE
#define TINYGLTF_NO_STB_IMAGE_WRITE
#define TINYGLTF_IMPLEMENTATION
#include <tiny_gltf.h>
