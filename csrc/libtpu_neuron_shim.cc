#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <mutex>
#include <vector>

// Lower glibc symbol version requirement for dlopen/dlsym/dlerror to GLIBC_2.2.5
// on x86_64 Linux to ensure manylinux2014 / manylinux_2_17 compatibility across
// all AWS Deep Learning AMIs (including Amazon Linux 2).
#if defined(__linux__) && defined(__x86_64__)
__asm__(".symver dlopen,dlopen@GLIBC_2.2.5");
__asm__(".symver dlsym,dlsym@GLIBC_2.2.5");
__asm__(".symver dlerror,dlerror@GLIBC_2.2.5");
#endif

extern "C" {

typedef struct PJRT_NamedValue {
  size_t struct_size;
  void* extension_start;
  const char* name;
  size_t name_size;
  int type;
  union {
    const char* string_value;
    int64_t int64_value;
    const int64_t* int64_array_value;
    float float_value;
    bool bool_value;
  };
  size_t value_size;
} PJRT_NamedValue;

typedef struct PJRT_Client_Create_Args {
  size_t struct_size;
  void* extension_start;
  const PJRT_NamedValue* create_options;
  size_t num_options;
  void* kv_get_callback;
  void* kv_get_user_arg;
  void* kv_put_callback;
  void* kv_put_user_arg;
  void* client;
  void* kv_try_get_callback;
  void* kv_try_get_user_arg;
} PJRT_Client_Create_Args;

typedef void* (*PJRT_Client_Create_Fn)(PJRT_Client_Create_Args*);

// Minimal ABI layout of PJRT_Api up through PJRT_Client_Create:
// - struct_size + extension_start (2 words)
// - pjrt_api_version (struct_size + extension_start + 2 ints = 3 words)
// - Error methods (3 pointers: Destroy, Message, GetCode)
// - Plugin methods (2 pointers: Initialize, Attributes)
// - Event methods (5 pointers: Destroy, IsReady, Error, Await, OnReady)
// Total methods preceding PJRT_Client_Create = 3 + 2 + 5 = 10 pointers.
typedef struct PJRT_Api {
  size_t struct_size;
  void* extension_start;
  struct {
    size_t struct_size;
    void* extension_start;
    int major_version;
    int minor_version;
  } pjrt_api_version;
  void* _methods_before_create[10];
  PJRT_Client_Create_Fn PJRT_Client_Create;
  void* _remaining_methods[256];
} PJRT_Api;

}  // extern "C"

namespace {

std::mutex g_mu;
PJRT_Api g_real_api;
PJRT_Api g_wrapped_api;
bool g_loaded = false;

void* WrappedClientCreate(PJRT_Client_Create_Args* args) {
  if (!g_loaded || g_real_api.PJRT_Client_Create == nullptr) return nullptr;

  // Filter create options: strip "premapped_buffer_size" which torch_tpu
  // passes unconditionally, but libneuronxla.so rejects.
  std::vector<PJRT_NamedValue> filtered;
  if (args->create_options != nullptr) {
    for (size_t i = 0; i < args->num_options; ++i) {
      const auto& opt = args->create_options[i];
      if (opt.name != nullptr &&
          ((opt.name_size == 21 && std::strncmp(opt.name, "premapped_buffer_size", 21) == 0) ||
           std::strcmp(opt.name, "premapped_buffer_size") == 0)) {
        continue;
      }
      filtered.push_back(opt);
    }
  }

  PJRT_Client_Create_Args modified = *args;
  modified.create_options = filtered.empty() ? nullptr : filtered.data();
  modified.num_options = filtered.size();
  void* err = g_real_api.PJRT_Client_Create(&modified);
  if (err == nullptr) {
    args->client = modified.client;
  }
  return err;
}

void EnsureInitialized() {
  std::lock_guard<std::mutex> lock(g_mu);
  if (g_loaded) return;

  const char* env_lib = std::getenv("TORCH_TPU_NEURON_BACKEND_LIB");
  const char* candidates[] = {
      env_lib,
      "libneuronpjrt.so",
      "libneuronxla.so",
      "/opt/aws/neuron/lib/libneuronpjrt.so",
      "/opt/aws/neuron/lib/libneuronxla.so",
      "/usr/local/lib/libneuronpjrt.so",
      "/usr/local/lib/libneuronxla.so",
  };

  for (const char* path : candidates) {
    if (path == nullptr || path[0] == '\0') continue;
    void* handle = dlopen(path, RTLD_NOW | RTLD_LOCAL | RTLD_NODELETE);
    if (handle == nullptr) continue;

    auto get_api = reinterpret_cast<const PJRT_Api* (*)()>(dlsym(handle, "GetPjrtApi"));
    if (get_api == nullptr) continue;

    const PJRT_Api* candidate_api = get_api();
    if (candidate_api != nullptr && candidate_api->struct_size > 0) {
      size_t copy_size = candidate_api->struct_size < sizeof(PJRT_Api)
                             ? candidate_api->struct_size
                             : sizeof(PJRT_Api);
      std::memcpy(&g_real_api, candidate_api, copy_size);
      std::memcpy(&g_wrapped_api, candidate_api, copy_size);
      g_wrapped_api.PJRT_Client_Create = &WrappedClientCreate;
      g_loaded = true;
      std::fprintf(stderr, "[torch_tpu_neuron] Connected to AWS Trainium runtime (%s)\n", path);
      break;
    }
  }
}

}  // namespace

extern "C" {

const PJRT_Api* GetPjrtApi() {
  EnsureInitialized();
  return &g_wrapped_api;
}

int IsNeuronBackendLoaded() {
  EnsureInitialized();
  return g_loaded ? 1 : 0;
}

}  // extern "C"
