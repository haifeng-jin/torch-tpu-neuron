#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

// ---------------------------------------------------------------------------
// Self-contained OpenXLA PJRT C-API ABI definitions
// ---------------------------------------------------------------------------

extern "C" {

typedef struct PJRT_Extension_Base {
  size_t struct_size;
  int type;
  struct PJRT_Extension_Base* next;
} PJRT_Extension_Base;

typedef struct PJRT_Api_Version {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  int major_version;
  int minor_version;
} PJRT_Api_Version;

typedef struct PJRT_Error PJRT_Error;
typedef struct PJRT_Client PJRT_Client;
typedef struct PJRT_Device PJRT_Device;
typedef struct PJRT_Memory PJRT_Memory;
typedef struct PJRT_DeviceDescription PJRT_DeviceDescription;
typedef struct PJRT_LoadedExecutable PJRT_LoadedExecutable;
typedef struct PJRT_Buffer PJRT_Buffer;
typedef struct PJRT_Event PJRT_Event;
typedef struct PJRT_ExecuteContext PJRT_ExecuteContext;
typedef struct PJRT_MultiSlice_Config PJRT_MultiSlice_Config;
typedef struct PJRT_SendCallbackInfo PJRT_SendCallbackInfo;
typedef struct PJRT_RecvCallbackInfo PJRT_RecvCallbackInfo;
typedef struct PJRT_HloOutputCallbackInfo PJRT_HloOutputCallbackInfo;

typedef enum {
  PJRT_NamedValue_kString = 0,
  PJRT_NamedValue_kInt64 = 1,
  PJRT_NamedValue_kInt64List = 2,
  PJRT_NamedValue_kFloat = 3,
  PJRT_NamedValue_kBool = 4,
} PJRT_NamedValue_Type;

typedef struct PJRT_NamedValue {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  const char* name;
  size_t name_size;
  PJRT_NamedValue_Type type;
  union {
    const char* string_value;
    int64_t int64_value;
    const int64_t* int64_array_value;
    float float_value;
    bool bool_value;
  };
  size_t value_size;
} PJRT_NamedValue;

typedef struct PJRT_Program {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  char* code;
  size_t code_size;
  const char* format;
  size_t format_size;
} PJRT_Program;

typedef struct PJRT_Client_Create_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  const PJRT_NamedValue* create_options;
  size_t num_options;
  void* kv_get_callback;
  void* kv_get_user_arg;
  void* kv_put_callback;
  void* kv_put_user_arg;
  PJRT_Client* client;
  void* kv_try_get_callback;
  void* kv_try_get_user_arg;
} PJRT_Client_Create_Args;

typedef struct PJRT_Client_PlatformName_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Client* client;
  const char* platform_name;
  size_t platform_name_size;
} PJRT_Client_PlatformName_Args;

typedef struct PJRT_DeviceDescription_Kind_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_DeviceDescription* device_description;
  const char* device_kind;
  size_t device_kind_size;
} PJRT_DeviceDescription_Kind_Args;

typedef struct PJRT_Client_Compile_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Client* client;
  const PJRT_Program* program;
  const char* compile_options;
  size_t compile_options_size;
  PJRT_LoadedExecutable* executable;
} PJRT_Client_Compile_Args;

typedef struct PJRT_ExecuteOptions {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_SendCallbackInfo** send_callbacks;
  PJRT_RecvCallbackInfo** recv_callbacks;
  size_t num_send_ops;
  size_t num_recv_ops;
  int launch_id;
  const int64_t* non_donatable_input_indices;
  size_t num_non_donatable_input_indices;
  PJRT_ExecuteContext* context;
  const char* call_location;
  size_t num_tasks;
  int* task_ids;
  int64_t* incarnation_ids;
  PJRT_MultiSlice_Config* multi_slice_config;
  bool use_major_to_minor_data_layout_for_callbacks;
  PJRT_HloOutputCallbackInfo* hlo_output_callbacks;
  size_t num_hlo_output_callbacks;
  const PJRT_NamedValue* custom_options;
  size_t num_custom_options;
} PJRT_ExecuteOptions;

typedef struct PJRT_LoadedExecutable_Execute_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_LoadedExecutable* executable;
  PJRT_ExecuteOptions* options;
  PJRT_Buffer* const* const* argument_lists;
  size_t num_devices;
  size_t num_args;
  PJRT_Buffer** const* output_lists;
  PJRT_Event** device_complete_events;
  PJRT_Device* execute_device;
} PJRT_LoadedExecutable_Execute_Args;

typedef struct PJRT_Plugin_Initialize_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
} PJRT_Plugin_Initialize_Args;

typedef PJRT_Error* (*PJRT_Plugin_Initialize_Fn)(PJRT_Plugin_Initialize_Args*);
typedef PJRT_Error* (*PJRT_Client_Create_Fn)(PJRT_Client_Create_Args*);
typedef PJRT_Error* (*PJRT_Client_PlatformName_Fn)(PJRT_Client_PlatformName_Args*);
typedef PJRT_Error* (*PJRT_DeviceDescription_Kind_Fn)(PJRT_DeviceDescription_Kind_Args*);
typedef PJRT_Error* (*PJRT_Client_Compile_Fn)(PJRT_Client_Compile_Args*);
typedef PJRT_Error* (*PJRT_LoadedExecutable_Execute_Fn)(PJRT_LoadedExecutable_Execute_Args*);

typedef struct PJRT_Api {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Api_Version pjrt_api_version;

  void* PJRT_Error_Destroy;
  void* PJRT_Error_Message;
  void* PJRT_Error_GetCode;

  PJRT_Plugin_Initialize_Fn PJRT_Plugin_Initialize;
  void* PJRT_Plugin_Attributes;

  void* PJRT_Event_Destroy;
  void* PJRT_Event_IsReady;
  void* PJRT_Event_Error;
  void* PJRT_Event_Await;
  void* PJRT_Event_OnReady;

  PJRT_Client_Create_Fn PJRT_Client_Create;
  void* PJRT_Client_Destroy;
  PJRT_Client_PlatformName_Fn PJRT_Client_PlatformName;
  void* PJRT_Client_ProcessIndex;
  void* PJRT_Client_PlatformVersion;
  void* PJRT_Client_Devices;
  void* PJRT_Client_AddressableDevices;
  void* PJRT_Client_LookupDevice;
  void* PJRT_Client_LookupAddressableDevice;
  void* PJRT_Client_AddressableMemories;
  PJRT_Client_Compile_Fn PJRT_Client_Compile;
  void* PJRT_Client_DefaultDeviceAssignment;
  void* PJRT_Client_BufferFromHostBuffer;

  void* PJRT_DeviceDescription_Id;
  void* PJRT_DeviceDescription_ProcessIndex;
  void* PJRT_DeviceDescription_Attributes;
  PJRT_DeviceDescription_Kind_Fn PJRT_DeviceDescription_Kind;
  void* PJRT_DeviceDescription_DebugString;
  void* PJRT_DeviceDescription_ToString;

  void* PJRT_Device_GetDescription;
  void* PJRT_Device_IsAddressable;
  void* PJRT_Device_LocalHardwareId;
  void* PJRT_Device_AddressableMemories;
  void* PJRT_Device_DefaultMemory;
  void* PJRT_Device_MemoryStats;

  void* PJRT_Memory_Id;
  void* PJRT_Memory_Kind;
  void* PJRT_Memory_DebugString;
  void* PJRT_Memory_ToString;
  void* PJRT_Memory_AddressableByDevices;

  void* PJRT_Executable_Destroy;
  void* PJRT_Executable_Name;
  void* PJRT_Executable_NumReplicas;
  void* PJRT_Executable_NumPartitions;
  void* PJRT_Executable_NumOutputs;
  void* PJRT_Executable_SizeOfGeneratedCodeInBytes;
  void* PJRT_Executable_GetCostAnalysis;
  void* PJRT_Executable_OutputMemoryKinds;
  void* PJRT_Executable_OptimizedProgram;
  void* PJRT_Executable_Serialize;

  void* PJRT_LoadedExecutable_Destroy;
  void* PJRT_LoadedExecutable_GetExecutable;
  void* PJRT_LoadedExecutable_AddressableDevices;
  void* PJRT_LoadedExecutable_Delete;
  void* PJRT_LoadedExecutable_IsDeleted;
  PJRT_LoadedExecutable_Execute_Fn PJRT_LoadedExecutable_Execute;

  void* _reserved[128];
} PJRT_Api;

}  // extern "C"

namespace {

std::atomic<int64_t> g_compile_count{0};
std::atomic<int64_t> g_execute_count{0};
std::atomic<int64_t> g_last_program_size{0};
char g_backend_mode[64] = "uninitialized";

std::mutex g_init_mutex;
PJRT_Api g_real_api_copy;
const PJRT_Api* g_real_api = nullptr;
PJRT_Api g_wrapped_api;
void* g_plugin_handle = nullptr;

typedef const PJRT_Api* (*GetPjrtApiFn)();

void InstallHooks();

void EnsureInitialized() {
  std::lock_guard<std::mutex> lock(g_init_mutex);
  if (g_real_api != nullptr) {
    return;
  }

  const char* explicit_lib = std::getenv("TORCH_TPU_NEURON_BACKEND_LIB");

  // 1. Check if an explicit backend was requested via env var.
  if (explicit_lib != nullptr && explicit_lib[0] != '\0') {
    g_plugin_handle = dlopen(explicit_lib, RTLD_NOW | RTLD_LOCAL | RTLD_NODELETE);
    if (g_plugin_handle != nullptr) {
      std::snprintf(g_backend_mode, sizeof(g_backend_mode), "custom:%s", explicit_lib);
    } else {
      std::fprintf(stderr,
                   "[libtpu_neuron_shim] Warning: Failed to dlopen backend library '%s': %s\n",
                   explicit_lib, dlerror());
    }
  }

  // 2. Discover AWS Trainium / Neuron runtime (libneuronxla.so).
  if (g_plugin_handle == nullptr && g_real_api == nullptr) {
    const char* neuron_libs[] = {
        "libneuronxla.so",
        "/opt/aws/neuron/lib/libneuronxla.so",
        "/usr/local/lib/libneuronxla.so",
    };
    for (const char* lib_path : neuron_libs) {
      g_plugin_handle = dlopen(lib_path, RTLD_NOW | RTLD_LOCAL | RTLD_NODELETE);
      if (g_plugin_handle != nullptr) {
        std::snprintf(g_backend_mode, sizeof(g_backend_mode), "trainium_hw");
        std::fprintf(stderr,
                     "[libtpu_neuron_shim] Connected to AWS Trainium runtime (%s)\n",
                     lib_path);
        break;
      }
    }
  }

  if (g_real_api == nullptr && g_plugin_handle != nullptr) {
    GetPjrtApiFn get_api = reinterpret_cast<GetPjrtApiFn>(dlsym(g_plugin_handle, "GetPjrtApi"));
    if (get_api != nullptr) {
      const PJRT_Api* candidate_api = get_api();
      if (candidate_api != nullptr &&
          candidate_api->struct_size >= sizeof(PJRT_Api_Version) &&
          candidate_api->pjrt_api_version.major_version == 0) {
        size_t copy_size = candidate_api->struct_size;
        if (copy_size > sizeof(PJRT_Api)) {
          copy_size = sizeof(PJRT_Api);
        }
        std::memcpy(&g_real_api_copy, candidate_api, copy_size);
        g_real_api_copy.struct_size = copy_size;
        g_real_api = &g_real_api_copy;

        std::memcpy(&g_wrapped_api, &g_real_api_copy, copy_size);
        g_wrapped_api.struct_size = copy_size;
      } else {
        std::fprintf(stderr,
                     "[libtpu_neuron_shim] Warning: Candidate PJRT API failed validation (ptr=%p)\n",
                     candidate_api);
      }
    }
  }

  InstallHooks();
}

void InstallHooks() {
  g_wrapped_api.PJRT_Plugin_Initialize = [](PJRT_Plugin_Initialize_Args* args) -> PJRT_Error* {
    EnsureInitialized();
    if (g_real_api != nullptr && g_real_api->PJRT_Plugin_Initialize != nullptr) {
      return g_real_api->PJRT_Plugin_Initialize(args);
    }
    return nullptr;
  };

  g_wrapped_api.PJRT_Client_Create = [](PJRT_Client_Create_Args* args) -> PJRT_Error* {
    EnsureInitialized();
    if (g_real_api == nullptr || g_real_api->PJRT_Client_Create == nullptr) {
      return nullptr;
    }

    // Filter create options: strip "premapped_buffer_size" which torch_tpu
    // passes unconditionally, but libneuronxla.so rejects.
    std::vector<PJRT_NamedValue> filtered_options;
    if (args->create_options != nullptr) {
      for (size_t i = 0; i < args->num_options; ++i) {
        const PJRT_NamedValue& opt = args->create_options[i];
        if (opt.name != nullptr && std::strcmp(opt.name, "premapped_buffer_size") == 0) {
          continue;  // Strip unsupported option.
        }
        filtered_options.push_back(opt);
      }
    }

    PJRT_Client_Create_Args modified_args = *args;
    modified_args.create_options = filtered_options.empty() ? nullptr : filtered_options.data();
    modified_args.num_options = filtered_options.size();

    PJRT_Error* err = g_real_api->PJRT_Client_Create(&modified_args);
    if (err == nullptr) {
      args->client = modified_args.client;
    }
    return err;
  };

  g_wrapped_api.PJRT_Client_PlatformName = [](PJRT_Client_PlatformName_Args* args) -> PJRT_Error* {
    EnsureInitialized();
    if (g_real_api != nullptr && g_real_api->PJRT_Client_PlatformName != nullptr) {
      PJRT_Error* err = g_real_api->PJRT_Client_PlatformName(args);
      if (err != nullptr) return err;
    }
    static const char kNeuronPlatform[] = "neuron";
    args->platform_name = kNeuronPlatform;
    args->platform_name_size = sizeof(kNeuronPlatform) - 1;
    return nullptr;
  };

  g_wrapped_api.PJRT_DeviceDescription_Kind = [](PJRT_DeviceDescription_Kind_Args* args) -> PJRT_Error* {
    EnsureInitialized();
    if (g_real_api != nullptr && g_real_api->PJRT_DeviceDescription_Kind != nullptr) {
      return g_real_api->PJRT_DeviceDescription_Kind(args);
    }
    return nullptr;
  };

  g_wrapped_api.PJRT_Client_Compile = [](PJRT_Client_Compile_Args* args) -> PJRT_Error* {
    EnsureInitialized();
    g_compile_count.fetch_add(1, std::memory_order_relaxed);

    if (args != nullptr && args->program != nullptr) {
      g_last_program_size.store(static_cast<int64_t>(args->program->code_size),
                                std::memory_order_relaxed);
    }

    if (g_real_api != nullptr && g_real_api->PJRT_Client_Compile != nullptr) {
      return g_real_api->PJRT_Client_Compile(args);
    }
    return nullptr;
  };

  g_wrapped_api.PJRT_LoadedExecutable_Execute = [](PJRT_LoadedExecutable_Execute_Args* args) -> PJRT_Error* {
    EnsureInitialized();
    g_execute_count.fetch_add(1, std::memory_order_relaxed);
    if (g_real_api != nullptr && g_real_api->PJRT_LoadedExecutable_Execute != nullptr) {
      return g_real_api->PJRT_LoadedExecutable_Execute(args);
    }
    return nullptr;
  };
}

}  // namespace

extern "C" {

const PJRT_Api* GetPjrtApi() {
  EnsureInitialized();
  if (g_wrapped_api.struct_size == 0) {
    g_wrapped_api.struct_size = sizeof(PJRT_Api);
    g_wrapped_api.pjrt_api_version.struct_size = sizeof(PJRT_Api_Version);
    g_wrapped_api.pjrt_api_version.extension_start = nullptr;
    g_wrapped_api.pjrt_api_version.major_version = 0;
    g_wrapped_api.pjrt_api_version.minor_version = 117;
    InstallHooks();
  }
  return &g_wrapped_api;
}

void GetNeuronShimStats(int64_t* compile_count, int64_t* execute_count,
                        int64_t* last_program_size, const char** backend_mode) {
  EnsureInitialized();
  if (compile_count != nullptr) {
    *compile_count = g_compile_count.load(std::memory_order_relaxed);
  }
  if (execute_count != nullptr) {
    *execute_count = g_execute_count.load(std::memory_order_relaxed);
  }
  if (last_program_size != nullptr) {
    *last_program_size = g_last_program_size.load(std::memory_order_relaxed);
  }
  if (backend_mode != nullptr) {
    *backend_mode = g_backend_mode;
  }
}

}  // extern "C"
