// Minimal standalone dummy PJRT plugin for AWS Trainium / Neuron testing.
// Emulates libneuronxla.so:
// 1. Rejects "premapped_buffer_size" in PJRT_Client_Create (matching real Trainium behavior).
// 2. Implements client/device/topology metadata so torch_tpu and torch.accelerator can initialize.

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

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

typedef struct PJRT_NamedValue {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
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

struct PJRT_Error {
  std::string message;
  int code;
};

typedef struct PJRT_Error_Destroy_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Error* error;
} PJRT_Error_Destroy_Args;

typedef struct PJRT_Error_Message_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  const PJRT_Error* error;
  const char* message;
  size_t message_size;
} PJRT_Error_Message_Args;

typedef struct PJRT_Error_GetCode_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  const PJRT_Error* error;
  int code;
} PJRT_Error_GetCode_Args;

typedef void (*PJRT_Error_PayloadVisitor)(const char* key, size_t key_size,
                                          const char* value, size_t value_size,
                                          void* user_arg);

typedef struct PJRT_Error_ForEachPayload_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  const PJRT_Error* error;
  PJRT_Error_PayloadVisitor visitor;
  void* user_arg;
} PJRT_Error_ForEachPayload_Args;

typedef struct PJRT_Plugin_Initialize_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
} PJRT_Plugin_Initialize_Args;

typedef struct PJRT_Plugin_Attributes_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  const PJRT_NamedValue* attributes;
  size_t num_attributes;
} PJRT_Plugin_Attributes_Args;

typedef struct PJRT_Client PJRT_Client;
typedef struct PJRT_Device PJRT_Device;
typedef struct PJRT_DeviceDescription PJRT_DeviceDescription;
typedef struct PJRT_TopologyDescription PJRT_TopologyDescription;

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

typedef struct PJRT_Client_Destroy_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Client* client;
} PJRT_Client_Destroy_Args;

typedef struct PJRT_Client_PlatformName_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Client* client;
  const char* platform_name;
  size_t platform_name_size;
} PJRT_Client_PlatformName_Args;

typedef struct PJRT_Client_ProcessIndex_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Client* client;
  int process_index;
} PJRT_Client_ProcessIndex_Args;

typedef struct PJRT_Client_PlatformVersion_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Client* client;
  const char* platform_version;
  size_t platform_version_size;
} PJRT_Client_PlatformVersion_Args;

typedef struct PJRT_Client_Devices_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Client* client;
  PJRT_Device* const* devices;
  size_t num_devices;
} PJRT_Client_Devices_Args;

typedef struct PJRT_Client_AddressableDevices_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Client* client;
  PJRT_Device* const* addressable_devices;
  size_t num_addressable_devices;
} PJRT_Client_AddressableDevices_Args;

typedef struct PJRT_Client_LookupDevice_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Client* client;
  int id;
  PJRT_Device* device;
} PJRT_Client_LookupDevice_Args;

typedef struct PJRT_Client_LookupAddressableDevice_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Client* client;
  int local_hardware_id;
  PJRT_Device* addressable_device;
} PJRT_Client_LookupAddressableDevice_Args;

typedef struct PJRT_Client_AddressableMemories_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Client* client;
  void* const* addressable_memories;
  size_t num_addressable_memories;
} PJRT_Client_AddressableMemories_Args;

typedef struct PJRT_Client_TopologyDescription_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Client* client;
  PJRT_TopologyDescription* topology;
} PJRT_Client_TopologyDescription_Args;

typedef struct PJRT_DeviceDescription_Id_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_DeviceDescription* device_description;
  int id;
} PJRT_DeviceDescription_Id_Args;

typedef struct PJRT_DeviceDescription_ProcessIndex_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_DeviceDescription* device_description;
  int process_index;
} PJRT_DeviceDescription_ProcessIndex_Args;

typedef struct PJRT_DeviceDescription_Attributes_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_DeviceDescription* device_description;
  size_t num_attributes;
  const PJRT_NamedValue* attributes;
} PJRT_DeviceDescription_Attributes_Args;

typedef struct PJRT_DeviceDescription_Kind_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_DeviceDescription* device_description;
  const char* device_kind;
  size_t device_kind_size;
} PJRT_DeviceDescription_Kind_Args;

typedef struct PJRT_DeviceDescription_DebugString_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_DeviceDescription* device_description;
  const char* debug_string;
  size_t debug_string_size;
} PJRT_DeviceDescription_DebugString_Args;

typedef struct PJRT_DeviceDescription_ToString_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_DeviceDescription* device_description;
  const char* to_string;
  size_t to_string_size;
} PJRT_DeviceDescription_ToString_Args;

typedef struct PJRT_Device_GetDescription_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Device* device;
  PJRT_DeviceDescription* device_description;
} PJRT_Device_GetDescription_Args;

typedef struct PJRT_Device_IsAddressable_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Device* device;
  bool is_addressable;
} PJRT_Device_IsAddressable_Args;

typedef struct PJRT_Device_LocalHardwareId_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Device* device;
  int local_hardware_id;
} PJRT_Device_LocalHardwareId_Args;

typedef struct PJRT_Device_AddressableMemories_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Device* device;
  void* const* memories;
  size_t num_memories;
} PJRT_Device_AddressableMemories_Args;

typedef struct PJRT_Device_DefaultMemory_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Device* device;
  void* memory;
} PJRT_Device_DefaultMemory_Args;

typedef struct PJRT_Device_MemoryStats_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Device* device;
  int64_t bytes_in_use;
  int64_t peak_bytes_in_use;
  bool peak_bytes_in_use_is_set;
  int64_t num_allocs;
  bool num_allocs_is_set;
  int64_t largest_alloc_size;
  bool largest_alloc_size_is_set;
  int64_t bytes_limit;
  bool bytes_limit_is_set;
  int64_t bytes_reserved;
  bool bytes_reserved_is_set;
  int64_t peak_bytes_reserved;
  bool peak_bytes_reserved_is_set;
  int64_t bytes_reservable_limit;
  bool bytes_reservable_limit_is_set;
  int64_t largest_free_block_bytes;
  bool largest_free_block_bytes_is_set;
  int64_t pool_bytes;
  bool pool_bytes_is_set;
  int64_t peak_pool_bytes;
  bool peak_pool_bytes_is_set;
  int64_t peak_allocated_bytes;
  bool peak_allocated_bytes_is_set;
} PJRT_Device_MemoryStats_Args;

typedef struct PJRT_TopologyDescription_Destroy_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_TopologyDescription* topology;
} PJRT_TopologyDescription_Destroy_Args;

typedef struct PJRT_TopologyDescription_PlatformName_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  const PJRT_TopologyDescription* topology;
  const char* platform_name;
  size_t platform_name_size;
} PJRT_TopologyDescription_PlatformName_Args;

typedef struct PJRT_TopologyDescription_PlatformVersion_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_TopologyDescription* topology;
  const char* platform_version;
  size_t platform_version_size;
} PJRT_TopologyDescription_PlatformVersion_Args;

typedef struct PJRT_TopologyDescription_GetDeviceDescriptions_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  const PJRT_TopologyDescription* topology;
  PJRT_DeviceDescription* const* descriptions;
  size_t num_descriptions;
} PJRT_TopologyDescription_GetDeviceDescriptions_Args;

typedef struct PJRT_TopologyDescription_Attributes_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_TopologyDescription* topology;
  const PJRT_NamedValue* attributes;
  size_t num_attributes;
} PJRT_TopologyDescription_Attributes_Args;

typedef struct PJRT_Api {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Api_Version pjrt_api_version;

  void (*PJRT_Error_Destroy)(PJRT_Error_Destroy_Args*);
  void (*PJRT_Error_Message)(PJRT_Error_Message_Args*);
  PJRT_Error* (*PJRT_Error_GetCode)(PJRT_Error_GetCode_Args*);

  PJRT_Error* (*PJRT_Plugin_Initialize)(PJRT_Plugin_Initialize_Args*);
  PJRT_Error* (*PJRT_Plugin_Attributes)(PJRT_Plugin_Attributes_Args*);

  void* PJRT_Event_Destroy;
  void* PJRT_Event_IsReady;
  void* PJRT_Event_Error;
  void* PJRT_Event_Await;
  void* PJRT_Event_OnReady;

  PJRT_Error* (*PJRT_Client_Create)(PJRT_Client_Create_Args*);
  PJRT_Error* (*PJRT_Client_Destroy)(PJRT_Client_Destroy_Args*);
  PJRT_Error* (*PJRT_Client_PlatformName)(PJRT_Client_PlatformName_Args*);
  PJRT_Error* (*PJRT_Client_ProcessIndex)(PJRT_Client_ProcessIndex_Args*);
  PJRT_Error* (*PJRT_Client_PlatformVersion)(PJRT_Client_PlatformVersion_Args*);
  PJRT_Error* (*PJRT_Client_Devices)(PJRT_Client_Devices_Args*);
  PJRT_Error* (*PJRT_Client_AddressableDevices)(PJRT_Client_AddressableDevices_Args*);
  PJRT_Error* (*PJRT_Client_LookupDevice)(PJRT_Client_LookupDevice_Args*);
  PJRT_Error* (*PJRT_Client_LookupAddressableDevice)(PJRT_Client_LookupAddressableDevice_Args*);
  PJRT_Error* (*PJRT_Client_AddressableMemories)(PJRT_Client_AddressableMemories_Args*);
  void* PJRT_Client_Compile;
  void* PJRT_Client_DefaultDeviceAssignment;
  void* PJRT_Client_BufferFromHostBuffer;

  PJRT_Error* (*PJRT_DeviceDescription_Id)(PJRT_DeviceDescription_Id_Args*);
  PJRT_Error* (*PJRT_DeviceDescription_ProcessIndex)(PJRT_DeviceDescription_ProcessIndex_Args*);
  PJRT_Error* (*PJRT_DeviceDescription_Attributes)(PJRT_DeviceDescription_Attributes_Args*);
  PJRT_Error* (*PJRT_DeviceDescription_Kind)(PJRT_DeviceDescription_Kind_Args*);
  PJRT_Error* (*PJRT_DeviceDescription_DebugString)(PJRT_DeviceDescription_DebugString_Args*);
  PJRT_Error* (*PJRT_DeviceDescription_ToString)(PJRT_DeviceDescription_ToString_Args*);

  PJRT_Error* (*PJRT_Device_GetDescription)(PJRT_Device_GetDescription_Args*);
  PJRT_Error* (*PJRT_Device_IsAddressable)(PJRT_Device_IsAddressable_Args*);
  PJRT_Error* (*PJRT_Device_LocalHardwareId)(PJRT_Device_LocalHardwareId_Args*);
  PJRT_Error* (*PJRT_Device_AddressableMemories)(PJRT_Device_AddressableMemories_Args*);
  PJRT_Error* (*PJRT_Device_DefaultMemory)(PJRT_Device_DefaultMemory_Args*);
  PJRT_Error* (*PJRT_Device_MemoryStats)(PJRT_Device_MemoryStats_Args*);

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
  void* PJRT_LoadedExecutable_Execute;
  void* PJRT_Executable_DeserializeAndLoad;
  void* PJRT_LoadedExecutable_Fingerprint;

  void* PJRT_Buffer_Destroy;
  void* PJRT_Buffer_ElementType;
  void* PJRT_Buffer_Dimensions;
  void* PJRT_Buffer_UnpaddedDimensions;
  void* PJRT_Buffer_DynamicDimensionIndices;
  void* PJRT_Buffer_GetMemoryLayout;
  void* PJRT_Buffer_OnDeviceSizeInBytes;
  void* PJRT_Buffer_Device;
  void* PJRT_Buffer_Memory;
  void* PJRT_Buffer_Delete;
  void* PJRT_Buffer_IsDeleted;
  void* PJRT_Buffer_CopyToDevice;
  void* PJRT_Buffer_ToHostBuffer;
  void* PJRT_Buffer_IsOnCpu;
  void* PJRT_Buffer_ReadyEvent;
  void* PJRT_Buffer_UnsafePointer;
  void* PJRT_Buffer_IncreaseExternalReferenceCount;
  void* PJRT_Buffer_DecreaseExternalReferenceCount;
  void* PJRT_Buffer_OpaqueDeviceMemoryDataPointer;

  void* PJRT_CopyToDeviceStream_Destroy;
  void* PJRT_CopyToDeviceStream_AddChunk;
  void* PJRT_CopyToDeviceStream_TotalBytes;
  void* PJRT_CopyToDeviceStream_GranuleSize;
  void* PJRT_CopyToDeviceStream_CurrentBytes;

  void* PJRT_TopologyDescription_Create;
  PJRT_Error* (*PJRT_TopologyDescription_Destroy)(PJRT_TopologyDescription_Destroy_Args*);
  PJRT_Error* (*PJRT_TopologyDescription_PlatformName)(PJRT_TopologyDescription_PlatformName_Args*);
  PJRT_Error* (*PJRT_TopologyDescription_PlatformVersion)(PJRT_TopologyDescription_PlatformVersion_Args*);
  PJRT_Error* (*PJRT_TopologyDescription_GetDeviceDescriptions)(PJRT_TopologyDescription_GetDeviceDescriptions_Args*);
  void* PJRT_TopologyDescription_Serialize;
  PJRT_Error* (*PJRT_TopologyDescription_Attributes)(PJRT_TopologyDescription_Attributes_Args*);

  void* PJRT_Compile;

  void* PJRT_Executable_OutputElementTypes;
  void* PJRT_Executable_OutputDimensions;
  void* PJRT_Buffer_CopyToMemory;
  void* PJRT_Client_CreateViewOfDeviceBuffer;
  void* PJRT_Executable_Fingerprint;

  PJRT_Error* (*PJRT_Client_TopologyDescription)(PJRT_Client_TopologyDescription_Args*);

  void* PJRT_Executable_GetCompiledMemoryStats;
  void* PJRT_Memory_Kind_Id;
  void* PJRT_ExecuteContext_Create;
  void* PJRT_ExecuteContext_Destroy;
  void* PJRT_Buffer_CopyRawToHost;
  void* PJRT_AsyncHostToDeviceTransferManager_Destroy;
  void* PJRT_AsyncHostToDeviceTransferManager_TransferData;
  void* PJRT_Client_CreateBuffersForAsyncHostToDevice;
  void* PJRT_AsyncHostToDeviceTransferManager_RetrieveBuffer;
  void* PJRT_AsyncHostToDeviceTransferManager_Device;
  void* PJRT_AsyncHostToDeviceTransferManager_BufferCount;
  void* PJRT_AsyncHostToDeviceTransferManager_BufferSize;
  void* PJRT_AsyncHostToDeviceTransferManager_SetBufferError;
  void* PJRT_AsyncHostToDeviceTransferManager_AddMetadata;
  void* PJRT_Client_DmaMap;
  void* PJRT_Client_DmaUnmap;
  void* PJRT_Client_CreateUninitializedBuffer;
  void* PJRT_Client_UpdateGlobalProcessInfo;
  void* PJRT_TopologyDescription_Deserialize;
  void* PJRT_Client_CreateAliasBuffer;
  void* PJRT_Client_FulfillAliasBuffer;
  void* PJRT_LoadedExecutable_GetDeviceAssignment;
  void* PJRT_Client_CreateErrorBuffer;
  void* PJRT_AsyncHostToDeviceTransferManager_TransferLiteral;
  void* PJRT_Buffer_CopyRawToHostFuture;
  void* PJRT_Device_PoisonExecution;
  void* PJRT_Device_CreateAsyncTrackingEvent;
  void* PJRT_AsyncTrackingEvent_Destroy;
  void* PJRT_Executable_GetCompileOptions;
  void* PJRT_Buffer_DonateWithControlDependency;
  void* PJRT_Event_Create;
  void* PJRT_Event_Set;
  void* PJRT_Device_GetAttributes;
  void* PJRT_Client_Load;
  void* PJRT_LoadedExecutable_AddressableDeviceLogicalIds;
  void* PJRT_Buffer_Bitcast;
  PJRT_Error* (*PJRT_Error_ForEachPayload)(PJRT_Error_ForEachPayload_Args*);
  void* PJRT_TopologyDescription_Fingerprint;
  void* PJRT_Executable_ParameterMemoryKinds;
  void* PJRT_Device_ClearMemoryStats;
  void* PJRT_TopologyDescription_MakeCanonicalShapeForMemorySpace;
  void* PJRT_TopologyDescription_GetMemorySpaceKindIds;
} PJRT_Api;

}  // extern "C"

struct PJRT_DeviceDescription {
  int id;
  int process_index;
  const char* kind;
  const char* debug_string;
  const char* to_string;
};

struct PJRT_Device {
  PJRT_DeviceDescription desc;
};

struct PJRT_TopologyDescription {
  const char* platform_name;
  const char* platform_version;
};

struct PJRT_Client {
  const char* platform_name;
  const char* platform_version;
};

namespace {

PJRT_Client g_client{"neuron", "AWS Neuron PJRT 2.x (dummy)"};
PJRT_TopologyDescription g_topology{"neuron", "AWS Neuron PJRT 2.x (dummy)"};

PJRT_Device g_devices[2] = {
    {PJRT_DeviceDescription{0, 0, "AWS Trainium", "NeuronDevice:0", "NeuronDevice(id=0)"}},
    {PJRT_DeviceDescription{1, 0, "AWS Trainium", "NeuronDevice:1", "NeuronDevice(id=1)"}},
};

PJRT_Device* g_device_ptrs[2] = {&g_devices[0], &g_devices[1]};
PJRT_DeviceDescription* g_desc_ptrs[2] = {&g_devices[0].desc, &g_devices[1].desc};

void Dummy_Error_Destroy(PJRT_Error_Destroy_Args* args) {
  if (args != nullptr && args->error != nullptr) {
    delete args->error;
  }
}

void Dummy_Error_Message(PJRT_Error_Message_Args* args) {
  if (args != nullptr && args->error != nullptr) {
    args->message = args->error->message.c_str();
    args->message_size = args->error->message.size();
  }
}

PJRT_Error* Dummy_Error_GetCode(PJRT_Error_GetCode_Args* args) {
  if (args != nullptr && args->error != nullptr) {
    args->code = args->error->code;
  }
  return nullptr;
}

PJRT_Error* Dummy_Error_ForEachPayload(PJRT_Error_ForEachPayload_Args*) {
  return nullptr;
}

PJRT_Error* Dummy_Plugin_Initialize(PJRT_Plugin_Initialize_Args*) {
  return nullptr;
}

PJRT_Error* Dummy_Plugin_Attributes(PJRT_Plugin_Attributes_Args* args) {
  if (args != nullptr) {
    args->attributes = nullptr;
    args->num_attributes = 0;
  }
  return nullptr;
}

PJRT_Error* Dummy_Client_Create(PJRT_Client_Create_Args* args) {
  if (args == nullptr) return nullptr;

  // Emulate real AWS Neuron libneuronxla.so behavior:
  // Reject "premapped_buffer_size" with INVALID_ARGUMENT (code 3).
  if (args->create_options != nullptr) {
    for (size_t i = 0; i < args->num_options; ++i) {
      const auto& opt = args->create_options[i];
      if (opt.name != nullptr) {
        if ((opt.name_size == 21 && std::strncmp(opt.name, "premapped_buffer_size", 21) == 0) ||
            std::strcmp(opt.name, "premapped_buffer_size") == 0) {
          return new PJRT_Error{"Unexpected option name passed to PJRT_Client_Create: premapped_buffer_size", 3};
        }
      }
    }
  }

  args->client = &g_client;
  return nullptr;
}

PJRT_Error* Dummy_Client_Destroy(PJRT_Client_Destroy_Args*) {
  return nullptr;
}

PJRT_Error* Dummy_Client_PlatformName(PJRT_Client_PlatformName_Args* args) {
  if (args != nullptr) {
    args->platform_name = "neuron";
    args->platform_name_size = 6;
  }
  return nullptr;
}

PJRT_Error* Dummy_Client_ProcessIndex(PJRT_Client_ProcessIndex_Args* args) {
  if (args != nullptr) {
    args->process_index = 0;
  }
  return nullptr;
}

PJRT_Error* Dummy_Client_PlatformVersion(PJRT_Client_PlatformVersion_Args* args) {
  if (args != nullptr) {
    args->platform_version = "AWS Neuron PJRT 2.x (dummy)";
    args->platform_version_size = std::strlen(args->platform_version);
  }
  return nullptr;
}

PJRT_Error* Dummy_Client_Devices(PJRT_Client_Devices_Args* args) {
  if (args != nullptr) {
    args->devices = g_device_ptrs;
    args->num_devices = 2;
  }
  return nullptr;
}

PJRT_Error* Dummy_Client_AddressableDevices(PJRT_Client_AddressableDevices_Args* args) {
  if (args != nullptr) {
    args->addressable_devices = g_device_ptrs;
    args->num_addressable_devices = 2;
  }
  return nullptr;
}

PJRT_Error* Dummy_Client_LookupDevice(PJRT_Client_LookupDevice_Args* args) {
  if (args != nullptr) {
    args->device = (args->id == 1) ? &g_devices[1] : &g_devices[0];
  }
  return nullptr;
}

PJRT_Error* Dummy_Client_LookupAddressableDevice(PJRT_Client_LookupAddressableDevice_Args* args) {
  if (args != nullptr) {
    args->addressable_device = (args->local_hardware_id == 1) ? &g_devices[1] : &g_devices[0];
  }
  return nullptr;
}

PJRT_Error* Dummy_Client_AddressableMemories(PJRT_Client_AddressableMemories_Args* args) {
  if (args != nullptr) {
    args->addressable_memories = nullptr;
    args->num_addressable_memories = 0;
  }
  return nullptr;
}

PJRT_Error* Dummy_Client_TopologyDescription(PJRT_Client_TopologyDescription_Args* args) {
  if (args != nullptr) {
    args->topology = &g_topology;
  }
  return nullptr;
}

PJRT_Error* Dummy_DeviceDescription_Id(PJRT_DeviceDescription_Id_Args* args) {
  if (args != nullptr && args->device_description != nullptr) {
    args->id = args->device_description->id;
  }
  return nullptr;
}

PJRT_Error* Dummy_DeviceDescription_ProcessIndex(PJRT_DeviceDescription_ProcessIndex_Args* args) {
  if (args != nullptr && args->device_description != nullptr) {
    args->process_index = args->device_description->process_index;
  }
  return nullptr;
}

PJRT_Error* Dummy_DeviceDescription_Attributes(PJRT_DeviceDescription_Attributes_Args* args) {
  if (args != nullptr) {
    args->attributes = nullptr;
    args->num_attributes = 0;
  }
  return nullptr;
}

PJRT_Error* Dummy_DeviceDescription_Kind(PJRT_DeviceDescription_Kind_Args* args) {
  if (args != nullptr && args->device_description != nullptr) {
    args->device_kind = args->device_description->kind;
    args->device_kind_size = std::strlen(args->device_kind);
  }
  return nullptr;
}

PJRT_Error* Dummy_DeviceDescription_DebugString(PJRT_DeviceDescription_DebugString_Args* args) {
  if (args != nullptr && args->device_description != nullptr) {
    args->debug_string = args->device_description->debug_string;
    args->debug_string_size = std::strlen(args->debug_string);
  }
  return nullptr;
}

PJRT_Error* Dummy_DeviceDescription_ToString(PJRT_DeviceDescription_ToString_Args* args) {
  if (args != nullptr && args->device_description != nullptr) {
    args->to_string = args->device_description->to_string;
    args->to_string_size = std::strlen(args->to_string);
  }
  return nullptr;
}

PJRT_Error* Dummy_Device_GetDescription(PJRT_Device_GetDescription_Args* args) {
  if (args != nullptr && args->device != nullptr) {
    args->device_description = &args->device->desc;
  }
  return nullptr;
}

PJRT_Error* Dummy_Device_IsAddressable(PJRT_Device_IsAddressable_Args* args) {
  if (args != nullptr) {
    args->is_addressable = true;
  }
  return nullptr;
}

PJRT_Error* Dummy_Device_LocalHardwareId(PJRT_Device_LocalHardwareId_Args* args) {
  if (args != nullptr && args->device != nullptr) {
    args->local_hardware_id = args->device->desc.id;
  }
  return nullptr;
}

PJRT_Error* Dummy_Device_AddressableMemories(PJRT_Device_AddressableMemories_Args* args) {
  if (args != nullptr) {
    args->memories = nullptr;
    args->num_memories = 0;
  }
  return nullptr;
}

PJRT_Error* Dummy_Device_DefaultMemory(PJRT_Device_DefaultMemory_Args* args) {
  if (args != nullptr) {
    args->memory = nullptr;
  }
  return new PJRT_Error{"DefaultMemory unimplemented in dummy", 12};
}

PJRT_Error* Dummy_Device_MemoryStats(PJRT_Device_MemoryStats_Args* args) {
  if (args != nullptr) {
    args->bytes_in_use = 0;
    args->peak_bytes_in_use = 0;
    args->peak_bytes_in_use_is_set = true;
    args->num_allocs = 0;
    args->num_allocs_is_set = true;
    args->largest_alloc_size = 0;
    args->largest_alloc_size_is_set = true;
    args->bytes_limit = 32LL * 1024 * 1024 * 1024;
    args->bytes_limit_is_set = true;
    args->bytes_reserved = 0;
    args->bytes_reserved_is_set = true;
    args->peak_bytes_reserved = 0;
    args->peak_bytes_reserved_is_set = true;
    args->bytes_reservable_limit = 32LL * 1024 * 1024 * 1024;
    args->bytes_reservable_limit_is_set = true;
    args->largest_free_block_bytes = 32LL * 1024 * 1024 * 1024;
    args->largest_free_block_bytes_is_set = true;
    args->pool_bytes = 0;
    args->pool_bytes_is_set = true;
    args->peak_pool_bytes = 0;
    args->peak_pool_bytes_is_set = true;
    args->peak_allocated_bytes = 0;
    args->peak_allocated_bytes_is_set = true;
  }
  return nullptr;
}

PJRT_Error* Dummy_TopologyDescription_Destroy(PJRT_TopologyDescription_Destroy_Args*) {
  return nullptr;
}

PJRT_Error* Dummy_TopologyDescription_PlatformName(PJRT_TopologyDescription_PlatformName_Args* args) {
  if (args != nullptr) {
    args->platform_name = "neuron";
    args->platform_name_size = 6;
  }
  return nullptr;
}

PJRT_Error* Dummy_TopologyDescription_PlatformVersion(PJRT_TopologyDescription_PlatformVersion_Args* args) {
  if (args != nullptr) {
    args->platform_version = "AWS Neuron PJRT 2.x (dummy)";
    args->platform_version_size = std::strlen(args->platform_version);
  }
  return nullptr;
}

PJRT_Error* Dummy_TopologyDescription_GetDeviceDescriptions(PJRT_TopologyDescription_GetDeviceDescriptions_Args* args) {
  if (args != nullptr) {
    args->descriptions = g_desc_ptrs;
    args->num_descriptions = 2;
  }
  return nullptr;
}

PJRT_Error* Dummy_TopologyDescription_Attributes(PJRT_TopologyDescription_Attributes_Args* args) {
  if (args != nullptr) {
    args->attributes = nullptr;
    args->num_attributes = 0;
  }
  return nullptr;
}

PJRT_Api g_api = [] {
  PJRT_Api api;
  std::memset(&api, 0, sizeof(api));
  api.struct_size = sizeof(api);
  api.extension_start = nullptr;
  api.pjrt_api_version = {sizeof(PJRT_Api_Version), nullptr, 0, 60};

  api.PJRT_Error_Destroy = &Dummy_Error_Destroy;
  api.PJRT_Error_Message = &Dummy_Error_Message;
  api.PJRT_Error_GetCode = &Dummy_Error_GetCode;

  api.PJRT_Plugin_Initialize = &Dummy_Plugin_Initialize;
  api.PJRT_Plugin_Attributes = &Dummy_Plugin_Attributes;

  api.PJRT_Client_Create = &Dummy_Client_Create;
  api.PJRT_Client_Destroy = &Dummy_Client_Destroy;
  api.PJRT_Client_PlatformName = &Dummy_Client_PlatformName;
  api.PJRT_Client_ProcessIndex = &Dummy_Client_ProcessIndex;
  api.PJRT_Client_PlatformVersion = &Dummy_Client_PlatformVersion;
  api.PJRT_Client_Devices = &Dummy_Client_Devices;
  api.PJRT_Client_AddressableDevices = &Dummy_Client_AddressableDevices;
  api.PJRT_Client_LookupDevice = &Dummy_Client_LookupDevice;
  api.PJRT_Client_LookupAddressableDevice = &Dummy_Client_LookupAddressableDevice;
  api.PJRT_Client_AddressableMemories = &Dummy_Client_AddressableMemories;

  api.PJRT_DeviceDescription_Id = &Dummy_DeviceDescription_Id;
  api.PJRT_DeviceDescription_ProcessIndex = &Dummy_DeviceDescription_ProcessIndex;
  api.PJRT_DeviceDescription_Attributes = &Dummy_DeviceDescription_Attributes;
  api.PJRT_DeviceDescription_Kind = &Dummy_DeviceDescription_Kind;
  api.PJRT_DeviceDescription_DebugString = &Dummy_DeviceDescription_DebugString;
  api.PJRT_DeviceDescription_ToString = &Dummy_DeviceDescription_ToString;

  api.PJRT_Device_GetDescription = &Dummy_Device_GetDescription;
  api.PJRT_Device_IsAddressable = &Dummy_Device_IsAddressable;
  api.PJRT_Device_LocalHardwareId = &Dummy_Device_LocalHardwareId;
  api.PJRT_Device_AddressableMemories = &Dummy_Device_AddressableMemories;
  api.PJRT_Device_DefaultMemory = &Dummy_Device_DefaultMemory;
  api.PJRT_Device_MemoryStats = &Dummy_Device_MemoryStats;

  api.PJRT_TopologyDescription_Destroy = &Dummy_TopologyDescription_Destroy;
  api.PJRT_TopologyDescription_PlatformName = &Dummy_TopologyDescription_PlatformName;
  api.PJRT_TopologyDescription_PlatformVersion = &Dummy_TopologyDescription_PlatformVersion;
  api.PJRT_TopologyDescription_GetDeviceDescriptions = &Dummy_TopologyDescription_GetDeviceDescriptions;
  api.PJRT_TopologyDescription_Attributes = &Dummy_TopologyDescription_Attributes;

  api.PJRT_Client_TopologyDescription = &Dummy_Client_TopologyDescription;
  api.PJRT_Error_ForEachPayload = &Dummy_Error_ForEachPayload;

  return api;
}();

}  // namespace

extern "C" {

const PJRT_Api* GetPjrtApi() {
  return &g_api;
}

}  // extern "C"
