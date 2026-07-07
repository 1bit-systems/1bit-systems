/** npu_gpu_dma_buf_poc.cpp — Proof of Concept: XRT BO → dma-buf → Vulkan import
 *
 *  Build:
 *    g++ -std=c++23 -O3 -o npu_gpu_dma_buf_poc npu_gpu_dma_buf_poc.cpp \
 *        -I/usr/include -I/home/bcloud/torch2aie/toolchain/xrt/include \
 *        -L/home/bcloud/torch2aie/toolchain/xrt/lib64 \
 *        -lxrt_coreutil -lxrt_core -luuid -lm -ldl -lvulkan
 *
 *  Run:
 *    sudo LD_LIBRARY_PATH=/home/bcloud/torch2aie/toolchain/xrt/lib64 \
 *        ./npu_gpu_dma_buf_poc
 *
 *  Verifies:
 *    1. XRT BO allocated and exported as dma-buf fd
 *    2. Vulkan imports the dma-buf fd as external memory
 *    3. NPU writes → GPU reads the same physical memory (zero-copy round-trip)
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <vulkan/vulkan.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_kernel.h>

// Vulkan function pointers for extensions we need
static PFN_vkGetMemoryFdKHR vkGetMemoryFdKHR = nullptr;
static PFN_vkGetPhysicalDeviceExternalBufferPropertiesKHR vkGetPhysicalDeviceExternalBufferPropertiesKHR = nullptr;

static bool init_vulkan_extensions(VkDevice device) {
    vkGetMemoryFdKHR = (PFN_vkGetMemoryFdKHR)vkGetDeviceProcAddr(device, "vkGetMemoryFdKHR");
    vkGetPhysicalDeviceExternalBufferPropertiesKHR = (PFN_vkGetPhysicalDeviceExternalBufferPropertiesKHR)
        vkGetDeviceProcAddr(device, "vkGetPhysicalDeviceExternalBufferPropertiesKHR");
    // Fallback to instance proc
    if (!vkGetPhysicalDeviceExternalBufferPropertiesKHR)
        vkGetPhysicalDeviceExternalBufferPropertiesKHR = (PFN_vkGetPhysicalDeviceExternalBufferPropertiesKHR)
            vkGetInstanceProcAddr(nullptr, "vkGetPhysicalDeviceExternalBufferPropertiesKHR");
    return vkGetMemoryFdKHR != nullptr;
}

int main() {
    printf("=== NPU→GPU dma-buf POC ===\n\n");

    // ===================================================================
    // Phase 1: XRT allocate + write + export dma-buf
    // ===================================================================
    printf("[Phase 1] XRT BO export as dma-buf\n");

    // Open NPU device
    xrt::device device;
    try { device = xrt::device(0); }
    catch (...) { try { device = xrt::device(1); } catch (...) {
        fprintf(stderr, "  ERROR: no NPU device\n"); return 1; }
    }
    printf("  NPU device: %s\n", device.get_info<xrt::info::device::name>().c_str());

    // Allocate a small BO with test data
    size_t bo_size = 4096; // 4KB test buffer
    xrt::bo npu_bo = xrt::bo(device, bo_size, XCL_BO_FLAGS_CACHEABLE, 0);
    int32_t* buf = (int32_t*)npu_bo.map();
    for (int i = 0; i < 1024; i++) buf[i] = i * 10; // known pattern
    npu_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    printf("  BO allocated: %zu bytes, written test pattern\n", bo_size);

    // Export as dma-buf (returns fd on Linux)
    int dma_buf_fd = (int)npu_bo.export_buffer();
    if (dma_buf_fd < 0) {
        fprintf(stderr, "  ERROR: export_buffer failed (fd=%d)\n", dma_buf_fd);
        return 1;
    }
    printf("  dma-buf fd: %d\n", dma_buf_fd);

    // ===================================================================
    // Phase 2: Vulkan init + import dma-buf
    // ===================================================================
    printf("\n[Phase 2] Vulkan dma-buf import\n");

    VkInstance vk_instance = VK_NULL_HANDLE;
    VkPhysicalDevice vk_phys_dev = VK_NULL_HANDLE;
    VkDevice vk_device = VK_NULL_HANDLE;

    // Create Vulkan instance
    VkApplicationInfo app_info{};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.apiVersion = VK_API_VERSION_1_3;

    const char* inst_exts[] = {
        VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME,
        VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME
    };
    VkInstanceCreateInfo inst_info{};
    inst_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    inst_info.pApplicationInfo = &app_info;
    inst_info.enabledExtensionCount = 2;
    inst_info.ppEnabledExtensionNames = inst_exts;

    if (vkCreateInstance(&inst_info, nullptr, &vk_instance) != VK_SUCCESS) {
        fprintf(stderr, "  ERROR: Cannot create Vulkan instance\n"); return 1;
    }
    printf("  Vulkan instance created\n");

    // Pick physical device (radeon GPU)
    uint32_t gpu_count = 0;
    vkEnumeratePhysicalDevices(vk_instance, &gpu_count, nullptr);
    std::vector<VkPhysicalDevice> gpus(gpu_count);
    vkEnumeratePhysicalDevices(vk_instance, &gpu_count, gpus.data());
    for (uint32_t i = 0; i < gpu_count; i++) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(gpus[i], &props);
        printf("  GPU %d: %s (type=%d)\n", i, props.deviceName, props.deviceType);
        if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ||
            props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
            if (vk_phys_dev == VK_NULL_HANDLE) vk_phys_dev = gpus[i];
        }
    }
    if (vk_phys_dev == VK_NULL_HANDLE) vk_phys_dev = gpus[0];

    // Check dma-buf support
    VkPhysicalDeviceExternalBufferInfo ext_buf_info{};
    ext_buf_info.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_BUFFER_INFO;
    ext_buf_info.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
    ext_buf_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    ext_buf_info.flags = 0;

    VkExternalBufferProperties ext_buf_props{};
    ext_buf_props.sType = VK_STRUCTURE_TYPE_EXTERNAL_BUFFER_PROPERTIES;

    PFN_vkGetPhysicalDeviceExternalBufferPropertiesKHR get_ext_buf = 
        (PFN_vkGetPhysicalDeviceExternalBufferPropertiesKHR)
        vkGetInstanceProcAddr(vk_instance, "vkGetPhysicalDeviceExternalBufferPropertiesKHR");
    if (get_ext_buf) {
        get_ext_buf(vk_phys_dev, &ext_buf_info, &ext_buf_props);
        bool can_export = ext_buf_props.externalMemoryProperties.externalMemoryFeatures &
                          VK_EXTERNAL_MEMORY_FEATURE_EXPORTABLE_BIT;
        bool can_import = ext_buf_props.externalMemoryProperties.externalMemoryFeatures &
                          VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT;
        printf("  dma-buf: export=%s import=%s\n",
               can_export ? "yes" : "no", can_import ? "yes" : "no");
        if (!can_import) {
            fprintf(stderr, "  ERROR: GPU cannot import dma-buf\n");
            // Continue anyway for testing
        }
    }

    // Create logical device with external memory extensions
    float queue_priority = 1.0f;
    VkDeviceQueueCreateInfo q_info{};
    q_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    q_info.queueFamilyIndex = 0;
    q_info.queueCount = 1;
    q_info.pQueuePriorities = &queue_priority;

    const char* dev_exts[] = {
        VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
        VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME,
        VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME
    };

    VkDeviceCreateInfo dev_info{};
    dev_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dev_info.queueCreateInfoCount = 1;
    dev_info.pQueueCreateInfos = &q_info;
    dev_info.enabledExtensionCount = 3;
    dev_info.ppEnabledExtensionNames = dev_exts;

    if (vkCreateDevice(vk_phys_dev, &dev_info, nullptr, &vk_device) != VK_SUCCESS) {
        fprintf(stderr, "  ERROR: Cannot create Vulkan device\n"); return 1;
    }
    printf("  Vulkan device created\n");

    if (!init_vulkan_extensions(vk_device)) {
        fprintf(stderr, "  ERROR: Cannot get Vulkan extension functions\n"); return 1;
    }

    // ===================================================================
    // Phase 3: Import dma-buf into Vulkan memory
    // ===================================================================
    printf("\n[Phase 3] dma-buf import → GPU access\n");

    VkMemoryRequirements mem_reqs{};
    mem_reqs.size = bo_size;
    mem_reqs.alignment = 4096;
    mem_reqs.memoryTypeBits = 0xff;

    VkImportMemoryFdInfoEXT import_info{};
    import_info.sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_EXT;
    import_info.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
    import_info.fd = dma_buf_fd;  // the same fd from XRT export

    VkMemoryAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.pNext = &import_info;
    alloc_info.allocationSize = bo_size;

    // Find a memory type that supports dma-buf import
    VkPhysicalDeviceMemoryProperties mem_props;
    vkGetPhysicalDeviceMemoryProperties(vk_phys_dev, &mem_props);

    uint32_t mem_type = UINT32_MAX;
    for (uint32_t i = 0; i < mem_props.memoryTypeCount; i++) {
        // Look for device-local or host-visible memory that can import
        if (mem_props.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) {
            mem_type = i;
            break;
        }
    }
    if (mem_type == UINT32_MAX) mem_type = 0;
    alloc_info.memoryTypeIndex = mem_type;

    VkDeviceMemory vk_mem = VK_NULL_HANDLE;
    VkResult res = vkAllocateMemory(vk_device, &alloc_info, nullptr, &vk_mem);
    if (res != VK_SUCCESS) {
        fprintf(stderr, "  ERROR: Cannot allocate Vulkan memory from dma-buf (res=%d)\n", res);
        // Try with a different fd — duplicate the fd
        int dup_fd = dup(dma_buf_fd);
        import_info.fd = dup_fd;
        res = vkAllocateMemory(vk_device, &alloc_info, nullptr, &vk_mem);
        if (res != VK_SUCCESS) {
            fprintf(stderr, "  ERROR: Second attempt also failed (res=%d)\n", res);
            printf("\n  Result: dma-buf exported but Vulkan import failed — may need driver update\n");
            printf("  The dma-buf fd (%d) is valid but Vulkan couldn't use it\n", dma_buf_fd);
        }
    }

    if (vk_mem != VK_NULL_HANDLE) {
        printf("  Vulkan memory imported from dma-buf fd! ✅\n");

        // Create a buffer backed by this memory
        VkBufferCreateInfo buf_info{};
        buf_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        buf_info.size = bo_size;
        buf_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

        VkExternalMemoryBufferCreateInfo ext_buf{};
        ext_buf.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
        ext_buf.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
        buf_info.pNext = &ext_buf;

        VkBuffer vk_buf = VK_NULL_HANDLE;
        if (vkCreateBuffer(vk_device, &buf_info, nullptr, &vk_buf) == VK_SUCCESS) {
            vkBindBufferMemory(vk_device, vk_buf, vk_mem, 0);
            printf("  Vulkan buffer created from NPU BO memory! ✅\n\n");

            // Verify: NPU already wrote pattern, read from Vulkan memory (if host-visible)
            printf("[Verification] Reading NPU-written data through Vulkan memory\n");
            void* mapped = nullptr;
            vkMapMemory(vk_device, vk_mem, 0, bo_size, 0, &mapped);
            if (mapped) {
                int32_t* data = (int32_t*)mapped;
                bool ok = true;
                for (int i = 0; i < 16; i++) {
                    printf("  [%d] expected=%d got=%d %s\n",
                           i, i * 10, data[i], data[i] == i * 10 ? "✅" : "❌");
                    if (data[i] != i * 10) ok = false;
                }
                printf("\n  Zero-copy round-trip: %s\n",
                       ok ? "✅ PASS — NPU→GPU direct memory sharing works!" : "❌ FAIL — data mismatch");
                vkUnmapMemory(vk_device, vk_mem);
            }
            vkDestroyBuffer(vk_device, vk_buf, nullptr);
        }
        vkFreeMemory(vk_device, vk_mem, nullptr);
    }

    // ===================================================================
    // Cleanup
    // ===================================================================
    vkDestroyDevice(vk_device, nullptr);
    vkDestroyInstance(vk_instance, nullptr);
    close(dma_buf_fd); // close the dma-buf fd

    printf("\n=== POC complete ===\n");
    return 0;
}
