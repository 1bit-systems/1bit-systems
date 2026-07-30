#pragma once
// vulkan_rt.h — minimal, dedicated Vulkan compute runtime for the 1bit
// engine's Vulkan backend. Adapted from the proven boilerplate in
// npu-sandbox/vulkan-gevm/phase2.cpp (already verified working on this
// box's RADV/Strix Halo driver) -- NOT linked from zinc, deliberately
// small (this only ever needs a handful of DMMV-shaped pipelines, not a
// general multi-backend engine).
//
// Buffers are host-visible + host-coherent only (no staging/device-local
// split). On this engine's target hardware (APUs with unified memory) that
// IS device-accessible memory, so this is both simpler and avoids an
// unnecessary copy -- consistent with the zero-copy direction already
// explored for this engine. If this ever needs to run well on a discrete
// GPU, add a staging-upload path then; don't build it speculatively now.
//
// External memory (dma-buf) support for NPU zero-copy (issue #1217):
// VkCtx::init() enables VK_KHR_external_memory_fd and the related instance
// extension when available.  GpuBuffer::create_from_dma_buf() imports a
// SharedBO dma-buf fd as Vulkan device memory so the GPU shader can read and
// write it without any CPU copy.
#ifndef VULKAN_RT_H
#define VULKAN_RT_H

#define VK_USE_PLATFORM_XLIB_KHR 0
#include <vulkan/vulkan.h>
#include <vulkan/vulkan_core.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <fstream>
#include <vector>

namespace vkrt {

// Log Vulkan errors instead of killing the process, so transient GPU issues
// (e.g. VK_ERROR_DEVICE_LOST) don't take down the server.
#define VKRT_BAIL(fmt, ...) do { fprintf(stderr, "vulkan_rt FATAL: " fmt "\n", ##__VA_ARGS__); return; } while (0)
#define VKRT_CK(call) do { VkResult r_ = (call); if (r_ != VK_SUCCESS) { \
    fprintf(stderr, "vulkan_rt VK_ERR %s:%d: %s -> %d\n", __FILE__, __LINE__, #call, r_); return; } } while (0)

inline std::vector<uint32_t> loadSpirv(const char* path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) { fprintf(stderr, "vulkan_rt FATAL: Cannot open %s\n", path); return {}; }
    size_t sz = static_cast<size_t>(f.tellg());
    f.seekg(0);
    std::vector<uint32_t> code(sz / 4);
    f.read(reinterpret_cast<char*>(code.data()), static_cast<std::streamsize>(sz));
    return code;
}

inline uint32_t findMemType(const VkPhysicalDeviceMemoryProperties& mp, uint32_t bits, VkMemoryPropertyFlags props) {
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++) {
        if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & props) == props) return i;
    }
    fprintf(stderr, "vulkan_rt FATAL: No suitable memory type\n");
    return 0;
}

struct GpuBuffer {
    VkBuffer buf = VK_NULL_HANDLE;
    VkDeviceMemory mem = VK_NULL_HANDLE;
    size_t size = 0;
    VkDevice dev = VK_NULL_HANDLE;
    bool imported_ = false;  // true when memory was imported (not owned by us)

    void create(VkDevice d, const VkPhysicalDeviceMemoryProperties& mp, size_t sz, VkBufferUsageFlags usage) {
        dev = d;
        size = sz;
        VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bi.size = sz;
        bi.usage = usage;
        VKRT_CK(vkCreateBuffer(dev, &bi, nullptr, &buf));
        VkMemoryRequirements mr;
        vkGetBufferMemoryRequirements(dev, buf, &mr);
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        ai.allocationSize = mr.size;
        ai.memoryTypeIndex = findMemType(mp, mr.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        VKRT_CK(vkAllocateMemory(dev, &ai, nullptr, &mem));
        VKRT_CK(vkBindBufferMemory(dev, buf, mem, 0));
    }

    // Import a Linux dma-buf fd (e.g. from SharedBO::dma_buf_fd()) as Vulkan
    // device memory — zero-copy NPU↔GPU path (issue #1217).
    // Requires VK_KHR_external_memory_fd on the device (enabled in VkCtx::init).
    // Returns false if the extension is unavailable or the import fails.
    bool create_from_dma_buf(VkDevice d,
                              const VkPhysicalDeviceMemoryProperties& mp,
                              size_t sz, int dma_fd,
                              VkBufferUsageFlags usage) {
        dev = d; size = sz;

        // Buffer with VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT declared.
        VkExternalMemoryBufferCreateInfo ext_bi{
            VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO};
        ext_bi.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;

        VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bi.pNext = &ext_bi;
        bi.size  = sz;
        bi.usage = usage;
        if (vkCreateBuffer(dev, &bi, nullptr, &buf) != VK_SUCCESS) {
            fprintf(stderr, "vulkan_rt: create_from_dma_buf: vkCreateBuffer failed\n");
            return false;
        }

        VkMemoryRequirements mr;
        vkGetBufferMemoryRequirements(dev, buf, &mr);

        // Import the dma-buf fd as Vulkan device memory.
        VkImportMemoryFdInfoKHR import_info{VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR};
        import_info.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
        import_info.fd         = dma_fd;

        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        ai.pNext          = &import_info;
        ai.allocationSize = mr.size;
        // SharedBO pages are HOST_ONLY coherent system RAM — pick the first
        // memory type that matches the buffer's requirements and is host-visible.
        ai.memoryTypeIndex = findMemType(mp, mr.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (vkAllocateMemory(dev, &ai, nullptr, &mem) != VK_SUCCESS) {
            fprintf(stderr, "vulkan_rt: create_from_dma_buf: vkAllocateMemory failed\n");
            vkDestroyBuffer(dev, buf, nullptr); buf = VK_NULL_HANDLE;
            return false;
        }
        if (vkBindBufferMemory(dev, buf, mem, 0) != VK_SUCCESS) {
            fprintf(stderr, "vulkan_rt: create_from_dma_buf: vkBindBufferMemory failed\n");
            vkFreeMemory(dev, mem, nullptr); mem = VK_NULL_HANDLE;
            vkDestroyBuffer(dev, buf, nullptr); buf = VK_NULL_HANDLE;
            return false;
        }
        imported_ = true;
        return true;
    }

    void upload(const void* data) {
        void* p;
        VKRT_CK(vkMapMemory(dev, mem, 0, size, 0, &p));
        memcpy(p, data, size);
        vkUnmapMemory(dev, mem);
    }
    void download(void* data) const {
        void* p;
        VKRT_CK(vkMapMemory(dev, mem, 0, size, 0, &p));
        memcpy(data, p, size);
        vkUnmapMemory(dev, mem);
    }
    void destroy() {
        if (mem) { vkFreeMemory(dev, mem, nullptr); mem = VK_NULL_HANDLE; }
        if (buf) { vkDestroyBuffer(dev, buf, nullptr); buf = VK_NULL_HANDLE; }
        imported_ = false;
    }
};

struct VkCtx {
    VkInstance inst = VK_NULL_HANDLE;
    VkPhysicalDevice phys = VK_NULL_HANDLE;
    VkDevice dev = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    VkCommandPool cmdPool = VK_NULL_HANDLE;
    VkPhysicalDeviceMemoryProperties memProps{};
    VkDescriptorPool dpool = VK_NULL_HANDLE;
    VkQueryPool queryPool = VK_NULL_HANDLE;
    float timestampPeriodNs = 1.0f;
    char deviceName[256] = {0};

    // Whether VK_KHR_external_memory_fd is available on the chosen device.
    bool ext_mem_fd = false;

    void init() {
        VkApplicationInfo ai{VK_STRUCTURE_TYPE_APPLICATION_INFO};
        ai.pApplicationName = "1bit-vulkan-rt";
        ai.apiVersion = VK_API_VERSION_1_2;

        // VK_KHR_external_memory_capabilities is an instance extension needed
        // before VK_KHR_external_memory_fd (device extension) can be used.
        const char* inst_exts[] = {
            "VK_KHR_external_memory_capabilities",
            "VK_KHR_get_physical_device_properties2",
        };
        VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        ici.pApplicationInfo        = &ai;
        ici.enabledExtensionCount   = 2;
        ici.ppEnabledExtensionNames = inst_exts;
        // If the instance extensions are unsupported, fall back to no extensions.
        if (vkCreateInstance(&ici, nullptr, &inst) != VK_SUCCESS) {
            ici.enabledExtensionCount   = 0;
            ici.ppEnabledExtensionNames = nullptr;
            VKRT_CK(vkCreateInstance(&ici, nullptr, &inst));
        }

        uint32_t nd = 0;
        VKRT_CK(vkEnumeratePhysicalDevices(inst, &nd, nullptr));
        if (nd == 0) VKRT_BAIL("No Vulkan-capable devices found");
        std::vector<VkPhysicalDevice> devs(nd);
        VKRT_CK(vkEnumeratePhysicalDevices(inst, &nd, devs.data()));

        for (auto d : devs) {
            VkPhysicalDeviceProperties dp;
            vkGetPhysicalDeviceProperties(d, &dp);
            if (dp.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ||
                dp.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
                phys = d;
                vkGetPhysicalDeviceMemoryProperties(d, &memProps);
                timestampPeriodNs = dp.limits.timestampPeriod;
                snprintf(deviceName, sizeof(deviceName), "%s", dp.deviceName);
                break;
            }
        }
        if (!phys) VKRT_BAIL("No integrated/discrete GPU found");

        // Probe for VK_KHR_external_memory_fd device extension (needed for
        // dma-buf import — issue #1217).
        uint32_t ext_count = 0;
        vkEnumerateDeviceExtensionProperties(phys, nullptr, &ext_count, nullptr);
        std::vector<VkExtensionProperties> avail_exts(ext_count);
        vkEnumerateDeviceExtensionProperties(phys, nullptr, &ext_count, avail_exts.data());
        for (auto& e : avail_exts) {
            if (strcmp(e.extensionName, "VK_KHR_external_memory_fd") == 0) {
                ext_mem_fd = true;
                break;
            }
        }

        float qp = 1.0f;
        VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        qci.queueCount = 1;
        qci.pQueuePriorities = &qp;
        VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        dci.queueCreateInfoCount = 1;
        dci.pQueueCreateInfos = &qci;

        const char* dev_exts[] = {
            "VK_KHR_external_memory",
            "VK_KHR_external_memory_fd",
        };
        if (ext_mem_fd) {
            dci.enabledExtensionCount   = 2;
            dci.ppEnabledExtensionNames = dev_exts;
        }
        if (vkCreateDevice(phys, &dci, nullptr, &dev) != VK_SUCCESS) {
            // Retry without the external-memory extensions if unavailable.
            dci.enabledExtensionCount   = 0;
            dci.ppEnabledExtensionNames = nullptr;
            ext_mem_fd = false;
            VKRT_CK(vkCreateDevice(phys, &dci, nullptr, &dev));
        }
        vkGetDeviceQueue(dev, 0, 0, &queue);

        VkCommandPoolCreateInfo cpci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        VKRT_CK(vkCreateCommandPool(dev, &cpci, nullptr, &cmdPool));

        VkDescriptorPoolSize dps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 64};
        VkDescriptorPoolCreateInfo dpc{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        dpc.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
        dpc.maxSets = 32;
        dpc.poolSizeCount = 1;
        dpc.pPoolSizes = &dps;
        VKRT_CK(vkCreateDescriptorPool(dev, &dpc, nullptr, &dpool));

        VkQueryPoolCreateInfo qpci{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        qpci.queryType = VK_QUERY_TYPE_TIMESTAMP;
        qpci.queryCount = 2;
        VKRT_CK(vkCreateQueryPool(dev, &qpci, nullptr, &queryPool));
    }

    void destroy() {
        if (queryPool) vkDestroyQueryPool(dev, queryPool, nullptr);
        if (dpool) vkDestroyDescriptorPool(dev, dpool, nullptr);
        if (cmdPool) vkDestroyCommandPool(dev, cmdPool, nullptr);
        if (dev) vkDestroyDevice(dev, nullptr);
        if (inst) vkDestroyInstance(inst, nullptr);
    }
};

struct Pipeline {
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkShaderModule shader = VK_NULL_HANDLE;
    VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
    uint32_t pcSize = 0;

    void create(VkCtx& ctx, const char* spvPath, int numBindings, uint32_t pcSizeIn) {
        pcSize = pcSizeIn;
        auto spv = loadSpirv(spvPath);
        VkShaderModuleCreateInfo sm{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        sm.codeSize = spv.size() * 4;
        sm.pCode = spv.data();
        VKRT_CK(vkCreateShaderModule(ctx.dev, &sm, nullptr, &shader));

        std::vector<VkDescriptorSetLayoutBinding> bindings(static_cast<size_t>(numBindings));
        for (int i = 0; i < numBindings; i++) {
            bindings[static_cast<size_t>(i)] = {static_cast<uint32_t>(i), VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        }
        VkDescriptorSetLayoutCreateInfo dslci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        dslci.bindingCount = static_cast<uint32_t>(numBindings);
        dslci.pBindings = bindings.data();
        VKRT_CK(vkCreateDescriptorSetLayout(ctx.dev, &dslci, nullptr, &dsl));

        VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, pcSize};
        VkPipelineLayoutCreateInfo pl{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pl.setLayoutCount = 1;
        pl.pSetLayouts = &dsl;
        pl.pushConstantRangeCount = pcSize > 0 ? 1u : 0u;
        pl.pPushConstantRanges = pcSize > 0 ? &pcr : nullptr;
        VKRT_CK(vkCreatePipelineLayout(ctx.dev, &pl, nullptr, &layout));

        VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stage.module = shader;
        stage.pName = "main";
        VkComputePipelineCreateInfo cp{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        cp.stage = stage;
        cp.layout = layout;
        cp.basePipelineIndex = -1;
        VKRT_CK(vkCreateComputePipelines(ctx.dev, VK_NULL_HANDLE, 1, &cp, nullptr, &pipeline));
    }

    void destroy(VkDevice dev) {
        if (pipeline) vkDestroyPipeline(dev, pipeline, nullptr);
        if (layout) vkDestroyPipelineLayout(dev, layout, nullptr);
        if (dsl) vkDestroyDescriptorSetLayout(dev, dsl, nullptr);
        if (shader) vkDestroyShaderModule(dev, shader, nullptr);
    }
};

inline VkDescriptorSet createDescriptorSet(VkCtx& ctx, Pipeline& p, GpuBuffer** bufs, int n) {
    VkDescriptorSet ds;
    VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dai.descriptorPool = ctx.dpool;
    dai.descriptorSetCount = 1;
    dai.pSetLayouts = &p.dsl;
    { VkResult r_ = vkAllocateDescriptorSets(ctx.dev, &dai, &ds); if (r_ != VK_SUCCESS) { fprintf(stderr, "vulkan_rt VK_ERR %s:%d: %s -> %d\n", __FILE__, __LINE__, "vkAllocateDescriptorSets", r_); return VK_NULL_HANDLE; } }

    std::vector<VkDescriptorBufferInfo> dbis(static_cast<size_t>(n));
    std::vector<VkWriteDescriptorSet> writes(static_cast<size_t>(n));
    for (int i = 0; i < n; i++) {
        dbis[static_cast<size_t>(i)] = {bufs[i]->buf, 0, VK_WHOLE_SIZE};
        writes[static_cast<size_t>(i)] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, ds, static_cast<uint32_t>(i), 0, 1,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &dbis[static_cast<size_t>(i)], nullptr};
    }
    vkUpdateDescriptorSets(ctx.dev, static_cast<uint32_t>(n), writes.data(), 0, nullptr);
    return ds;
}

// Single dispatch, blocking (submit + wait idle). Used for correctness checks.
inline void dispatchOnce(VkCtx& ctx, Pipeline& p, VkDescriptorSet ds, uint32_t gx, uint32_t gy, uint32_t gz,
                          const void* pcData) {
    VkCommandBuffer cmd;
    VkCommandBufferAllocateInfo cba{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cba.commandPool = ctx.cmdPool;
    cba.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cba.commandBufferCount = 1;
    VKRT_CK(vkAllocateCommandBuffers(ctx.dev, &cba, &cmd));

    VkCommandBufferBeginInfo cbb{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    cbb.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VKRT_CK(vkBeginCommandBuffer(cmd, &cbb));
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p.pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p.layout, 0, 1, &ds, 0, nullptr);
    if (pcData && p.pcSize > 0) vkCmdPushConstants(cmd, p.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, p.pcSize, pcData);
    vkCmdDispatch(cmd, gx, gy, gz);
    VKRT_CK(vkEndCommandBuffer(cmd));

    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    VKRT_CK(vkQueueSubmit(ctx.queue, 1, &si, VK_NULL_HANDLE));
    VKRT_CK(vkQueueWaitIdle(ctx.queue));
    vkFreeCommandBuffers(ctx.dev, ctx.cmdPool, 1, &cmd);
}

// Repeated back-to-back dispatch of the same pipeline/descriptor set/push
// constants, timed with a GPU timestamp query pair around the whole batch.
// Used for steady-state throughput measurement (warmup pass has timing
// discarded by the caller; measured pass reads elapsedMs()).
inline double dispatchRepeatedTimed(VkCtx& ctx, Pipeline& p, VkDescriptorSet ds, uint32_t gx, uint32_t gy, uint32_t gz,
                                     const void* pcData, uint32_t iterations) {
    VkCommandBuffer cmd;
    VkCommandBufferAllocateInfo cba{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cba.commandPool = ctx.cmdPool;
    cba.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cba.commandBufferCount = 1;
    { VkResult r_ = vkAllocateCommandBuffers(ctx.dev, &cba, &cmd); if (r_ != VK_SUCCESS) { fprintf(stderr, "vulkan_rt VK_ERR %s:%d: %s -> %d\n", __FILE__, __LINE__, "vkAllocateCommandBuffers", r_); return 0.0; } }

    VkCommandBufferBeginInfo cbb{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    cbb.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    { VkResult r_ = vkBeginCommandBuffer(cmd, &cbb); if (r_ != VK_SUCCESS) { fprintf(stderr, "vulkan_rt VK_ERR %s:%d: %s -> %d\n", __FILE__, __LINE__, "vkBeginCommandBuffer", r_); return 0.0; } }
    vkCmdResetQueryPool(cmd, ctx.queryPool, 0, 2);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p.pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p.layout, 0, 1, &ds, 0, nullptr);
    if (pcData && p.pcSize > 0) vkCmdPushConstants(cmd, p.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, p.pcSize, pcData);
    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, ctx.queryPool, 0);
    for (uint32_t i = 0; i < iterations; i++) {
        vkCmdDispatch(cmd, gx, gy, gz);
    }
    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, ctx.queryPool, 1);
    { VkResult r_ = vkEndCommandBuffer(cmd); if (r_ != VK_SUCCESS) { fprintf(stderr, "vulkan_rt VK_ERR %s:%d: %s -> %d\n", __FILE__, __LINE__, "vkEndCommandBuffer", r_); return 0.0; } }

    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    { VkResult r_ = vkQueueSubmit(ctx.queue, 1, &si, VK_NULL_HANDLE); if (r_ != VK_SUCCESS) { fprintf(stderr, "vulkan_rt VK_ERR %s:%d: %s -> %d\n", __FILE__, __LINE__, "vkQueueSubmit", r_); return 0.0; } }
    { VkResult r_ = vkQueueWaitIdle(ctx.queue); if (r_ != VK_SUCCESS) { fprintf(stderr, "vulkan_rt VK_ERR %s:%d: %s -> %d\n", __FILE__, __LINE__, "vkQueueWaitIdle", r_); return 0.0; } }

    uint64_t timestamps[2];
    { VkResult r_ = vkGetQueryPoolResults(ctx.dev, ctx.queryPool, 0, 2, sizeof(timestamps), timestamps, sizeof(uint64_t),
                                   VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT); if (r_ != VK_SUCCESS) { fprintf(stderr, "vulkan_rt VK_ERR %s:%d: %s -> %d\n", __FILE__, __LINE__, "vkGetQueryPoolResults", r_); return 0.0; } }
    vkFreeCommandBuffers(ctx.dev, ctx.cmdPool, 1, &cmd);

    double elapsed_ns = static_cast<double>(timestamps[1] - timestamps[0]) * static_cast<double>(ctx.timestampPeriodNs);
    return elapsed_ns / 1e6; // ms
}

} // namespace vkrt

#endif // VULKAN_RT_H
