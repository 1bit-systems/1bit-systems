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
#ifndef VULKAN_RT_H
#define VULKAN_RT_H

#include <vulkan/vulkan.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <fstream>
#include <vector>

namespace vkrt {

#define VKRT_BAIL(fmt, ...) do { fprintf(stderr, "vulkan_rt FATAL: " fmt "\n", ##__VA_ARGS__); exit(1); } while (0)
#define VKRT_CK(call) do { VkResult r_ = (call); if (r_ != VK_SUCCESS) { \
    fprintf(stderr, "vulkan_rt VK_ERR %s:%d: %s -> %d\n", __FILE__, __LINE__, #call, r_); exit(1); } } while (0)

inline std::vector<uint32_t> loadSpirv(const char* path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) VKRT_BAIL("Cannot open %s", path);
    size_t sz = (size_t)f.tellg();
    f.seekg(0);
    std::vector<uint32_t> code(sz / 4);
    f.read(reinterpret_cast<char*>(code.data()), (std::streamsize)sz);
    return code;
}

inline uint32_t findMemType(const VkPhysicalDeviceMemoryProperties& mp, uint32_t bits, VkMemoryPropertyFlags props) {
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++) {
        if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & props) == props) return i;
    }
    VKRT_BAIL("No suitable memory type");
    return 0;
}

struct GpuBuffer {
    VkBuffer buf = VK_NULL_HANDLE;
    VkDeviceMemory mem = VK_NULL_HANDLE;
    size_t size = 0;
    VkDevice dev = VK_NULL_HANDLE;

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
        if (mem) vkFreeMemory(dev, mem, nullptr);
        if (buf) vkDestroyBuffer(dev, buf, nullptr);
        mem = VK_NULL_HANDLE;
        buf = VK_NULL_HANDLE;
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

    void init() {
        VkApplicationInfo ai{VK_STRUCTURE_TYPE_APPLICATION_INFO};
        ai.pApplicationName = "1bit-vulkan-rt";
        ai.apiVersion = VK_API_VERSION_1_2;
        VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        ici.pApplicationInfo = &ai;
        VKRT_CK(vkCreateInstance(&ici, nullptr, &inst));

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

        float qp = 1.0f;
        VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        qci.queueCount = 1;
        qci.pQueuePriorities = &qp;
        VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        dci.queueCreateInfoCount = 1;
        dci.pQueueCreateInfos = &qci;
        VKRT_CK(vkCreateDevice(phys, &dci, nullptr, &dev));
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

        std::vector<VkDescriptorSetLayoutBinding> bindings((size_t)numBindings);
        for (int i = 0; i < numBindings; i++) {
            bindings[(size_t)i] = {(uint32_t)i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        }
        VkDescriptorSetLayoutCreateInfo dslci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        dslci.bindingCount = (uint32_t)numBindings;
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
    VKRT_CK(vkAllocateDescriptorSets(ctx.dev, &dai, &ds));

    std::vector<VkDescriptorBufferInfo> dbis((size_t)n);
    std::vector<VkWriteDescriptorSet> writes((size_t)n);
    for (int i = 0; i < n; i++) {
        dbis[(size_t)i] = {bufs[i]->buf, 0, VK_WHOLE_SIZE};
        writes[(size_t)i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, ds, (uint32_t)i, 0, 1,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &dbis[(size_t)i], nullptr};
    }
    vkUpdateDescriptorSets(ctx.dev, (uint32_t)n, writes.data(), 0, nullptr);
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
    VKRT_CK(vkAllocateCommandBuffers(ctx.dev, &cba, &cmd));

    VkCommandBufferBeginInfo cbb{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    cbb.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VKRT_CK(vkBeginCommandBuffer(cmd, &cbb));
    vkCmdResetQueryPool(cmd, ctx.queryPool, 0, 2);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p.pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p.layout, 0, 1, &ds, 0, nullptr);
    if (pcData && p.pcSize > 0) vkCmdPushConstants(cmd, p.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, p.pcSize, pcData);
    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, ctx.queryPool, 0);
    for (uint32_t i = 0; i < iterations; i++) {
        vkCmdDispatch(cmd, gx, gy, gz);
    }
    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, ctx.queryPool, 1);
    VKRT_CK(vkEndCommandBuffer(cmd));

    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    VKRT_CK(vkQueueSubmit(ctx.queue, 1, &si, VK_NULL_HANDLE));
    VKRT_CK(vkQueueWaitIdle(ctx.queue));

    uint64_t timestamps[2];
    VKRT_CK(vkGetQueryPoolResults(ctx.dev, ctx.queryPool, 0, 2, sizeof(timestamps), timestamps, sizeof(uint64_t),
                                   VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
    vkFreeCommandBuffers(ctx.dev, ctx.cmdPool, 1, &cmd);

    double elapsed_ns = (double)(timestamps[1] - timestamps[0]) * (double)ctx.timestampPeriodNs;
    return elapsed_ns / 1e6; // ms
}

} // namespace vkrt

#endif // VULKAN_RT_H
