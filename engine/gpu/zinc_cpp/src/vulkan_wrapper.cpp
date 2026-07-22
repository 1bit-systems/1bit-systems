/// C++ Vulkan wrappers — implementation.
/// Fixed: #760 proper memory type selection, #764 fp16/int8 feature check
#include "vulkan_wrapper.h"
#include <set>
#include <algorithm>
#include <cstdio>

// ═══════════════════════════════════════════════════════════════════
//  ShaderCache
// ═══════════════════════════════════════════════════════════════════

ShaderCache::~ShaderCache() {
    for (auto& [name, mod] : cache_) {
        vkDestroyShaderModule(device_, mod, nullptr);
    }
}

VkShaderModule ShaderCache::load(const std::string& name) {
    auto it = cache_.find(name);
    if (it != cache_.end()) return it->second;
    return load_path(name + ".spv");
}

VkShaderModule ShaderCache::load_path(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) throw std::runtime_error("Cannot open shader: " + path);
    
    size_t size = (size_t)file.tellg();
    if (size % 4 != 0) throw std::runtime_error("SPIR-V file size not aligned to 4 bytes: " + path);
    if (size == 0) throw std::runtime_error("Empty SPIR-V file: " + path);
    
    file.seekg(0);
    std::vector<uint32_t> code(size / 4);
    file.read((char*)code.data(), size);

    VkShaderModuleCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    ci.codeSize = size;
    ci.pCode = code.data();

    VkShaderModule mod;
    VK_CHECK(vkCreateShaderModule(device_, &ci, nullptr, &mod));
    cache_[path] = mod;
    return mod;
}

// ═══════════════════════════════════════════════════════════════════
//  ComputePipelineCache
// ═══════════════════════════════════════════════════════════════════

ComputePipelineCache::~ComputePipelineCache() {
    for (auto& [name, pipe] : cache_) {
        vkDestroyPipeline(device_, pipe, nullptr);
    }
    if (pipeline_layout_) vkDestroyPipelineLayout(device_, pipeline_layout_, nullptr);
    if (desc_set_layout_) vkDestroyDescriptorSetLayout(device_, desc_set_layout_, nullptr);
}

void ComputePipelineCache::ensure_layout() {
    if (layout_inited_) return;

    VkDescriptorSetLayoutBinding bindings[3] = {};
    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings[0].descriptorCount = 1;
    bindings[1] = bindings[0]; bindings[1].binding = 1;
    bindings[2] = bindings[0]; bindings[2].binding = 2;

    VkDescriptorSetLayoutCreateInfo dci{};
    dci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dci.bindingCount = 3;
    dci.pBindings = bindings;
    VK_CHECK(vkCreateDescriptorSetLayout(device_, &dci, nullptr, &desc_set_layout_));

    VkPushConstantRange pcr{};
    pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcr.offset = 0;
    pcr.size = 128;

    VkPipelineLayoutCreateInfo plci{};
    plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &desc_set_layout_;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges = &pcr;
    VK_CHECK(vkCreatePipelineLayout(device_, &plci, nullptr, &pipeline_layout_));

    layout_inited_ = true;
}

VkPipeline ComputePipelineCache::get(const std::string& shader_name,
                                      const std::vector<VkSpecializationMapEntry>& spec_constants,
                                      const void* spec_data, size_t spec_data_size) {
    auto it = cache_.find(shader_name);
    if (it != cache_.end()) return it->second;

    ensure_layout();
    VkShaderModule mod = shaders_.load(shader_name);

    VkSpecializationInfo spec_info{};
    if (!spec_constants.empty()) {
        spec_info.mapEntryCount = (uint32_t)spec_constants.size();
        spec_info.pMapEntries = spec_constants.data();
        spec_info.dataSize = spec_data_size;
        spec_info.pData = spec_data;
    }

    VkPipelineShaderStageCreateInfo sci{};
    sci.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    sci.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    sci.module = mod;
    sci.pName = "main";
    if (spec_data_size > 0) sci.pSpecializationInfo = &spec_info;

    VkComputePipelineCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    ci.stage = sci;
    ci.layout = pipeline_layout_;

    VkPipeline pipe;
    VK_CHECK(vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &ci, nullptr, &pipe));
    cache_[shader_name] = pipe;
    return pipe;
}

// ═══════════════════════════════════════════════════════════════════
//  Memory type selection helper
// ═══════════════════════════════════════════════════════════════════

static uint32_t find_memory_type(VkPhysicalDevice phys_dev, uint32_t type_filter,
                                  VkMemoryPropertyFlags props) {
    VkPhysicalDeviceMemoryProperties mem_props;
    vkGetPhysicalDeviceMemoryProperties(phys_dev, &mem_props);
    for (uint32_t i = 0; i < mem_props.memoryTypeCount; i++) {
        if ((type_filter & (1u << i)) &&
            (mem_props.memoryTypes[i].propertyFlags & props) == props) {
            return i;
        }
    }
    throw std::runtime_error("Failed to find suitable memory type");
}

// ═══════════════════════════════════════════════════════════════════
//  GpuBuffer (with proper memory allocation)
// ═══════════════════════════════════════════════════════════════════

static VkPhysicalDevice g_phys_dev_for_buffer = VK_NULL_HANDLE; // set by ZincEngine

GpuBuffer::GpuBuffer(VkDevice device, VkDeviceSize size, VkBufferUsageFlags usage,
                      VkMemoryPropertyFlags mem_flags, VkSharingMode sharing)
    : device_(device), size_(size) {
    VkBufferCreateInfo bci{};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size = size;
    bci.usage = usage;
    bci.sharingMode = sharing;
    VK_CHECK(vkCreateBuffer(device_, &bci, nullptr, &buffer_));

    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(device_, buffer_, &req);

    uint32_t mem_type = find_memory_type(g_phys_dev_for_buffer, req.memoryTypeBits, mem_flags);

    VkMemoryAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = mem_type;
    VK_CHECK(vkAllocateMemory(device_, &ai, nullptr, &memory_));
    VK_CHECK(vkBindBufferMemory(device_, buffer_, memory_, 0));
}

GpuBuffer::GpuBuffer(GpuBuffer&& other) noexcept
    : device_(other.device_), buffer_(other.buffer_), memory_(other.memory_),
      size_(other.size_), mapped_(other.mapped_) {
    other.buffer_ = VK_NULL_HANDLE;
    other.memory_ = VK_NULL_HANDLE;
    other.mapped_ = nullptr;
}

GpuBuffer& GpuBuffer::operator=(GpuBuffer&& other) noexcept {
    if (this != &other) {
        destroy();
        device_ = other.device_;
        buffer_ = other.buffer_;
        memory_ = other.memory_;
        size_ = other.size_;
        mapped_ = other.mapped_;
        other.buffer_ = VK_NULL_HANDLE;
        other.memory_ = VK_NULL_HANDLE;
        other.mapped_ = nullptr;
    }
    return *this;
}

void* GpuBuffer::map(VkDeviceSize offset, VkDeviceSize size) {
    if (mapped_) return mapped_;
    VK_CHECK(vkMapMemory(device_, memory_, offset, size, 0, &mapped_));
    return mapped_;
}

void GpuBuffer::unmap() {
    if (mapped_) { vkUnmapMemory(device_, memory_); mapped_ = nullptr; }
}

void GpuBuffer::flush(VkDeviceSize offset, VkDeviceSize size) {
    VkMappedMemoryRange range{};
    range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
    range.memory = memory_;
    range.offset = offset;
    range.size = size;
    vkFlushMappedMemoryRanges(device_, 1, &range);
}

void GpuBuffer::invalidate(VkDeviceSize offset, VkDeviceSize size) {
    VkMappedMemoryRange range{};
    range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
    range.memory = memory_;
    range.offset = offset;
    range.size = size;
    vkInvalidateMappedMemoryRanges(device_, 1, &range);
}

void GpuBuffer::destroy() {
    if (mapped_) vkUnmapMemory(device_, memory_);
    if (buffer_) vkDestroyBuffer(device_, buffer_, nullptr);
    if (memory_) vkFreeMemory(device_, memory_, nullptr);
    buffer_ = VK_NULL_HANDLE;
    memory_ = VK_NULL_HANDLE;
    mapped_ = nullptr;
}

// ═══════════════════════════════════════════════════════════════════
//  CommandPool
// ═══════════════════════════════════════════════════════════════════

CommandPool::CommandPool(VkDevice device, uint32_t queue_family) : device_(device) {
    VkCommandPoolCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    ci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    ci.queueFamilyIndex = queue_family;
    VK_CHECK(vkCreateCommandPool(device_, &ci, nullptr, &pool_));
}

CommandPool::~CommandPool() {
    if (pool_) vkDestroyCommandPool(device_, pool_, nullptr);
}

VkCommandBuffer CommandPool::begin_once() {
    VkCommandBufferAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ai.commandPool = pool_;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    VkCommandBuffer cmd;
    VK_CHECK(vkAllocateCommandBuffers(device_, &ai, &cmd));
    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(cmd, &bi));
    return cmd;
}

void CommandPool::submit(VkCommandBuffer cmd, VkQueue queue, VkFence fence) {
    VK_CHECK(vkEndCommandBuffer(cmd));
    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    VK_CHECK(vkQueueSubmit(queue, 1, &si, fence));
}

void CommandPool::submit_and_wait(VkCommandBuffer cmd, VkQueue queue) {
    VkFence fence;
    VkFenceCreateInfo fci{};
    fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VK_CHECK(vkCreateFence(device_, &fci, nullptr, &fence));
    submit(cmd, queue, fence);
    VK_CHECK(vkWaitForFences(device_, 1, &fence, VK_TRUE, UINT64_MAX));
    vkDestroyFence(device_, fence, nullptr);
    vkFreeCommandBuffers(device_, pool_, 1, &cmd);
}

// ═══════════════════════════════════════════════════════════════════
//  ZincEngine
// ═══════════════════════════════════════════════════════════════════

void ZincEngine::init(const std::string& shader_dir, int device_idx) {
    shader_dir_ = shader_dir;

    // Create Vulkan instance
    VkApplicationInfo app{};
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "ZINC C++";
    app.applicationVersion = VK_MAKE_API_VERSION(0, 1, 0, 0);
    app.apiVersion = VK_API_VERSION_1_3;

    const char* extensions[] = {
        VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME
    };

    VkInstanceCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ici.pApplicationInfo = &app;
    ici.enabledExtensionCount = 1;
    ici.ppEnabledExtensionNames = extensions;
    VK_CHECK(vkCreateInstance(&ici, nullptr, &instance_));

    // Enumerate physical devices
    uint32_t count = 0;
    vkEnumeratePhysicalDevices(instance_, &count, nullptr);
    if (count == 0) throw std::runtime_error("No Vulkan GPU found");
    std::vector<VkPhysicalDevice> devices(count);
    vkEnumeratePhysicalDevices(instance_, &count, devices.data());

    // Select device by index or use first
    int idx = (device_idx >= 0 && (uint32_t)device_idx < count) ? device_idx : 0;
    phys_device_ = devices[idx];
    g_phys_dev_for_buffer = phys_device_;

    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(phys_device_, &props);
    printf("ZINC: GPU = %s\n", props.deviceName);

    // Check fp16/int8 support
    VkPhysicalDeviceShaderFloat16Int8Features f16i8{};
    f16i8.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES;
    
    VkPhysicalDeviceFeatures2 feat2{};
    feat2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    feat2.pNext = &f16i8;
    vkGetPhysicalDeviceFeatures2(phys_device_, &feat2);

    if (!f16i8.shaderFloat16) printf("ZINC: WARNING — GPU lacks fp16 compute, falling back to fp32\n");
    if (!f16i8.shaderInt8) printf("ZINC: WARNING — GPU lacks int8 compute\n");

    // Find compute queue family
    uint32_t qcount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(phys_device_, &qcount, nullptr);
    std::vector<VkQueueFamilyProperties> qprops(qcount);
    vkGetPhysicalDeviceQueueFamilyProperties(phys_device_, &qcount, qprops.data());

    for (uint32_t i = 0; i < qcount; i++) {
        if (qprops[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            queue_family_ = i;
            break;
        }
    }
    if (queue_family_ == UINT32_MAX) throw std::runtime_error("No compute queue");

    // Create logical device
    float priority = 1.0f;
    VkDeviceQueueCreateInfo qci{};
    qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qci.queueFamilyIndex = queue_family_;
    qci.queueCount = 1;
    qci.pQueuePriorities = &priority;

    const char* dev_exts[] = {
        VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME
    };

    VkDeviceCreateInfo dci{};
    dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = 1;
    dci.ppEnabledExtensionNames = dev_exts;
    // Only enable fp16/int8 if supported
    if (f16i8.shaderFloat16 || f16i8.shaderInt8) {
        f16i8.shaderFloat16 = f16i8.shaderFloat16 ? VK_TRUE : VK_FALSE;
        f16i8.shaderInt8 = f16i8.shaderInt8 ? VK_TRUE : VK_FALSE;
        dci.pNext = &f16i8;
    }
    VK_CHECK(vkCreateDevice(phys_device_, &dci, nullptr, &device_));

    vkGetDeviceQueue(device_, queue_family_, 0, &queue_);

    cmd_pool_ = std::make_unique<CommandPool>(device_, queue_family_);
    shader_cache_ = std::make_unique<ShaderCache>(device_);
    pipeline_cache_ = std::make_unique<ComputePipelineCache>(device_);

    printf("ZINC: Vulkan initialized\n");
}

bool ZincEngine::load_model(const std::string& gguf_path) {
    printf("ZINC: Loading model from %s\n", gguf_path.c_str());
    // TODO: full GGUF parsing and weight upload
    return true;
}

void ZincEngine::reset() { model_.current_pos = 0; }

int ZincEngine::generate(int token_id) {
    (void)token_id;
    return -1; // placeholder — use InferenceEngine via backend_zinc.cpp
}

void ZincEngine::destroy() {
    pipeline_cache_.reset();
    shader_cache_.reset();
    cmd_pool_.reset();
    if (device_) vkDestroyDevice(device_, nullptr);
    if (instance_) vkDestroyInstance(instance_, nullptr);
    device_ = VK_NULL_HANDLE;
    instance_ = VK_NULL_HANDLE;
}
