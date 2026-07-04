//! Vulkan handle types — thin wrappers over VkHandle.
//! Matches the pattern in zinc/src/vulkan/vk.zig.

const c = @cImport({
    @cInclude("vulkan/vulkan.h");
});

pub const VkInstance = c.VkInstance;
pub const VkPhysicalDevice = c.VkPhysicalDevice;
pub const VkDevice = c.VkDevice;
pub const VkQueue = c.VkQueue;
pub const VkCommandPool = c.VkCommandPool;
pub const VkCommandBuffer = c.VkCommandBuffer;
pub const VkDescriptorPool = c.VkDescriptorPool;
pub const VkDescriptorSetLayout = c.VkDescriptorSetLayout;
pub const VkDescriptorSet = c.VkDescriptorSet;
pub const VkPipelineLayout = c.VkPipelineLayout;
pub const VkPipeline = c.VkPipeline;
pub const VkShaderModule = c.VkShaderModule;
pub const VkBuffer = c.VkBuffer;
pub const VkDeviceMemory = c.VkDeviceMemory;
pub const VkFence = c.VkFence;

pub const API_VERSION = c.VK_API_VERSION_1_3;
