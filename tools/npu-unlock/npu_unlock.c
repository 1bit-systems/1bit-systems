// npu_unlock.c — Unlock AMD Strix Halo NPU from userspace
// Compile: gcc -O2 -o npu_unlock npu_unlock.c
// Run: sudo ./npu_unlock
//
// Patches the UEFI NVRAM variable that gates the NPU on AMI BIOS.
// No USB/SREP reboot needed. Run once, reboot, NPU stays unlocked.

#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <sys/utsname.h>

#define EFIVARFS "/sys/firmware/efi/efivars"

// Setup variable identifiers
static const char *SETUP_NAME  = "Setup";
static const char *SETUP_GUID  = "ec87d643-eba4-4bb5-a1e5-3f3e36b20da9";
static const char *AMD_SETUP   = "AmdSetup";
static const char *AMD_GUID    = "3a997502-647a-4c82-998e-52ef9486a247";
static const char *PBS_NAME    = "AMD_PBS_SETUP";
static const char *PBS_GUID    = "a339d746-f678-49b3-9fc7-54ce0f9df226";

// Known NPU unlock offsets (BIOS_version, var, offset, bit, value)
// Populated as discovered — contributions welcome
struct npu_offset {
    const char *bios_version;
    const char *var_name;
    const char *var_guid;
    uint16_t    offset;
    uint8_t     bit;     // 0xFF = whole byte
    uint8_t     locked;
    uint8_t     unlocked;
} known_offsets[] = {
    // TODO: Discover and add your BIOS version's offset here
    // { "1.07", "AmdSetup", AMD_GUID, 0x??, 0xFF, 0x00, 0x01 },
    { NULL, NULL, NULL, 0, 0, 0, 0 }  // sentinel
};

// Read an EFI variable. Returns malloc'd payload, sets *out_len.
static uint8_t *read_efivar(const char *name, const char *guid, size_t *out_len) {
    char path[256];
    snprintf(path, sizeof(path), "%s/%s-%s", EFIVARFS, name, guid);
    
    int fd = open(path, O_RDONLY);
    if (fd < 0) return NULL;
    
    struct stat st;
    if (fstat(fd, &st) < 0) { close(fd); return NULL; }
    
    uint8_t *buf = malloc(st.st_size);
    if (!buf) { close(fd); return NULL; }
    
    ssize_t n = read(fd, buf, st.st_size);
    close(fd);
    
    if (n < 5) { free(buf); return NULL; }  // need attrs + at least 1 byte
    
    *out_len = n - 4;  // skip 4-byte attribute header
    uint8_t *payload = malloc(*out_len);
    memcpy(payload, buf + 4, *out_len);
    free(buf);
    return payload;
}

// Write an EFI variable payload
static int write_efivar(const char *name, const char *guid, 
                         const uint8_t *payload, size_t len) {
    char path[256];
    snprintf(path, sizeof(path), "%s/%s-%s", EFIVARFS, name, guid);
    
    int fd = open(path, O_WRONLY);
    if (fd < 0) return -1;
    
    uint32_t attrs = 0x07;  // NV | BS | RT
    uint8_t header[4];
    memcpy(header, &attrs, 4);
    
    uint8_t *buf = malloc(4 + len);
    memcpy(buf, header, 4);
    memcpy(buf + 4, payload, len);
    
    ssize_t n = write(fd, buf, 4 + len);
    int err = (n < 0) ? -1 : 0;
    
    free(buf);
    close(fd);
    return err;
}

// Check if NPU is already unlocked
static int check_npu(void) {
    // Check lspci for NPU device
    FILE *fp = popen("lspci -nn 2>/dev/null | grep -q 'Neural Processing Unit'", "r");
    int pci_found = (pclose(fp) == 0);
    
    // Check amdxdna module
    fp = fopen("/proc/modules", "r");
    int mod_loaded = 0;
    if (fp) {
        char line[256];
        while (fgets(line, sizeof(line), fp))
            if (strstr(line, "amdxdna")) { mod_loaded = 1; break; }
        fclose(fp);
    }
    
    printf("NPU PCI:      %s\n", pci_found ? "✅ detected" : "❌ not found");
    printf("amdxdna:      %s\n", mod_loaded ? "✅ loaded" : "❌ not loaded");
    
    return pci_found && mod_loaded;
}

// Read BIOS version
static const char *get_bios_version(void) {
    static char version[64] = {0};
    if (version[0]) return version;
    FILE *fp = fopen("/sys/class/dmi/id/bios_version", "r");
    if (fp) {
        if (fgets(version, sizeof(version), fp))
            version[strcspn(version, "\n")] = 0;
        fclose(fp);
    }
    return version[0] ? version : "unknown";
}

// Try to find NPU offset by scanning for differences between two dumps
// (used when the offset is unknown)
static int scan_and_patch(void) {
    printf("No known offset for BIOS %s.\n", get_bios_version());
    printf("To discover: dump AmdSetup, disable NPU in BIOS, dump again, diff:\n");
    printf("  diff <(xxd dump1.bin) <(xxd dump2.bin)\n");
    return -1;
}

// Apply the NPU unlock
static int do_unlock(void) {
    if (check_npu()) {
        printf("NPU is already unlocked.\n");
        return 0;
    }
    
    const char *bios = get_bios_version();
    printf("BIOS: %s\n", bios);
    
    for (int i = 0; known_offsets[i].var_name; i++) {
        if (strcmp(known_offsets[i].bios_version, bios) != 0 &&
            strcmp(known_offsets[i].bios_version, "any") != 0)
            continue;
        
        struct npu_offset *o = &known_offsets[i];
        size_t len;
        uint8_t *payload = read_efivar(o->var_name, o->var_guid, &len);
        if (!payload) continue;
        
        if (o->offset >= len) { free(payload); continue; }
        
        printf("Offset: %s[0x%04x] = 0x%02x → 0x%02x\n",
               o->var_name, o->offset, payload[o->offset], o->unlocked);
        
        if (payload[o->offset] == o->unlocked) {
            printf("Already unlocked at this offset.\n");
            free(payload);
            return 0;
        }
        
        payload[o->offset] = o->unlocked;
        int ret = write_efivar(o->var_name, o->var_guid, payload, len);
        free(payload);
        
        if (ret == 0) {
            printf("✅ NPU unlock written! Reboot to apply.\n");
            return 0;
        }
        printf("❌ Write failed: %s\n", strerror(errno));
        return -1;
    }
    
    return scan_and_patch();
}

int main(int argc, char **argv) {
    if (geteuid() != 0) {
        fprintf(stderr, "❌ Must run as root\n");
        return 1;
    }
    
    struct stat st;
    if (stat(EFIVARFS, &st) != 0 || !S_ISDIR(st.st_mode)) {
        fprintf(stderr, "❌ efivarfs not mounted: mount -t efivarfs none %s\n", EFIVARFS);
        return 1;
    }
    
    const char *cmd = argc > 1 ? argv[1] : "status";
    
    if (strcmp(cmd, "status") == 0) {
        check_npu();
    } else if (strcmp(cmd, "unlock") == 0) {
        return do_unlock();
    } else if (strcmp(cmd, "dump") == 0) {
        // Dump AmdSetup to stdout as hex
        size_t len;
        uint8_t *p = read_efivar(AMD_SETUP, AMD_GUID, &len);
        if (!p) { fprintf(stderr, "Can't read AmdSetup\n"); return 1; }
        for (size_t i = 0; i < len; i++) {
            if (i % 16 == 0) printf("%04zx: ", i);
            printf("%02x%c", p[i], (i % 16 == 15) ? '\n' : ' ');
        }
        if (len % 16) printf("\n");
        free(p);
    } else {
        printf("Usage: npu_unlock [status|unlock|dump]\n");
    }
    return 0;
}
