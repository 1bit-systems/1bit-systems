// Thin wrappers around Tauri's invoke() for host-side commands defined
// in src-tauri/src/commands.rs. Each export here mirrors one rustdoc'd
// command on the Rust side.

import { invoke } from "@tauri-apps/api/core";

export interface ServiceStatus {
  readonly name: string;
  readonly active: boolean;
  readonly pid?: number;
}

// Read the contents of an in-tree runbook by relative path under
// cpp/agent/configs/runbooks/. Path is sanitized host-side.
export async function readRunbook(relPath: string): Promise<string> {
  return invoke<string>("read_runbook", { relPath });
}

// List the runbooks the host can see. Returns relative paths.
export async function listRunbooks(): Promise<string[]> {
  return invoke<string[]>("list_runbooks");
}

// systemctl --user is-active for the 1bit-halo-* unit family.
export async function serviceStatus(unit: string): Promise<ServiceStatus> {
  return invoke<ServiceStatus>("service_status", { unit });
}
