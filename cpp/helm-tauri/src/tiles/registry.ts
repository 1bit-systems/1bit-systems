import type { ComponentType } from "react";
import { HaloTile } from "./HaloTile";
import { MCPTile } from "./MCPTile";
import { BenchTile } from "./BenchTile";
import { RunbookTile } from "./RunbookTile";

export type TileId = "halo" | "mcp" | "bench" | "runbook";

export interface TileDescriptor {
  readonly title: string;
  readonly Component: ComponentType;
}

// Mosaic dispatches by string id; this map is the single source of
// truth for the tile -> component binding. Adding a new tile = adding a
// row here + extending the TileId union above.
export const TILE_REGISTRY: Record<TileId, TileDescriptor> = {
  halo: { title: "halo", Component: HaloTile },
  mcp: { title: "MCP", Component: MCPTile },
  bench: { title: "bench", Component: BenchTile },
  runbook: { title: "runbook", Component: RunbookTile },
};
