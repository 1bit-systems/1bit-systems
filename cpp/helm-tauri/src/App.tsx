import { useState, type JSX } from "react";
import { Mosaic, MosaicWindow } from "react-mosaic-component";
import type { MosaicNode } from "react-mosaic-component";
import { TILE_REGISTRY, type TileId } from "./tiles/registry";

// Initial 2x2 layout. Mosaic uses a binary tree of node ids — every
// `direction: "row" | "column"` node has `first` + `second` children.
//
//   ┌────────┬────────┐
//   │ halo   │ MCP    │
//   ├────────┼────────┤
//   │ bench  │ runbook│
//   └────────┴────────┘
const DEFAULT_LAYOUT: MosaicNode<TileId> = {
  direction: "column",
  splitPercentage: 50,
  first: {
    direction: "row",
    splitPercentage: 50,
    first: "halo",
    second: "mcp",
  },
  second: {
    direction: "row",
    splitPercentage: 50,
    first: "bench",
    second: "runbook",
  },
};

export function App(): JSX.Element {
  const [layout, setLayout] = useState<MosaicNode<TileId> | null>(
    DEFAULT_LAYOUT,
  );

  return (
    <div className="helm-root">
      <Mosaic<TileId>
        renderTile={(id, path) => {
          const tile = TILE_REGISTRY[id];
          return (
            <MosaicWindow<TileId>
              path={path}
              title={tile.title}
              toolbarControls={[]}
            >
              <tile.Component />
            </MosaicWindow>
          );
        }}
        value={layout}
        onChange={(next) => setLayout(next)}
        onRelease={(next) => setLayout(next)}
        className="mosaic-blueprint-theme bp5-dark"
      />
    </div>
  );
}
