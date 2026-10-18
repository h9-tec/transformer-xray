/** Observes every memory access and warp instruction. Task 5 fills this in;
 * for now it counts instructions and reports zeros elsewhere. */

import type { ThreadId } from './errors';
import type { LaunchStats, RaceReport } from './index';
import type { DeviceBuffer, Hooks } from './runtime';

export class Instrument implements Hooks {
  constructor(
    readonly enabled: boolean,
    readonly buffers: DeviceBuffer[],
  ) {}
  onWarpInstruction(_active: number): void {}
  onGlobal(_kind: 'load' | 'store' | 'atomic', _buf: DeviceBuffer, _addrs: number[], _line: number, _threads: ThreadId[]): void {}
  onShared(_kind: 'load' | 'store' | 'atomic', _addrs: number[], _line: number, _threads: ThreadId[]): void {}
  onBarrier(): void {}
  onBlockStart(_block: [number, number, number]): void {}
  stats(): LaunchStats {
    return {
      laneOps: 0,
      warpInstructions: 0,
      divergentInstructions: 0,
      globalRequests: 0,
      globalSectors: 0,
      globalBytes: 0,
      sharedRequests: 0,
      sharedWorstConflict: 0,
      atomics: 0,
      barriers: 0,
    };
  }
  races(): RaceReport[] {
    return [];
  }
}
