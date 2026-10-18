/** Observes every memory access and warp instruction and turns them into
 * the numbers a GPU engineer reads: sectors per request, bank conflicts,
 * divergence, and data races. Uses lib/memory so the landing hero and the
 * interpreter can never disagree. */

import { bankConflicts, sectorsTouched } from '@/lib/memory';
import type { ThreadId } from './errors';
import type { LaunchStats, RaceReport } from './index';
import { popcount, type DeviceBuffer, type Hooks } from './runtime';

type AccessKind = 'load' | 'store' | 'atomic';

interface Rec {
  thread: ThreadId;
  key: string;
  line: number;
  block: string;
  epoch: number;
  kind: AccessKind;
}

const tkey = (t: ThreadId) => `${t.block.join(',')}|${t.thread.join(',')}`;
const bkey = (t: ThreadId) => t.block.join(',');

const MAX_RACES = 50;

export class Instrument implements Hooks {
  private warpInstructions = 0;
  private divergent = 0;
  private globalRequests = 0;
  private globalSectors = 0;
  private sharedRequests = 0;
  private sharedWorst = 0;
  private epoch = 0;
  /** last writer / last reader per address: bufId → addr → rec */
  private writes = new Map<number, Map<number, Rec>>();
  private reads = new Map<number, Map<number, Rec>>();
  /** grouped by (space, buffer, firstLine, secondLine) */
  private groups = new Map<string, RaceReport>();
  private raceList: RaceReport[] = [];

  constructor(
    readonly enabled: boolean,
    readonly buffers: DeviceBuffer[],
  ) {}

  onBlockStart(): void {
    // shared memory is per block: forget its history, restart epochs
    for (const b of this.buffers) {
      if (!b?.shared) continue;
      this.writes.delete(b.id);
      this.reads.delete(b.id);
    }
    this.epoch = 0;
  }

  onBarrier(): void {
    this.epoch++;
  }

  onWarpInstruction(active: number, valid: number): void {
    if (!this.enabled) return;
    this.warpInstructions++;
    if (popcount(active) < popcount(valid)) this.divergent++;
  }

  onGlobal(kind: AccessKind, buf: DeviceBuffer, addrs: number[], line: number, threads: ThreadId[]): void {
    if (!this.enabled) return;
    this.globalRequests++;
    this.globalSectors += sectorsTouched(addrs);
    this.track('global', buf, kind, addrs, line, threads);
  }

  onShared(kind: AccessKind, buf: DeviceBuffer, addrs: number[], line: number, threads: ThreadId[]): void {
    if (!this.enabled) return;
    this.sharedRequests++;
    this.sharedWorst = Math.max(this.sharedWorst, bankConflicts(addrs));
    this.track('shared', buf, kind, addrs, line, threads);
  }

  private maps(buf: DeviceBuffer): { writes: Map<number, Rec>; reads: Map<number, Rec> } {
    let w = this.writes.get(buf.id);
    let r = this.reads.get(buf.id);
    if (!w) this.writes.set(buf.id, (w = new Map()));
    if (!r) this.reads.set(buf.id, (r = new Map()));
    return { writes: w, reads: r };
  }

  /** Race rule: two accesses to one address by different threads, not both
   * atomic, at least one a write, with no barrier between them (same block,
   * same epoch) or from different blocks (never ordered). */
  private track(
    space: 'shared' | 'global',
    buf: DeviceBuffer,
    kind: AccessKind,
    addrs: number[],
    line: number,
    threads: ThreadId[],
  ): void {
    const { writes, reads } = this.maps(buf);
    const bufName = buf.name;
    for (let i = 0; i < addrs.length; i++) {
      const a = addrs[i];
      const t = threads[i];
      const me: Rec = { thread: t, key: tkey(t), line, block: bkey(t), epoch: this.epoch, kind };
      const lastW = writes.get(a);
      if (lastW) this.maybeRace(space, bufName, a, lastW, me);
      if (kind !== 'load') {
        const lastR = reads.get(a);
        if (lastR) this.maybeRace(space, bufName, a, lastR, me);
        writes.set(a, me);
      } else {
        reads.set(a, me);
      }
    }
  }

  private maybeRace(space: 'shared' | 'global', bufName: string, addr: number, prev: Rec, cur: Rec): void {
    if (prev.key === cur.key) return;
    if (prev.kind === 'atomic' && cur.kind === 'atomic') return;
    if (prev.kind === 'load' && cur.kind === 'load') return;
    const unordered = prev.block !== cur.block || prev.epoch === cur.epoch;
    if (!unordered) return;
    // Source order, not execution order: warps are scheduled one after
    // another, so the "second" access can run first. A learner reads
    // "line 4 writes, line 5 reads" either way.
    const [a, b] = prev.line <= cur.line ? [prev, cur] : [cur, prev];
    const key = `${space}:${bufName}:${a.line}:${b.line}`;
    const g = this.groups.get(key);
    if (g) {
      g.count++;
      return;
    }
    if (this.raceList.length >= MAX_RACES) return;
    const report: RaceReport = {
      kind: space,
      address: addr,
      buffer: bufName,
      firstLine: a.line,
      secondLine: b.line,
      threads: [a.thread, b.thread],
      count: 1,
    };
    this.groups.set(key, report);
    this.raceList.push(report);
  }

  stats(): LaunchStats {
    return {
      laneOps: 0,
      warpInstructions: this.warpInstructions,
      divergentInstructions: this.divergent,
      globalRequests: this.globalRequests,
      globalSectors: this.globalSectors,
      globalBytes: this.globalSectors * 32,
      sharedRequests: this.sharedRequests,
      sharedWorstConflict: this.sharedWorst,
      atomics: 0,
      barriers: 0,
    };
  }

  races(): RaceReport[] {
    return this.raceList;
  }
}
