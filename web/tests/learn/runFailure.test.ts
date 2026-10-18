import { describe, expect, it } from 'vitest';
import { CudaClientError } from '@/cuda/worker/protocol';
import { classifyRunFailure } from '@/learn/runFailure';

describe('classifyRunFailure', () => {
  it('a cancelled run is not an error to show', () => {
    expect(classifyRunFailure(new CudaClientError('cancelled', 'stopped'))).toEqual({ kind: 'cancelled' });
  });
  it('a crashed worker is a message with no line', () => {
    expect(classifyRunFailure(new CudaClientError('crashed', 'the kernel runner restarted'))).toEqual({ kind: 'crash', message: 'the kernel runner restarted' });
  });
  it('anything else is a crash with its message', () => {
    expect(classifyRunFailure(new Error('boom'))).toEqual({ kind: 'crash', message: 'boom' });
  });
});
