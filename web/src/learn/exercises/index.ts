/** Every lesson's exercise, keyed by lesson slug. */

import type { Exercise } from '../exercise';
import { helloThread } from './hello-thread';

export const EXERCISES: Record<string, Exercise> = {
  [helloThread.slug]: helloThread,
};
