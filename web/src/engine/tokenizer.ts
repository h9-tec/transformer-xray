/** Character-level tokenizer over the Tiny Shakespeare charset (65 chars). */

export class CharTokenizer {
  readonly chars: string;
  private stoi = new Map<string, number>();

  constructor(chars: string) {
    this.chars = chars;
    for (let i = 0; i < chars.length; i++) this.stoi.set(chars[i], i);
  }

  get vocabSize(): number {
    return this.chars.length;
  }

  /** Throws with the offending character and position — the UI turns this
   * into an inline "this character isn't in the vocabulary" hint. */
  encode(text: string): Int32Array {
    const ids = new Int32Array(text.length);
    for (let i = 0; i < text.length; i++) {
      const id = this.stoi.get(text[i]);
      if (id === undefined) {
        throw new UnknownCharError(text[i], i);
      }
      ids[i] = id;
    }
    return ids;
  }

  decode(ids: ArrayLike<number>): string {
    let out = '';
    for (let i = 0; i < ids.length; i++) out += this.chars[ids[i]];
    return out;
  }
}

export class UnknownCharError extends Error {
  constructor(
    readonly char: string,
    readonly position: number,
  ) {
    super(`character ${JSON.stringify(char)} at position ${position} is not in the vocabulary`);
    this.name = 'UnknownCharError';
  }
}
