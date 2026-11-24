export interface CharTokenizer {
  vocab: string[];
  tokenToId: Map<string, number>;
  idToToken: string[];
  frequencies: number[];
  vocabSize: number;
}

export function formatTokenLabel(token: string): string {
  if (token === ' ') return '␣';
  if (token === '\n') return '↵';
  if (token === '\t') return '⇥';
  if (token === '\r') return '⏎';
  return token;
}

export function describeToken(token: string): string {
  if (token === ' ') return 'SPACE';
  if (token === '\n') return 'NEWLINE';
  if (token === '\t') return 'TAB';
  return `'${token}'`;
}

export function buildTokenizer(text: string): CharTokenizer {
  const counts = new Map<string, number>();
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    counts.set(ch, (counts.get(ch) ?? 0) + 1);
  }

  // Ensure a minimal fallback vocabulary if user clears custom text temporarily
  if (counts.size === 0) {
    for (const ch of 'abcdefghijklmnopqrstuvwxyz ') {
      counts.set(ch, 1);
    }
  }

  const vocab = Array.from(counts.keys()).sort((a, b) => {
    const fa = counts.get(a) ?? 0;
    const fb = counts.get(b) ?? 0;
    if (fb !== fa) return fb - fa;
    return a.localeCompare(b);
  });

  const tokenToId = new Map<string, number>();
  const frequencies: number[] = [];
  const total = Math.max(1, text.length);

  vocab.forEach((ch, idx) => {
    tokenToId.set(ch, idx);
    frequencies.push((counts.get(ch) ?? 0) / total);
  });

  return {
    vocab,
    tokenToId,
    idToToken: vocab,
    frequencies,
    vocabSize: vocab.length,
  };
}

export function encodeText(tokenizer: CharTokenizer, text: string): number[] {
  const ids: number[] = [];
  for (let i = 0; i < text.length; i++) {
    const id = tokenizer.tokenToId.get(text[i]);
    if (id !== undefined) {
      ids.push(id);
    }
  }
  return ids;
}
