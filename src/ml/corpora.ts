export interface CorpusPreset {
  id: string;
  name: string;
  category: string;
  defaultPrompt: string;
  description: string;
  text: string;
}

export const CORPUS_PRESETS: CorpusPreset[] = [
  {
    id: 'shakespeare',
    name: 'Sonnets & Verse',
    category: 'Literature',
    defaultPrompt: 'Shall I compare ',
    description: 'Rhythmic Elizabethan verse with rich punctuation and meter.',
    text: `Shall I compare thee to a summer's day?
Thou art more lovely and more temperate:
Rough winds do shake the darling buds of May,
And summer's lease hath all too short a date:
Sometime too hot the eye of heaven shines,
And often is his gold complexion dimm'd;
And every fair from fair sometime declines,
By chance, or nature's changing course untrimm'd;
But thy eternal summer shall not fade,
Nor lose possession of that fair thou ow'st;
Nor shall death brag thou wander'st in his shade,
When in eternal lines to time thou grow'st:
So long as men can breathe, or eyes can see,
So long lives this, and this gives life to thee.

When in disgrace with fortune and men's eyes,
I all alone beweep my outcast state,
And trouble deaf heaven with my bootless cries,
And look upon myself, and curse my fate,
Wishing me like to one more rich in hope,
Featur'd like him, like him with friends possess'd,
Desiring this man's art, and that man's scope,
With what I most enjoy contented least;
Yet in these thoughts myself almost despising,
Haply I think on thee, and then my state,
Like to the lark at break of day arising
From sullen earth, sings hymns at heaven's gate;
For thy sweet love remember'd such wealth brings
That then I scorn to change my state with kings.`,
  },
  {
    id: 'python',
    name: 'Python & Logic',
    category: 'Code',
    defaultPrompt: 'def step(',
    description: 'Indented function blocks, loops, return clauses, and brackets.',
    text: `def step(x, w, b):
    y = x * w + b
    if y > 0:
        return y
    return 0

def norm(vec):
    total = 0
    for val in vec:
        total += val * val
    return total

def attend(q, k, v):
    score = dot(q, k)
    prob = softmax(score)
    return prob * v

def train(data, steps):
    loss = 1.0
    for i in range(steps):
        loss = loss * 0.95
        if loss < 0.1:
            return loss
    return loss

def dot(a, b):
    out = 0
    for i in range(len(a)):
        out += a[i] * b[i]
    return out
`,
  },
  {
    id: 'arithmetic',
    name: 'Arithmetic Rules',
    category: 'Symbolic',
    defaultPrompt: '12 + 15 = ',
    description: 'Structured equations and sequential numeric identities.',
    text: `0 + 0 = 0
1 + 1 = 2
2 + 3 = 5
4 + 5 = 9
5 + 7 = 12
6 + 8 = 14
9 + 3 = 12
10 + 10 = 20
11 + 12 = 23
12 + 15 = 27
14 + 16 = 30
18 + 11 = 29
20 + 25 = 45
21 + 14 = 35
30 + 12 = 42
2 * 3 = 6
3 * 4 = 12
4 * 5 = 20
5 * 6 = 30
6 * 7 = 42
7 * 8 = 56
8 * 9 = 72
9 * 9 = 81
10 - 4 = 6
15 - 7 = 8
20 - 9 = 11
25 - 12 = 13
32 - 16 = 16
`,
  },
  {
    id: 'observatory',
    name: 'Stellar Catalog',
    category: 'Astronomy',
    defaultPrompt: 'HD 189733 ',
    description: 'Star designations, spectral classes, and orbital telemetry.',
    text: `HD 189733 b · K2V · 64.5 ly · period 2.22 d
HD 209458 b · G0V · 159.0 ly · period 3.52 d
Kepler-186 f · M1V · 582.0 ly · period 129.9 d
Kepler-442 b · K5V · 1206.0 ly · period 112.3 d
TRAPPIST-1 e · M8V · 40.7 ly · period 6.10 d
TRAPPIST-1 f · M8V · 40.7 ly · period 9.21 d
Proxima Cen b · M5V · 4.24 ly · period 11.19 d
Tau Ceti e · G8V · 11.9 ly · period 162.9 d
Sirius A · A1V · 8.60 ly · mag -1.46
Vega Lyr · A0V · 25.0 ly · mag 0.03
Altair Aql · A7V · 16.7 ly · mag 0.76
Rigel Ori · B8Ia · 863.0 ly · mag 0.13
Betelgeuse · M1Ia · 548.0 ly · mag 0.42
Aldebaran · K5III · 65.3 ly · mag 0.86
`,
  },
];
