'use client';

/** CodeMirror 6 editor for CUDA source, themed from the design tokens.
 * `errorLine` paints one line; `id` registers a tiny test hook on
 * window.__cm so e2e can set text without typing into the editor. */

import { cpp } from '@codemirror/lang-cpp';
import { HighlightStyle, syntaxHighlighting } from '@codemirror/language';
import { EditorState, StateEffect, StateField } from '@codemirror/state';
import { Decoration, EditorView, type DecorationSet } from '@codemirror/view';
import { tags } from '@lezer/highlight';
import { basicSetup } from 'codemirror';
import { useEffect, useRef } from 'react';

const setErrorLine = StateEffect.define<number | null>();
const errorLineField = StateField.define<DecorationSet>({
  create: () => Decoration.none,
  update(deco, tr) {
    deco = deco.map(tr.changes);
    for (const e of tr.effects) {
      if (!e.is(setErrorLine)) continue;
      if (e.value === null || e.value < 1 || e.value > tr.state.doc.lines) return Decoration.none;
      const line = tr.state.doc.line(e.value);
      return Decoration.set([Decoration.line({ class: 'cm-error-line' }).range(line.from)]);
    }
    return deco;
  },
  provide: (f) => EditorView.decorations.from(f),
});

const theme = EditorView.theme(
  {
    '&': { backgroundColor: '#080b14', color: '#e8ecf7', fontSize: '13.5px', borderRadius: '0 0 18px 18px' },
    '.cm-content': { fontFamily: 'var(--font-mono)', padding: '12px 0', caretColor: '#ff9a4d' },
    '.cm-scroller': { fontFamily: 'var(--font-mono)', lineHeight: '1.6', maxHeight: '480px' },
    '.cm-gutters': { backgroundColor: '#080b14', color: '#5f6a85', border: 'none', paddingLeft: '6px' },
    '.cm-activeLine': { backgroundColor: 'rgba(255,255,255,0.025)' },
    '.cm-activeLineGutter': { backgroundColor: 'transparent', color: '#9aa5bf' },
    '&.cm-focused .cm-cursor': { borderLeftColor: '#ff9a4d' },
    '&.cm-focused .cm-selectionBackground, .cm-selectionBackground': { backgroundColor: 'rgba(116,199,255,0.18)' },
    '.cm-error-line': { backgroundColor: 'rgba(255,107,122,0.14)', boxShadow: 'inset 3px 0 0 #ff6b7a' },
    '&.cm-focused': { outline: 'none' },
    '.cm-matchingBracket': { backgroundColor: 'rgba(255,154,77,0.2)', outline: 'none' },
  },
  { dark: true },
);

const highlight = HighlightStyle.define([
  { tag: [tags.keyword, tags.modifier, tags.controlKeyword, tags.definitionKeyword], color: '#c4a6ff' },
  { tag: [tags.typeName, tags.standard(tags.typeName)], color: '#74c7ff' },
  { tag: [tags.number, tags.bool], color: '#ffd479' },
  { tag: [tags.comment, tags.lineComment, tags.blockComment], color: '#5f6a85', fontStyle: 'italic' },
  { tag: [tags.function(tags.variableName), tags.function(tags.definition(tags.variableName))], color: '#ff9a4d' },
  { tag: [tags.processingInstruction, tags.macroName], color: '#ff7ab6' },
  { tag: tags.string, color: '#5ee6a8' },
  { tag: tags.operator, color: '#9aa5bf' },
]);

export interface EditorProps {
  id: string;
  value: string;
  onChange: (text: string) => void;
  errorLine: number | null;
}

declare global {
  interface Window {
    __cm?: Record<string, { setText(text: string): void; getText(): string }>;
  }
}

export function Editor({ id, value, onChange, errorLine }: EditorProps) {
  const host = useRef<HTMLDivElement>(null);
  const view = useRef<EditorView | null>(null);
  const onChangeRef = useRef(onChange);
  onChangeRef.current = onChange;

  useEffect(() => {
    if (!host.current) return;
    const v = new EditorView({
      state: EditorState.create({
        doc: value,
        extensions: [
          basicSetup,
          cpp(),
          theme,
          syntaxHighlighting(highlight),
          errorLineField,
          EditorView.updateListener.of((u) => {
            if (u.docChanged) onChangeRef.current(u.state.doc.toString());
          }),
        ],
      }),
      parent: host.current,
    });
    view.current = v;
    window.__cm ??= {};
    window.__cm[id] = {
      setText: (text) => v.dispatch({ changes: { from: 0, to: v.state.doc.length, insert: text } }),
      getText: () => v.state.doc.toString(),
    };
    return () => {
      delete window.__cm?.[id];
      v.destroy();
      view.current = null;
    };
    // mount once; value changes from outside go through setValue below
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [id]);

  // external value changes (reset / show solution)
  useEffect(() => {
    const v = view.current;
    if (!v) return;
    if (v.state.doc.toString() !== value) {
      v.dispatch({ changes: { from: 0, to: v.state.doc.length, insert: value } });
    }
  }, [value]);

  useEffect(() => {
    view.current?.dispatch({ effects: setErrorLine.of(errorLine) });
  }, [errorLine, value]);

  return <div ref={host} data-testid="editor" className="min-w-0 overflow-hidden text-start" dir="ltr" />;
}
