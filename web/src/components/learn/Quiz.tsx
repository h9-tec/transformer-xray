'use client';

/** Three questions, graded only when the learner commits. Explanations
 * appear after checking, right or wrong. The score is stored separately
 * from exercise progress. */

import { useTranslations } from 'next-intl';
import { useState } from 'react';
import { QUIZZES } from '@/learn/quizzes';
import { useLearnStore } from '@/stores/learnStore';

export function Quiz({ slug }: { slug: string }) {
  const t = useTranslations('learn');
  const quiz = QUIZZES[slug];
  const setQuiz = useLearnStore((s) => s.setQuiz);
  const [picked, setPicked] = useState<(number | null)[]>(() => quiz.questions.map(() => null));
  const [checked, setChecked] = useState(false);
  if (!quiz) return null;
  const score = quiz.questions.reduce((a, q, i) => a + (picked[i] === q.answer ? 1 : 0), 0);
  const allPicked = picked.every((p) => p !== null);

  return (
    <div className="panel p-5" data-testid="quiz" lang="en" dir="ltr">
      <ol className="m-0 flex list-none flex-col gap-6 p-0">
        {quiz.questions.map((q, qi) => {
          const correct = checked && picked[qi] === q.answer;
          const wrong = checked && picked[qi] !== null && picked[qi] !== q.answer;
          return (
            <li key={qi}>
              <p className="m-0 mb-2.5 text-[17px] font-semibold text-ink">
                <span className="nums me-2 text-faint">{qi + 1}.</span>
                {q.q}
              </p>
              <div className="flex flex-col gap-2">
                {q.choices.map((c, ci) => {
                  const isPicked = picked[qi] === ci;
                  const isAnswer = checked && ci === q.answer;
                  return (
                    <label
                      key={ci}
                      className={`flex cursor-pointer items-center gap-3 rounded-[10px] border px-3.5 py-2.5 text-[15px] ${
                        isAnswer
                          ? 'border-ok/50 bg-ok/10'
                          : isPicked && wrong
                            ? 'border-danger/50 bg-danger/10'
                            : 'border-line bg-surface hover:border-fwd/40'
                      }`}
                    >
                      <input
                        type="radio"
                        name={`${slug}-q${qi}`}
                        checked={isPicked}
                        disabled={checked}
                        onChange={() => setPicked((p) => p.map((v, i) => (i === qi ? ci : v)))}
                        className="accent-[#74c7ff]"
                      />
                      <span className="nums">{c}</span>
                    </label>
                  );
                })}
              </div>
              {checked && (
                <p
                  data-testid={`quiz-why-${qi}`}
                  className={`m-0 mt-2.5 rounded-[10px] border border-dashed px-3.5 py-2.5 text-[14px] ${correct ? 'border-ok/40 text-muted' : 'border-accent/40 bg-accent/5 text-muted'}`}
                >
                  <b className={correct ? 'text-ok' : 'text-accent'}>{correct ? t('quizRight') : t('quizWrong')} </b>
                  {q.why}
                </p>
              )}
            </li>
          );
        })}
      </ol>
      <div className="mt-5 flex flex-wrap items-center gap-3">
        {!checked ? (
          <button
            type="button"
            disabled={!allPicked}
            onClick={() => {
              setChecked(true);
              setQuiz(slug, score);
            }}
            className="btn-primary !py-2 disabled:opacity-40"
          >
            {t('checkAnswers')}
          </button>
        ) : (
          <>
            <span className="nums text-[15px] font-semibold" data-testid="quiz-score">
              {t('quizScore', { k: score, n: quiz.questions.length })}
            </span>
            <button
              type="button"
              onClick={() => {
                setChecked(false);
                setPicked(quiz.questions.map(() => null));
              }}
              className="rounded-full border border-line px-3 py-1 text-[13px] text-muted hover:text-ink"
            >
              {t('quizRetry')}
            </button>
          </>
        )}
      </div>
    </div>
  );
}
