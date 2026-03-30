import json
import asyncio
from pathlib import Path

import pandas as pd
from pydantic import BaseModel
from pydantic_ai import Agent

from logs import LOG_DIR


evaluation_prompt = """
Use this checklist to evaluate the quality of an AI agent's answer (<ANSWER>) to a user question (<QUESTION>).
We also include the log (<LOG>) for analysis.

For each item, check if the condition is met.

Checklist:

- instructions_follow: The agent followed the instructions in <INSTRUCTIONS>
- answer_relevant: The response directly addresses the user's question
- answer_clear: The answer is clear and correct
- answer_citations: The response includes citations or references when required
- completeness: The response is complete and covers the key aspects of the request
- tool_call_search: The search tool is invoked

Output true/false for each check and provide a short explanation for your judgment.
""".strip()


class EvaluationCheck(BaseModel):
    check_name: str
    justification: str
    check_pass: bool


class EvaluationChecklist(BaseModel):
    checklist: list[EvaluationCheck]
    summary: str


eval_agent = Agent(
    name="eval_agent",
    model="openai:gpt-4o-mini",
    instructions=evaluation_prompt,
    output_type=EvaluationChecklist,
)


user_prompt_format = """
<INSTRUCTIONS>{instructions}</INSTRUCTIONS>
<QUESTION>{question}</QUESTION>
<ANSWER>{answer}</ANSWER>
<LOG>{log}</LOG>
""".strip()


def load_log_file(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        data["log_file"] = str(path)
        return data


def simplify_log_messages(messages):
    log_simplified = []

    for m in messages:
        parts = []

        for original_part in m["parts"]:
            part = original_part.copy()
            kind = part.get("part_kind")

            if kind == "user-prompt":
                part.pop("timestamp", None)

            elif kind == "tool-call":
                part.pop("tool_call_id", None)

            elif kind == "tool-return":
                part.pop("tool_call_id", None)
                part.pop("metadata", None)
                part.pop("timestamp", None)
                part["content"] = "RETURN_RESULTS_REDACTED"

            elif kind == "text":
                part.pop("id", None)

            parts.append(part)

        message = {
            "kind": m["kind"],
            "parts": parts,
        }
        log_simplified.append(message)

    return log_simplified


def extract_question_answer(messages):
    """
    Extract the first user's question and the last text's answer from the log.
    """
    question = None
    answer = None

    for m in messages:
        for p in m.get("parts", []):
            if p.get("part_kind") == "user-prompt" and question is None:
                question = p.get("content")

    for m in reversed(messages):
        for p in m.get("parts", []):
            if p.get("part_kind") == "text":
                answer = p.get("content")
                break
        if answer is not None:
            break

    return question, answer


async def evaluate_log_record(eval_agent, log_record):
    messages = log_record["messages"]

    instructions = log_record["system_prompt"]
    question, answer = extract_question_answer(messages)

    if question is None:
        raise ValueError(f"Could not extract question from log: {log_record['log_file']}")
    if answer is None:
        raise ValueError(f"Could not extract answer from log: {log_record['log_file']}")

    log_simplified = simplify_log_messages(messages)
    log_json = json.dumps(log_simplified, ensure_ascii=False)

    user_prompt = user_prompt_format.format(
        instructions=instructions,
        question=question,
        answer=answer,
        log=log_json,
    )

    result = await eval_agent.run(user_prompt, output_type=EvaluationChecklist)
    return result.output


async def main():
    log_files = sorted(LOG_DIR.glob("*.json"))

    if not log_files:
        print("No log files found.")
        return

    eval_results = []

    for log_file in log_files:
        log_record = load_log_file(log_file)

        if log_record.get("agent_name") != "gh_agent":
            continue

        print(f"Evaluating: {log_file.name}")
        eval_result = await evaluate_log_record(eval_agent, log_record)
        eval_results.append((log_record, eval_result))

    if not eval_results:
        print("No matching gh_agent logs found.")
        return

    print(f"\nFinished evaluating {len(eval_results)} logs.")

    rows = []

    for log_record, eval_result in eval_results:
        messages = log_record["messages"]
        question, answer = extract_question_answer(messages)

        row = {
            "file": Path(log_record["log_file"]).name,
            "question": question,
            "answer": answer,
        }

        checks = {c.check_name: c.check_pass for c in eval_result.checklist}
        row.update(checks)

        rows.append(row)

    df_evals = pd.DataFrame(rows)

    print("\nDATAFRAME:")
    print(df_evals)

    print("\nMETRICS:")
    print(df_evals.mean(numeric_only=True))

    if "tool_call_search" in df_evals.columns:
        print("\nFAILED tool_call_search:")
        print(df_evals[df_evals["tool_call_search"] == False][["file", "question", "answer"]])

    if "answer_citations" in df_evals.columns:
        print("\nFAILED answer_citations:")
        print(df_evals[df_evals["answer_citations"] == False][["file", "question", "answer"]])

    if "completeness" in df_evals.columns:
        print("\nFAILED completeness:")
        print(df_evals[df_evals["completeness"] == False][["file", "question", "answer"]])


if __name__ == "__main__":
    asyncio.run(main())