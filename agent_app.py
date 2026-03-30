import asyncio

import ingest
import search_agent
import logs


REPO_OWNER = "luongnv89"
REPO_NAME = "claude-howto"


def initialize_index():
    print(f"Starting AI Doc Assistant for {REPO_OWNER}/{REPO_NAME}")
    print("Initializing data ingestion and indexing...")

    index, docs = ingest.index_data(
        REPO_OWNER,
        REPO_NAME,
        filter_func=None,
        chunk=True,
        chunking_params={"size": 2000, "step": 1000},
    )

    print(f"Indexed {len(docs)} records successfully.")
    return index, docs


def initialize_agent(index, docs):
    print("Initializing search agent...")
    agent = search_agent.init_agent(index, docs, REPO_OWNER, REPO_NAME)
    print("Agent initialized successfully!")
    return agent


def main():
    index, docs = initialize_index()
    agent = initialize_agent(index, docs)

    print("\nReady to answer your questions!")
    print("Type 'stop' to exit the program.\n")

    while True:
        question = input("Your question: ").strip()

        if question.lower() == "stop":
            print("Goodbye!")
            break

        if not question:
            continue

        print("Processing your question...")
        response = asyncio.run(agent.run(user_prompt=question))

        logs.log_interaction_to_file(agent, response.new_messages(), source="user")

        print("\nResponse:\n")
        print(response.output)
        print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()