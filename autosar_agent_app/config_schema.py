# Release to folder name mapping
RELEASE_FOLDER_MAP = {
    "R4.4.0": "autosar_public_docs_4_4_0",
    "R4.3.1": "autosar_public_docs_4_3_1",  # Example for future
    "R4.2": "autosar_public_docs_4_2",      # Example for future
    "R4.1": "autosar_public_docs_4_1",      # Example for future
    "R4.0": "autosar_public_docs_4_0",      # Example for future
    # Add more mappings as needed
}

AVAILABLE_LLMS = {
    "groq": [
        "llama-3.1-8b-instant",
        "llama-3.1-70b-versatile",
        "qwen/qwen3-32b",
        "groq/compound",
        "moonshotai/kimi-k2-instruct-0905",
        "meta-llama/llama-prompt-guard-2-86m"
    ],
    "openai": [
        "gpt-4",
        "gpt-4-turbo-preview",
        "gpt-4o",
        "gpt-3.5-turbo",
        "gpt-3.5-turbo-16k"
    ],
    "ollama": [
        "gemma3",
        "llama2"
    ]
}

AVAILABLE_EMBEDDINGS = {
    "huggingface": [
        "all-MiniLM-L6-v2",
        "multi-qa-MiniLM-L6-cos-v1",
        "all-mpnet-base-v2"
        
    ],
    "openai": [
        "text-embedding-3-small",
        "text-embedding-3-large"
    ],
    "ollama": [
        "nomic-embed-text"
    ]
}

AUTOSAR_RELEASES = [
    "R4.0","R4.1","R4.2","R4.3.1",
    "R4.4.0","R29-11","R20-11","R21-11",
    "R22-11","R23-11","R24-11","R25-11"
]
