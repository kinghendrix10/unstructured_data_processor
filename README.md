# Unstructured Data Processor

This package provides tools for processing unstructured data, extracting entities and relationships, and loading the structured data into a Neo4j database.

## Installation

```bash
pip install git+https://github.com/yourusername/unstructured_data_processor.git
```

## Usage

Here's a basic example of how to use the Unstructured Data Processor:

```python
import asyncio
from unstructured_data_processor import UnstructuredDataProcessor, Neo4jLoader
from llama_index.llms.groq import Groq

    # Initialize the LLM
    llm = Groq(model="llama3-8b-8192", api_key="your_groq_api_key", temperature=0)

    # Initialize the UnstructuredDataProcessor
    processor = UnstructuredDataProcessor(
        llm,
        rate_limit=60,  # 60 requests per minute
        time_period=60,  # 1 minute
        max_tokens=1000000  # 1 million tokens (adjust based on your plan)
    )

    # Customize the processor (optional)
    # processor.set_entity_types(["Person", "Organization", "Location", "Event"])
    # processor.set_output_format("json")
    # processor.set_batch_size(20)

    # Document folder
    input_directory = "/path/to/your/documents"

    # Process documents
    structured_data = await processor.restructure_documents(input_directory)

    # Initialize Neo4j loader and load data (optional)
    # neo4j_loader = Neo4jLoader("bolt://localhost:7687", "neo4j", "password")
    # neo4j_loader.load_data(structured_data)
    # neo4j_loader.close()

```

## Customization

The Unstructured Data Processor offers various customization options:

- `set_custom_prompt(prompt)`: Set a custom prompt for entity and relationship extraction.
- `set_entity_types(types)`: Define custom entity types to extract.
- `set_relationship_types(types)`: Define custom relationship types to extract.
- `add_preprocessing_step(function)`: Add a custom preprocessing step.
- `set_document_parser(function)`: Set a custom document parsing function.
- `update_rate_limit(rate, period)`: Update the rate limiting settings.
- `set_output_format(format)`: Set the output format (json, csv, or graphml).
- `set_logging_config(level, format)`: Configure logging.
- `set_llm_model(model_name, **kwargs)`: Switch to a different LLM model.
- `set_batch_size(size)`: Set the batch size for processing.
- `set_max_retries(retries)`: Set the maximum number of retries for failed API calls.
- `enable_progress_tracking(callback)`: Enable progress tracking with a custom callback.
- `add_pipeline_step(function, position)`: Add a custom step to the processing pipeline.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.
