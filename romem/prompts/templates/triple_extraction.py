from .ner import one_shot_ner_paragraph, one_shot_ner_output
from ...utils.llm_utils import convert_format_to_template

ner_conditioned_re_system = """Your task is to construct an RDF (Resource Description Framework) graph from the given passages and named entity lists.
Respond with JSON where each triple entry also carries timing metadata.

Requirements:
- Represent each triple as an object with fields: head, relation, tail, text_time, observed_time.
- text_time must be a normalized date string in YYYY-MM-DD, or "" if no time is mentioned.
- If only a month or year is mentioned, use the first day of that month or year.
- If the year is missing, use the year from observed_time.
- Resolve relative expressions using the closest explicit date mentioned in the passage. If none exists, use observed_time.
- observed_time is always set to the provided observed_time string.
- Do not infer a time if the passage does not contain a temporal expression.
- Do NOT use time expressions as head or tail entities; keep time only in text_time.
- Each triple should contain at least one, but preferably two, of the named entities in the list.
- Resolve pronouns to their specific names.
- Extract all factual relations that appear in the passage, including actions, states, plans, and events.
- Do not omit facts just because they seem minor; prioritize completeness over brevity.
- Do not invent people or relationships that are not mentioned.
- Duration handling: if the passage states a duration such as "for 5 years" and an explicit reference date is present,
  infer the start date and put it in text_time. Keep the tail as a non-time concept (e.g., "marriage", "employment", "residence").
"""


ner_conditioned_re_frame = """Convert the paragraph into a JSON dict with a named entity list and a triple list.
Paragraph:
```
{passage}
```

{named_entity_json}

observed_time: {observed_time}
"""


ner_conditioned_re_input = ner_conditioned_re_frame.format(
    passage=one_shot_ner_paragraph,
    named_entity_json=one_shot_ner_output,
    observed_time="{observed_time}",
)


ner_conditioned_re_output = """{
  "triples": [
    {"head": "Radio City", "relation": "located in", "tail": "India", "text_time": "", "observed_time": "2024-01-01T00:00:00Z"},
    {"head": "Radio City", "relation": "started on", "tail": "private FM radio station", "text_time": "2001-07-03", "observed_time": "2024-01-01T00:00:00Z"},
    {"head": "Radio City", "relation": "plays songs in", "tail": "Hindi", "text_time": "", "observed_time": "2024-01-01T00:00:00Z"},
    {"head": "PlanetRadiocity.com", "relation": "launched in", "tail": "New Media", "text_time": "2008-05-01", "observed_time": "2024-01-01T00:00:00Z"}
  ]
}"""

example_paragraph_2 = """Project Atlas partnered with Northwind in March 2021. Last year, Project Atlas built a prototype in Berlin."""

example_named_entity_json_2 = """{
  "named_entities": ["Project Atlas", "Northwind", "prototype", "Berlin"]
}"""

example_observed_time_2 = "2024-06-15T00:00:00Z"

example_input_2 = ner_conditioned_re_frame.format(
    passage=example_paragraph_2,
    named_entity_json=example_named_entity_json_2,
    observed_time=example_observed_time_2,
)

example_output_2 = """{
  "triples": [
    {"head": "Project Atlas", "relation": "partnered with", "tail": "Northwind", "text_time": "2021-03-01", "observed_time": "2024-06-15T00:00:00Z"},
    {"head": "Project Atlas", "relation": "built", "tail": "prototype", "text_time": "2023-06-15", "observed_time": "2024-06-15T00:00:00Z"},
    {"head": "Project Atlas", "relation": "built in", "tail": "Berlin", "text_time": "2023-06-15", "observed_time": "2024-06-15T00:00:00Z"}
  ]
}"""

example_paragraph_3 = """On May 9, 2023, Alex said they had been married for 5 years and were happy about it."""

example_named_entity_json_3 = """{
  "named_entities": ["Alex"]
}"""

example_observed_time_3 = "2026-01-01T00:00:00Z"

example_input_3 = ner_conditioned_re_frame.format(
    passage=example_paragraph_3,
    named_entity_json=example_named_entity_json_3,
    observed_time=example_observed_time_3,
)

example_output_3 = """{
  "triples": [
    {"head": "Alex", "relation": "is in", "tail": "marriage", "text_time": "2018-05-09", "observed_time": "2026-01-01T00:00:00Z"}
  ]
}"""


prompt_template = [
    {"role": "system", "content": ner_conditioned_re_system},
    {"role": "user", "content": ner_conditioned_re_input},
    {"role": "assistant", "content": ner_conditioned_re_output},
    {"role": "user", "content": example_input_2},
    {"role": "assistant", "content": example_output_2},
    {"role": "user", "content": example_input_3},
    {"role": "assistant", "content": example_output_3},
    {"role": "user", "content": convert_format_to_template(original_string=ner_conditioned_re_frame, placeholder_mapping=None, static_values=None)}
]
