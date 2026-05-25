prompt_template = [
    {
        "role": "system",
        "content": (
            "You extract the time constraint from a query. "
            "Return a single line only."
        ),
    },
    {
        "role": "user",
        "content": (
            "Given the query, output the time constraint, temporal ordering intent, and whether the query asks for a time.\n"
            "Return exactly one line in this format:\n"
            "time=YYYY-MM-DD; ordering=earliest|latest|none; time_request=yes|no\n\n"
            "Rules:\n"
            "- If a time constraint exists, return it as YYYY-MM-DD.\n"
            "- If the query only specifies a month or year, use the first day of that month or year.\n"
            "- If the query omits the year, use the year from the reference date.\n"
            "- Resolve relative expressions using the reference date.\n"
            "- If no time constraint exists, return time=NONE.\n"
            "- ordering=earliest for queries like \"earliest/first/oldest\".\n"
            "- ordering=latest for queries like \"latest/most recent/last time\".\n"
            "- ordering=none otherwise.\n\n"
            "- time_request=yes if the query asks for a time as the answer (e.g., when/what year/which year/what date).\n"
            "- time_request=no if the query only uses time as a constraint or does not ask for time.\n\n"
            "Reference date (UTC): ${reference_date}\n"
            "Query: ${query}"
        ),
    },
]
