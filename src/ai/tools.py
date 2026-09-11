DATASET_METADATA_TOOL = {
    "type": "function",
    "name": "get_dataset_metadata",
    "description": (
        "Get basic information about the currently uploaded dataset, "
        "including the number of rows, number of columns, column names, "
        "and data types."
    ),
    "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False
    }
}

def get_dataset_metadata(df):
    metadata = {
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": df.columns.tolist(),
        "dtypes": {
            column: str(df[column].dtype)
            for column in df.columns
        }
    }

    return metadata